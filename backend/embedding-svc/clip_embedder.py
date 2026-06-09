import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_MODEL   = "openai/clip-vit-base-patch32"
EMBEDDING_DIM   = 512
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


# =============================================================================
# CLIPEmbedder
# =============================================================================

class CLIPEmbedder:
    """
    Generates L2-normalized CLIP embeddings for images and/or text.

    Both modalities are projected into the same 512-dim space, so their
    embeddings are directly comparable via cosine similarity.

    Parameters
    ----------
    model_name  : HuggingFace model ID. Default: 'openai/clip-vit-base-patch32'
    batch_size  : Number of items processed per forward pass.
    device      : 'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Examples
    --------
    >>> embedder = CLIPEmbedder()
    >>> img_emb  = embedder.embed_images(["cat.jpg", "dog.png"])   # (2, 512)
    >>> txt_emb  = embedder.embed_texts(["a cat", "a dog"])        # (2, 512)
    >>> sim      = img_emb @ txt_emb.T                             # cosine sim
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str = None,
    ):
        # Device selection
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[CLIPEmbedder] Loading '{model_name}' on {self.device}…")
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.batch_size = batch_size
        print(f"[CLIPEmbedder] Ready. Embedding dim: {EMBEDDING_DIM}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization."""
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norms

    def _load_image(self, path: str) -> Image.Image:
        """Loads an image from disk, converts to RGB."""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Could not load image '{path}': {e}. Replacing with blank image.")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_images(self, paths: list[str]) -> np.ndarray:
        """
        Generates CLIP image embeddings from a list of file paths.

        Parameters
        ----------
        paths : List of local image file paths.

        Returns
        -------
        np.ndarray of shape (n, 512), L2-normalized.
        """
        if not paths:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(paths), self.batch_size):
            batch_paths  = paths[start : start + self.batch_size]
            batch_images = [self._load_image(p) for p in batch_paths]

            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.get_image_features(**inputs)

            # get_image_features may return a raw tensor or a ModelOutput object
            # depending on the transformers version — handle both cases
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                features = output.pooler_output
            else:
                features = output.last_hidden_state[:, 0, :]

            all_embeddings.append(features.detach().cpu().float().numpy())

            end = min(start + self.batch_size, len(paths))
            print(f"   [images] {end}/{len(paths)} embedded…")

        embeddings = np.vstack(all_embeddings)
        return self._l2_normalize(embeddings)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generates CLIP text embeddings from a list of strings.

        CLIP's text encoder is limited to 77 tokens; longer strings are
        automatically truncated by the processor.

        Parameters
        ----------
        texts : List of strings.

        Returns
        -------
        np.ndarray of shape (n, 512), L2-normalized.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        # Replace empty strings to avoid tokenizer errors
        safe_texts   = [t if str(t).strip() else " " for t in texts]
        all_embeddings = []

        for start in range(0, len(safe_texts), self.batch_size):
            batch = safe_texts[start : start + self.batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.get_text_features(**inputs)

            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                features = output.pooler_output
            else:
                features = output.last_hidden_state[:, 0, :]

            all_embeddings.append(features.detach().cpu().float().numpy())

            end = min(start + self.batch_size, len(safe_texts))
            print(f"   [texts] {end}/{len(safe_texts)} embedded…")

        embeddings = np.vstack(all_embeddings)
        return self._l2_normalize(embeddings)

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        image_cols: list[str] = None,
        text_cols:  list[str] = None,
    ) -> np.ndarray:
        """
        Convenience method: embeds image and/or text columns from a DataFrame
        and returns a single fused matrix (horizontal concatenation).

        Each modality block is L2-normalized independently before concatenation,
        so neither dominates by magnitude.

        Parameters
        ----------
        df         : Input DataFrame.
        image_cols : Column names containing local image file paths.
        text_cols  : Column names containing text strings.

        Returns
        -------
        np.ndarray of shape (n_rows, total_dim)
          - image only  → (n, 512)
          - text only   → (n, 512)
          - both        → (n, 1024)
        """
        blocks = []

        if image_cols:
            for col in image_cols:
                print(f"\n[embed_dataframe] Processing image column: '{col}'")
                paths = df[col].fillna("").astype(str).tolist()
                blocks.append(self.embed_images(paths))

        if text_cols:
            print(f"\n[embed_dataframe] Processing text columns: {text_cols}")
            merged = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
            blocks.append(self.embed_texts(merged))

        if not blocks:
            raise ValueError("Provide at least one of: image_cols, text_cols.")

        fused = np.hstack(blocks)
        print(f"\n[embed_dataframe] Output shape: {fused.shape}")
        return fused

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_embeddings(
        embeddings: np.ndarray,
        output_path: str,
        ids: list = None,
        prefix: str = "emb",
    ) -> pd.DataFrame:
        """
        Saves embeddings to a CSV file.

        Each embedding dimension becomes a column named '{prefix}_0',
        '{prefix}_1', ..., '{prefix}_N'. An optional 'id' column is
        prepended when ids are provided (e.g. file names, row indices).

        Parameters
        ----------
        embeddings  : np.ndarray of shape (n, dim).
        output_path : Destination CSV path (e.g. 'embeddings.csv').
        ids         : Optional list of identifiers (file names, labels, etc.).
                      Length must match embeddings.shape[0].
        prefix      : Column name prefix for embedding dimensions.

        Returns
        -------
        pd.DataFrame with the saved content.

        Example
        -------
        >>> CLIPEmbedder.save_embeddings(img_emb, "img_emb.csv", ids=image_paths)
        """
        n, dim = embeddings.shape
        col_names = [f"{prefix}_{i}" for i in range(dim)]
        df = pd.DataFrame(embeddings, columns=col_names)

        if ids is not None:
            if len(ids) != n:
                raise ValueError(
                    f"ids length ({len(ids)}) does not match "
                    f"embeddings rows ({n})."
                )
            df.insert(0, "id", ids)

        df.to_csv(output_path, index=False)
        print(f"[save_embeddings] Saved {n} embeddings ({dim} dims) -> '{output_path}'")
        return df

    @staticmethod
    def load_embeddings(
        csv_path: str,
        prefix: str = "emb",
        return_ids: bool = False,
    ):
        """
        Loads embeddings previously saved with save_embeddings().

        Parameters
        ----------
        csv_path   : Path to the CSV file.
        prefix     : Column prefix used when saving (default: 'emb').
        return_ids : If True, also returns the 'id' column as a list.

        Returns
        -------
        embeddings : np.ndarray of shape (n, dim)
        ids        : list of ids — only returned when return_ids=True

        Example
        -------
        >>> emb = CLIPEmbedder.load_embeddings("img_emb.csv")
        >>> emb, ids = CLIPEmbedder.load_embeddings("img_emb.csv", return_ids=True)
        """
        df = pd.read_csv(csv_path)
        emb_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]

        if not emb_cols:
            raise ValueError(
                f"No columns with prefix '{prefix}_' found in '{csv_path}'. "
                f"Available columns: {list(df.columns)}"
            )

        embeddings = df[emb_cols].values.astype(np.float32)
        print(
            f"[load_embeddings] Loaded {embeddings.shape[0]} embeddings "
            f"({embeddings.shape[1]} dims) from '{csv_path}'"
        )

        if return_ids:
            ids = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
            return embeddings, ids

        return embeddings


# =============================================================================
# Standalone utility functions (no class instantiation needed for quick use)
# =============================================================================

def embed_images_from_folder(
    folder: str,
    batch_size: int = 64,
    extensions: set[str] = IMAGE_EXTS,
) -> tuple[np.ndarray, list[str]]:
    """
    Embeds all images found in a folder.

    Returns
    -------
    embeddings : np.ndarray of shape (n_images, 512)
    file_names : list of file names in the same order as embeddings
    """
    folder_path = Path(folder)
    paths = sorted([
        str(p) for p in folder_path.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not paths:
        raise FileNotFoundError(f"No images found in '{folder}'.")

    print(f"[embed_images_from_folder] Found {len(paths)} images in '{folder}'.")
    embedder   = CLIPEmbedder(batch_size=batch_size)
    embeddings = embedder.embed_images(paths)
    file_names = [Path(p).name for p in paths]
    return embeddings, file_names


def embed_csv_column(
    csv_path: str,
    column: str,
    modality: str,
    output_path: str = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Reads a CSV, embeds one column, and optionally saves embeddings to .npy.

    Parameters
    ----------
    csv_path    : Path to the CSV file.
    column      : Name of the column to embed.
    modality    : 'image' (column contains file paths) or 'text'.
    output_path : If provided, saves the array as a .npy file.
    batch_size  : Batch size for inference.

    Returns
    -------
    np.ndarray of shape (n_rows, 512)
    """
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in '{csv_path}'.")

    embedder = CLIPEmbedder(batch_size=batch_size)
    values   = df[column].fillna("").astype(str).tolist()

    if modality == "image":
        embeddings = embedder.embed_images(values)
    elif modality == "text":
        embeddings = embedder.embed_texts(values)
    else:
        raise ValueError("modality must be 'image' or 'text'.")

    if output_path:
        np.save(output_path, embeddings)
        print(f"[embed_csv_column] Embeddings saved to '{output_path}'.")

    return embeddings