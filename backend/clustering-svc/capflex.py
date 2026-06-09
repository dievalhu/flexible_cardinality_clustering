# -----------------------------------------------------------------------------
# 0. IMPORTS & DEPENDENCIES
# -----------------------------------------------------------------------------
import os
import warnings
import hashlib
import random
import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import LabelEncoder

# PuLP — LP / Transport solver
try:
    import pulp
except ImportError:
    raise ImportError("Install PuLP: pip install pulp")

# joblib — parallel execution
try:
    from joblib import Parallel, delayed
except ImportError:
    raise ImportError("Install joblib: pip install joblib")

# AMI metric
try:
    from sklearn.metrics import adjusted_mutual_info_score
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 2. UTILITY & PRE-PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def remove_label_column(df: pd.DataFrame, label_col_name: str):
    """Separates the ground truth label column from the feature space."""
    if label_col_name not in df.columns:
        warnings.warn("Label column does not exist.")
        return df, None
    labels = df[label_col_name].values
    data_only = df.drop(columns=[label_col_name])
    return data_only, labels


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes ID/empty columns, imputes missing values using the mean,
    and encodes categorical variables into numeric format.
    """
    df = df.copy()

    id_cols = [c for c in df.columns if c.strip().lower() == "id"]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)

    df.dropna(axis=1, how="all", inplace=True)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    for col in df.columns:
        if df[col].isna().any():
            col_mean = df[col].mean()
            if np.isnan(col_mean):
                col_mean = 0.0
            df[col].fillna(col_mean, inplace=True)

    return df.reset_index(drop=True)


def compute_cardinality_ranges(target: np.ndarray, delta: float):
    """
    Computes the allowable lower and upper bounds [L, U] for each cluster's
    size based on the user-defined target cardinality vector and delta.
    """
    lower = np.floor(target * (1 - delta)).astype(int)
    upper = np.round(target * (1 + delta)).astype(int)
    return {"lower": lower, "upper": upper}


def count_combinations_exact(n: int, ranges: dict) -> int:
    """
    Calculates the exact theoretical number of valid cardinality partitions
    using dynamic programming.
    """
    lower = ranges["lower"]
    upper = ranges["upper"]
    k = len(lower)
    S = n - int(np.sum(lower))
    R = upper - lower

    if S < 0:
        return 0

    dp = [0] * (S + 1)
    dp[0] = 1

    for i in range(k):
        limit = int(R[i])
        new_dp = [0] * (S + 1)
        for s in range(S + 1):
            if dp[s] > 0:
                for v in range(limit + 1):
                    if s + v <= S:
                        new_dp[s + v] += dp[s]
        dp = new_dp

    return dp[S]


def generate_smart_random_pool(n: int, ranges: dict, n_samples: int, seed: int = 124) -> list:
    """
    Generates a valid random pool of candidate cardinality vectors that
    strictly satisfy total instance count and range constraints.
    """
    rng = random.Random(seed)
    lower = ranges["lower"]
    upper = ranges["upper"]
    k = len(lower)

    pool = []
    seen = set()
    attempts = 0
    max_attempts = n_samples * 20

    while len(pool) < n_samples and attempts < max_attempts:
        attempts += 1
        current_vec = [0] * k
        current_sum = 0
        valid_path = True

        for i in range(k - 1):
            min_needed_future = int(np.sum(lower[i + 1:]))
            max_possible_future = int(np.sum(upper[i + 1:]))

            safe_min = max(int(lower[i]), n - current_sum - max_possible_future)
            safe_max = min(int(upper[i]), n - current_sum - min_needed_future)

            if safe_min > safe_max:
                valid_path = False
                break

            val = rng.randint(safe_min, safe_max)
            current_vec[i] = val
            current_sum += val

        if valid_path:
            remainder = n - current_sum
            if int(lower[k - 1]) <= remainder <= int(upper[k - 1]):
                current_vec[k - 1] = remainder
                key = tuple(current_vec)
                if key not in seen:
                    seen.add(key)
                    pool.append(np.array(current_vec, dtype=int))

    return pool


# -----------------------------------------------------------------------------
# 3. CORE CLUSTERING LOGIC
# -----------------------------------------------------------------------------

def get_cost_matrix(X_norm: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Computes the cosine-distance cost matrix between data points and centroids."""
    centers_norms = np.sqrt(np.sum(centers ** 2, axis=1, keepdims=True)) + 1e-10
    centers_norm = centers / centers_norms
    sim = X_norm @ centers_norm.T        # (n, k)
    dists = 1.0 - sim
    dists = np.clip(dists, 0, None)
    return dists


def optimize_constrained_clustering(
    X: np.ndarray,
    X_norm: np.ndarray,
    card_constraints: np.ndarray,
    max_iter: int = 20,
    seed: int = None
) -> dict:
    """
    Solves the constrained clustering assignment problem using PuLP (transport
    formulation) to minimize global cosine-distance cost.

    Equivalent to R's lp.transport with row_signs='=' and col_signs='='.
    """
    rng = np.random.RandomState(seed)
    k = len(card_constraints)
    n = nrow = X.shape[0]

    idx = rng.choice(n, k, replace=False)
    centroids = X[idx].copy().astype(float)

    old_p = np.zeros(n, dtype=int)
    valid = True

    for iteration in range(max_iter):
        costs = get_cost_matrix(X_norm, centroids)   # (n, k)

        # --- Transport LP via PuLP ---
        prob = pulp.LpProblem("transport", pulp.LpMinimize)

        # Decision variables: x[i][j] ∈ {0, 1}  (binary for transport with supply=1)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous")
              for j in range(k)] for i in range(n)]

        # Objective: minimize total cost
        prob += pulp.lpSum(costs[i, j] * x[i][j] for i in range(n) for j in range(k))

        # Row constraints: each point assigned to exactly 1 cluster
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(k)) == 1

        # Column constraints: each cluster receives exactly card_constraints[j] points
        for j in range(k):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == int(card_constraints[j])

        solver = pulp.getSolver("PULP_CBC_CMD", msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            valid = False
            break

        # Extract assignment matrix
        sol = np.array([[pulp.value(x[i][j]) for j in range(k)] for i in range(n)])
        p = np.argmax(sol, axis=1)  # cluster index per point (0-based)

        if np.all(p == old_p):
            break
        old_p = p.copy()

        # Update centroids
        for j in range(k):
            mask = (p == j)
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
            else:
                centroids[j] = X[rng.randint(0, n)]

    return {"p": p, "centroids": centroids, "valid": valid}


def evaluate_solution(
    X: np.ndarray,
    X_norm: np.ndarray,
    target_card: np.ndarray,
    card: np.ndarray,
    true_labels=None,
    seed_val: int = None
):
    """
    Wrapper that executes the optimization for a specific cardinality candidate
    and calculates performance metrics.

    Returns: [silhouette, ILVC, CLVC, CSVI, AMI]  or None if infeasible.
    """
    res = optimize_constrained_clustering(X, X_norm, card, max_iter=15, seed=seed_val)
    if not res["valid"]:
        return None

    p = res["p"]
    centroids = res["centroids"]
    n = X.shape[0]
    k = len(card)

    dists = get_cost_matrix(X_norm, centroids)   # (n, k)
    idx_mat = (np.arange(n), p)
    a_i = dists[idx_mat].copy()

    dists_copy = dists.copy()
    dists_copy[idx_mat] = np.inf
    b_i = dists_copy.min(axis=1)

    denom = np.maximum(a_i, b_i)
    sil_vals = np.where(denom == 0, 0.0, (b_i - a_i) / denom)
    sil_mean = float(np.nanmean(sil_vals))

    counts = np.bincount(p, minlength=k)
    ilvc = float(np.sum(np.abs(np.sort(counts) - np.sort(target_card))))
    clvc = float(np.sum(np.sort(counts) != np.sort(target_card)))
    csvi = 0.5 * (ilvc / n) + 0.5 * (clvc / k)

    ami_val = np.nan
    if true_labels is not None:
        tl = np.array([str(x).strip() for x in true_labels])
        if len(np.unique(tl)) >= 2 and len(tl) == len(p):
            tl_int = LabelEncoder().fit_transform(tl)
            ami_val = float(adjusted_mutual_info_score(tl_int, p))

    return [sil_mean, ilvc, clvc, csvi, ami_val]


# -----------------------------------------------------------------------------
# 4. PARALLEL SEARCH
# -----------------------------------------------------------------------------

def _evaluate_single(i, card_candidate, X, X_norm, target_card, true_labels, master_seed):
    """Worker function executed in parallel for each cardinality candidate."""
    current_seed = master_seed + (i * 7)
    metrics = evaluate_solution(X, X_norm, target_card, card_candidate, true_labels, seed_val=current_seed)
    if metrics is None:
        return None
    return {
        "solution_id": i,
        "silhouette":  metrics[0],
        "ILVC":        metrics[1],
        "CLVC":        metrics[2],
        "CSVI":        metrics[3],
        "AMI":         metrics[4],
        "cardinality": "-".join(str(c) for c in card_candidate),
        "saved_seed":  current_seed,
    }


def run_parallel_search(
    dataset: pd.DataFrame,
    target_card: np.ndarray,
    delta: float,
    true_labels=None,
    n_BA: int = 100,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Distributes the evaluation of the cardinality pool across processor cores
    to efficiently explore the solution space in parallel.
    """
    X = dataset.values.astype(float)
    X_norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + 1e-10
    X_norm = X / X_norms
    n = X.shape[0]

    ranges = compute_cardinality_ranges(target_card, delta)
    max_teorico = count_combinations_exact(n, ranges)
    n_samples_final = min(n_BA, max_teorico)

    pool = generate_smart_random_pool(n, ranges, n_samples_final, seed=124)
    n_jobs_pool = len(pool)

    if n_jobs_pool == 0:
        warnings.warn("Empty cardinality pool — no feasible combinations found.")
        return pd.DataFrame()

    print(f" - Combinations to explore: {n_jobs_pool}\n")

    MASTER_SEED = 777

    raw_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_evaluate_single)(
            i, pool[i], X, X_norm, target_card, true_labels, MASTER_SEED
        )
        for i in range(n_jobs_pool)
    )

    rows = [r for r in raw_results if r is not None]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# 4b. PARETO FRONT & KNEE POINT ANALYSIS
# -----------------------------------------------------------------------------

def analyze_pareto_exact(df_input: pd.DataFrame):
    """
    Identifies non-dominated solutions (Pareto Front) and selects the optimal
    trade-off solution using the Knee Point method.

    Maximizes Silhouette, minimizes CSVI.
    """
    if df_input is None or df_input.empty:
        return None

    df = df_input.copy()

    col_map = {"silhouette": "s", "Silhouette": "s", "CSVI": "csvi"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if "s" not in df.columns or "csvi" not in df.columns:
        raise ValueError("DataFrame must contain 'silhouette'/'s' and 'CSVI'/'csvi' columns.")

    # 1. Identify Non-Dominated Solutions (Pareto: max s, min csvi)
    s_vals   = df["s"].values
    csvi_vals = df["csvi"].values
    n = len(df)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            dominates = (
                s_vals[j]    >= s_vals[i]    and
                csvi_vals[j] <= csvi_vals[i] and
                (s_vals[j] > s_vals[i] or csvi_vals[j] < csvi_vals[i])
            )
            if dominates:
                pareto_mask[i] = False
                break

    df["pareto"] = pareto_mask
    pf = df[df["pareto"]].copy().sort_values("s", ascending=False).reset_index(drop=True)

    # 2. Knee Point — maximum perpendicular distance to the line between extremes
    def knee_point(pf_df: pd.DataFrame) -> pd.DataFrame:
        if len(pf_df) < 3:
            return pf_df.iloc[[0]]

        x = pf_df["s"].values
        y = pf_df["csvi"].values
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]

        denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if denom == 0:
            return pf_df.iloc[[0]]

        dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        dist[0]  = -np.inf
        dist[-1] = -np.inf

        return pf_df.iloc[[np.argmax(dist)]]

    kp = knee_point(pf)

    return {"pareto_full": pf, "knee_full": kp}
