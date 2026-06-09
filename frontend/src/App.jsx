import { useState, useEffect, useRef, useCallback } from "react";
import "./styles/capflex.css";

const CLUSTER_COLORS = [
  "#00FFB2", "#FF4D6D", "#4D9FFF", "#FFB800", "#BF5FFF",
  "#FF7A3D", "#00E0FF", "#FF3DE8", "#7FFF00", "#FF9AAA",
];

const BG     = "#F4F7FB";
const BORDER = "#D6E0EE";
const ACCENT = "#0066FF";
const MUTED  = "#8FA3BF";

const CLUSTERING_API = "/api/clustering";
const EMBEDDING_API  = "/api/embedding";

async function apiPost(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function pollStatus(statusUrl, onStatus, intervalMs = 1500) {
  return new Promise((resolve, reject) => {
    const iv = setInterval(async () => {
      try {
        const data = await apiGet(statusUrl);
        onStatus(data.status);
        if (data.status === "done")  { clearInterval(iv); resolve(data); }
        if (data.status === "error") { clearInterval(iv); reject(new Error(data.error || "Job failed")); }
      } catch (e) { clearInterval(iv); reject(e); }
    }, intervalMs);
  });
}

async function fetchCSV(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Could not download results CSV");
  const text = await res.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const obj  = {};
    headers.forEach((h, i) => {
      const v = vals[i]?.trim();
      obj[h] = isNaN(v) || v === "" ? v : +v;
    });
    return obj;
  });
}

function rowsToPoints(rows, embPrefix = "emb") {
  if (!rows.length) return [];
  const allKeys     = Object.keys(rows[0]);
  const featureKeys = allKeys.filter(
    (k) => k !== "cluster" && k !== "true_label" && k !== "id" && !k.startsWith(`${embPrefix}_`)
  );
  let xKey = featureKeys.find((k) => typeof rows[0][k] === "number");
  let yKey = featureKeys.filter((k) => typeof rows[0][k] === "number")[1];
  if (!xKey) xKey = `${embPrefix}_0`;
  if (!yKey) yKey = `${embPrefix}_1`;
  return rows.map((row, i) => {
    const features = {};
    featureKeys.slice(0, 6).forEach((k) => {
      if (typeof row[k] === "number") features[k] = row[k];
    });
    return {
      id: i,
      filename: row.id ?? null,
      x: row[xKey] ?? 0, y: row[yKey] ?? 0,
      cluster: row.cluster ?? null,
      trueLabel: row.true_label ?? null,
      features
    };
  });
}

const INPUT_TYPES = [
  { value: "tabular",    label: "Tabular",   icon: "⊞", desc: "Numeric CSV"       },
  { value: "images",     label: "Images",    icon: "🖼", desc: "Generate CLIP emb" },
  { value: "text",       label: "Text",      icon: "T",  desc: "Generate text emb" },
  { value: "embeddings", label: "Embedding", icon: "⬡", desc: "Load emb CSV"      },
];

const EMB_GEN_TYPES = ["images", "text"];

export default function CapFlexUI() {
  const [activeTab, setActiveTab]     = useState("pca");
  const [inputType, setInputType]     = useState("tabular");
  const skipInputTypeReset = useRef(false);

  const [file, setFile]               = useState(null);
  const [fileName, setFileName]       = useState(null);
  const [labelCol, setLabelCol]       = useState("");
  const [csvColumns, setCsvColumns]   = useState([]);
  const [excludedCols, setExcludedCols] = useState([]);
  const [targetCard, setTargetCard]   = useState("50,50,50");
  const [delta, setDelta]             = useState(0.1);
  const [maxIter, setMaxIter]         = useState("");
  const [embPrefix, setEmbPrefix]     = useState("emb");
  const [embJobId, setEmbJobId]       = useState(null);
  const [status, setStatus]           = useState("idle");
  const [statusMsg, setStatusMsg]     = useState("No data loaded");
  const [progress, setProgress]       = useState(0);
  const [points, setPoints]           = useState([]);
  const [clusterJobId, setClusterJobId] = useState(null);
  const [clustered, setClustered]     = useState(false);
  const [pareto, setPareto]           = useState([]);
  const [kneeMetrics, setKneeMetrics] = useState(null);
  const [clusterFilter, setClusterFilter] = useState(null);
  const [sortCol, setSortCol]         = useState(null);
  const [sortDir, setSortDir]         = useState("asc");
  const [tooltip, setTooltip]         = useState({ visible: false, x: 0, y: 0, data: null });

  const [mediaFiles, setMediaFiles]         = useState([]);   // images or text files
  const [textCsvColumns, setTextCsvColumns] = useState([]);   // columns from text CSV
  const [textColumn, setTextColumn]         = useState("");   // column to embed
  const [textIdColumn, setTextIdColumn]     = useState("");   // optional id column
  const [embGenStatus, setEmbGenStatus]     = useState("idle");
  const [embGenMsg, setEmbGenMsg]           = useState("No files selected");
  const [embGenProgress, setEmbGenProgress] = useState(0);
  const [embResultJobId, setEmbResultJobId] = useState(null);

  const canvasRef = useRef(null);

  useEffect(() => {
    if (skipInputTypeReset.current) {
      skipInputTypeReset.current = false;
      return;
    }
    setMediaFiles([]);
    setTextCsvColumns([]); setTextColumn(""); setTextIdColumn("");
    setEmbGenStatus("idle");
    setEmbGenMsg("No files selected");
    setEmbGenProgress(0);
    setEmbResultJobId(null);
    setPoints([]); setClustered(false); setPareto([]); setKneeMetrics(null);
    setClusterFilter(null); setStatus("idle"); setStatusMsg("No data loaded");
    setProgress(0); setFile(null); setFileName(null); setEmbJobId(null);
    setCsvColumns([]); setExcludedCols([]); setLabelCol(""); setClusterJobId(null);
    setActiveTab("pca");
  }, [inputType]);

  const handleFileChange = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setFileName(f.name); setEmbJobId(null);
    setStatusMsg(`File selected: ${f.name} — configure parameters and click Run`);
    setStatus("idle"); setClustered(false); setPareto([]); setKneeMetrics(null); setClusterFilter(null);

    try {
      const text    = await f.text();
      const lines   = text.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim());
      const rows    = lines.slice(1).map((line) => {
        const vals = line.split(",");
        const obj  = {};
        headers.forEach((h, i) => { const v = vals[i]?.trim(); obj[h] = isNaN(v) || v === "" ? v : +v; });
        return obj;
      });
      setCsvColumns(headers);
      setExcludedCols([]);
      setLabelCol("");
      const numericCols = headers.filter((h) => typeof rows[0][h] === "number");
      const xKey = numericCols[0];
      const yKey = numericCols[1] || numericCols[0];
      const preview = rows.map((row, i) => ({
        id: i, x: row[xKey] ?? 0, y: row[yKey] ?? 0,
        cluster: null,
        features: Object.fromEntries(numericCols.slice(0, 6).map((k) => [k, row[k]])),
      }));
      setPoints(preview);
      suggestCardinality(preview.length);
    } catch (_) {
      setCsvColumns([]); setPoints([]);
    }
  };

  const handleMediaChange = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setMediaFiles(files);
    setEmbGenStatus("idle");
    setEmbResultJobId(null);
    setEmbGenMsg(`${files.length} file${files.length > 1 ? "s" : ""} selected — click Generate`);
    setEmbGenProgress(0);
    suggestCardinality(files.length);
    if (inputType === "text" && files[0]) {
      try {
        const txt = await files[0].text();
        const headers = txt.trim().split("\n")[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
        setTextCsvColumns(headers);
        setTextColumn(headers[0] ?? "");
        setTextIdColumn("");
      } catch (_) { setTextCsvColumns([]); }
    }
  };

  const handleGenerateEmbeddings = async () => {
    if (!mediaFiles.length) return;
    try {
      setEmbGenStatus("loading"); setEmbResultJobId(null);
      setEmbGenMsg("Uploading files…"); setEmbGenProgress(15);

      const form = new FormData();

      let endpoint;
      if (inputType === "images") {
        mediaFiles.forEach((f) => form.append("files", f));
        endpoint = `${EMBEDDING_API}/embeddings/images`;
      } else {
        form.append("file", mediaFiles[0]);
        form.append("text_column", textColumn);
        if (textIdColumn.trim()) form.append("id_column", textIdColumn.trim());
        endpoint = `${EMBEDDING_API}/embeddings/texts`;
      }

      const submitted = await apiPost(endpoint, form);
      const jobId = submitted.job_id;

      const encMsg = inputType === "images"
        ? "Encoding images with CLIP…"
        : "Encoding text with embedding model…";

      setEmbGenMsg(inputType === "images" ? "Generating CLIP embeddings…" : "Generating text embeddings…");
      setEmbGenProgress(40);

      await pollStatus(`${EMBEDDING_API}/embeddings/status/${jobId}`, (s) => {
        if (s === "running") { setEmbGenMsg(encMsg); setEmbGenProgress(70); }
      });

      setEmbResultJobId(jobId);
      setEmbGenStatus("done");
      setEmbGenProgress(100);
      setEmbGenMsg(`Done — ${mediaFiles.length} embeddings generated`);
    } catch (err) {
      setEmbGenStatus("error");
      setEmbGenMsg(`Error: ${err.message}`);
      setEmbGenProgress(0);
    }
  };

  const handleDownloadEmbeddings = () => {
    if (!embResultJobId) return;
    window.open(`${EMBEDDING_API}/embeddings/download/${embResultJobId}`, "_blank");
  };

  const handleUseInClustering = async () => {
    if (!embResultJobId) return;
    const jobId = embResultJobId;
    setEmbJobId(jobId);
    setFile(null); setFileName(null);
    skipInputTypeReset.current = true;
    setInputType("embeddings");  
    setStatusMsg(`Using embedding job ${jobId.slice(0, 8)}… — configure cardinality and click Run`);
    setStatus("idle"); setClustered(false); setPoints([]); setPareto([]); setKneeMetrics(null);

    try {
      const rows = await fetchCSV(`${EMBEDDING_API}/embeddings/download/${jobId}`);
      if (rows.length) {
        const embCols = Object.keys(rows[0]).filter(k => k.startsWith("emb_") && typeof rows[0][k] === "number");
        const xKey = embCols[0] ?? "emb_0";
        const yKey = embCols[1] ?? "emb_1";
        const preview = rows.map((row, i) => ({
          id: i,
          filename: row.id ?? null,
          x: row[xKey] ?? 0,
          y: row[yKey] ?? 0,
          cluster: null,
          features: {},
        }));
        setPoints(preview);
        suggestCardinality(preview.length);
      }
    } catch (_) {}
  };

  const handleRun = async () => {
    if (!file && !embJobId) return;
    try {
      setStatus("loading"); setClustered(false); setPareto([]); setKneeMetrics(null); setClusterFilter(null);
      setStatusMsg("Submitting clustering job…"); setProgress(10);

      const form = new FormData();

      if (embJobId) {
        form.append("embedding_job_id", embJobId);
      } else {
        let fileToSend = file;
        if (excludedCols.length > 0) {
          const text    = await file.text();
          const lines   = text.trim().split("\n");
          const headers = lines[0].split(",").map((h) => h.trim());
          const keepCols = headers.filter((h) => !excludedCols.includes(h));
          const keepIdx  = keepCols.map((h) => headers.indexOf(h));
          const filtered = [
            keepCols.join(","),
            ...lines.slice(1).map((line) => {
              const vals = line.split(",");
              return keepIdx.map((i) => vals[i]).join(",");
            }),
          ].join("\n");
          fileToSend = new File([filtered], file.name, { type: "text/csv" });
        }
        form.append("file", fileToSend);
        const backendInputType = inputType === "embeddings" ? "embeddings" : "tabular";
        form.append("input_type", backendInputType);
        form.append("embedding_prefix", embPrefix);
        if (labelCol.trim()) form.append("label_column", labelCol.trim());
      }

      form.append("target_cardinality", targetCard);
      form.append("delta", delta);
      if (maxIter.trim()) form.append("max_iter", maxIter.trim());

      const submitted = await apiPost(`${CLUSTERING_API}/clustering/run`, form);
      const jobId = submitted.job_id;
      setClusterJobId(jobId);
      setStatusMsg("Exploring cardinality pool…"); setProgress(30);
      await pollStatus(`${CLUSTERING_API}/clustering/status/${jobId}`, (s) => {
        if (s === "running") { setStatusMsg("Running parallel LP optimization…"); setProgress(65); }
      });
      setStatusMsg("Fetching results…"); setProgress(85);
      const results = await apiGet(`${CLUSTERING_API}/clustering/results/${jobId}`);
      setStatusMsg("Downloading assignment data…"); setProgress(95);
      const rows = await fetchCSV(`${CLUSTERING_API}/clustering/download/${jobId}`);
      const newPoints = rowsToPoints(rows, embPrefix);
      const paretoUI = results.pareto_front.map((sol, idx) => ({
        ...sol,
        isKnee: sol.cardinality === results.knee_point.cardinality &&
                idx === results.pareto_front.findIndex((s) => s.cardinality === results.knee_point.cardinality),
      }));
      if (!paretoUI.some((s) => s.isKnee) && paretoUI.length)
        paretoUI[Math.floor(paretoUI.length / 2)].isKnee = true;
      setPoints(newPoints); setClustered(true); setPareto(paretoUI); setKneeMetrics(results.knee_point);
      setProgress(100); setStatus("done");
      setStatusMsg(`Optimal solution found — Silhouette: ${results.knee_point.silhouette} · CSVI: ${results.knee_point.CSVI} · Cardinality: ${results.knee_point.cardinality}`);
    } catch (err) { setStatus("error"); setStatusMsg(`Error: ${err.message}`); setProgress(0); }
  };

  const handleDownloadClustering = () => {
    if (!clusterJobId) return;
    window.open(`${CLUSTERING_API}/clustering/download/${clusterJobId}`, "_blank");
  };

  const suggestCardinality = (n) => {
    if (!n || n < 2) return;
    const half = Math.floor(n / 2);
    const other = n - half;
    setTargetCard(half === other ? `${half},${half}` : `${half},${other}`);
  };

  const handleReset = () => {
    setPoints([]); setClustered(false); setPareto([]); setKneeMetrics(null);
    setClusterFilter(null); setStatus("idle"); setStatusMsg("No data loaded");
    setProgress(0); setFile(null); setFileName(null); setEmbJobId(null);
    setCsvColumns([]); setExcludedCols([]); setLabelCol("");
    setMediaFiles([]); setTextCsvColumns([]); setTextColumn(""); setTextIdColumn("");
    setEmbGenStatus("idle"); setEmbGenMsg("No files selected");
    setEmbGenProgress(0); setEmbResultJobId(null); setActiveTab("pca");
    setInputType("tabular");
  };

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, PAD = 48;
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    if (!points.length) return;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xRange = xmax - xmin || 1, yRange = ymax - ymin || 1;
    const toCanvas = (px, py) => ({
      cx: PAD + ((px - xmin) / xRange) * (W - PAD * 2),
      cy: H - PAD - ((py - ymin) / yRange) * (H - PAD * 2),
    });
    ctx.strokeStyle = BORDER + "88"; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = PAD + (i / 5) * (W - PAD * 2), y = PAD + (i / 5) * (H - PAD * 2);
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }
    ctx.strokeStyle = BORDER; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, H - PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD); ctx.stroke();
    points.forEach((pt) => {
      const { cx, cy } = toCanvas(pt.x, pt.y);
      const isFiltered = clusterFilter !== null && pt.cluster !== clusterFilter;
      const color = (clustered && pt.cluster !== null) ? CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] : "#CBD5E1";
      ctx.globalAlpha = isFiltered ? 0.12 : 0.9;
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      if (!isFiltered) { ctx.strokeStyle = clustered ? color + "44" : BORDER; ctx.lineWidth = 1; ctx.stroke(); }
    });
    ctx.globalAlpha = 1;
  }, [points, clustered, clusterFilter]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; drawCanvas(); });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [drawCanvas]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  useEffect(() => {
    if (activeTab === "pca") {
      const t = setTimeout(() => {
        const canvas = canvasRef.current;
        if (canvas) { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; drawCanvas(); }
      }, 50);
      return () => clearTimeout(t);
    }
  }, [activeTab, drawCanvas]);

  const handleCanvasMouseMove = (e) => {
    if (!points.length || !clustered) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const W = canvas.width, H = canvas.height, PAD = 48;
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    let nearest = null, minDist = 14;
    points.forEach((pt) => {
      const cx = PAD + ((pt.x - xmin) / (xmax - xmin || 1)) * (W - PAD * 2);
      const cy = H - PAD - ((pt.y - ymin) / (ymax - ymin || 1)) * (H - PAD * 2);
      const d = Math.hypot(mx - cx, my - cy);
      if (d < minDist) { minDist = d; nearest = pt; }
    });
    if (nearest) setTooltip({ visible: true, x: e.clientX + 12, y: e.clientY - 10, data: nearest });
    else setTooltip((t) => ({ ...t, visible: false }));
  };

  const tableData = clustered
    ? points.filter((p) => clusterFilter === null || p.cluster === clusterFilter)
        .sort((a, b) => {
          if (!sortCol) return 0;
          const va = sortCol === "cluster" ? a.cluster : a.features?.[sortCol] ?? 0;
          const vb = sortCol === "cluster" ? b.cluster : b.features?.[sortCol] ?? 0;
          return sortDir === "asc" ? va - vb : vb - va;
        })
    : [];

  const handleSort = (col) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("asc"); }
  };

  const clusters = clustered ? [...new Set(points.map((p) => p.cluster))].sort((a, b) => a - b) : [];
  const knee = pareto.find((p) => p.isKnee);

  const isEmbGenMode   = EMB_GEN_TYPES.includes(inputType);   // Images or Text
  const isLoadEmbMode  = inputType === "embeddings";            // load CSV
  const isTabularMode  = inputType === "tabular";

  const mediaAccept = inputType === "images" ? "image/*" : ".txt,.csv,.tsv,text/plain";

  const canRunClustering = isTabularMode ? !!file : isLoadEmbMode ? (!!file || !!embJobId) : !!embJobId;

  return (
    <>
      <div className="app">
        <header className="header">
          <div className="logo" onClick={handleReset} style={{ cursor: "pointer" }}>Cap<span>Flex</span></div>
          <div className="badge">2026</div>
          <div style={{ fontSize: 13, color: MUTED, fontFamily: "'Space Mono', monospace" }}>
            Semi-supervised Flexible Cardinality Clustering
          </div>
          <nav className="tabs">
            <button className={`tab ${activeTab === "pca" ? "active" : ""}`} onClick={() => setActiveTab("pca")}>PCA Explorer</button>
            <button className={`tab ${activeTab === "table" ? "active" : ""}`} onClick={() => setActiveTab("table")} disabled={!clustered} style={{ opacity: clustered ? 1 : 0.3 }}>Cluster Results</button>
          </nav>
        </header>

        <div className="main">
          <aside className="sidebar">

            {/* ── DATA SOURCE TYPE SELECTOR ── */}
            <div className="section">
              <div className="section-title">Data Source</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {INPUT_TYPES.map((t) => (
                  <button
                    key={t.value}
                    onClick={() => setInputType(t.value)}
                    style={{
                      display: "flex", flexDirection: "column", alignItems: "center",
                      justifyContent: "center", gap: 4, padding: "12px 8px",
                      borderRadius: 8, cursor: "pointer",
                      border: inputType === t.value ? `2px solid ${ACCENT}` : "1.5px solid var(--border)",
                      background: inputType === t.value ? "#E8F0FF" : "var(--surface2)",
                      color: inputType === t.value ? ACCENT : MUTED,
                      fontFamily: "var(--mono)", fontSize: 11,
                      fontWeight: inputType === t.value ? 700 : 400,
                      transition: "all 0.15s ease",
                    }}
                  >
                    <span style={{ fontSize: 20 }}>{t.icon}</span>
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
            {isEmbGenMode && (
              <>
                <div className="section">
                  <div className="section-title">
                    {inputType === "images" ? "Image Files" : "Text Files"}
                  </div>
                  <div className={`upload-zone ${mediaFiles.length ? "has-file" : ""}`}>
                    <input type="file" accept={mediaAccept} multiple onChange={handleMediaChange} />
                    <div className="upload-icon">{mediaFiles.length ? (inputType === "images" ? "🖼" : "📄") : "⬆"}</div>
                    <div className="upload-text">
                      {mediaFiles.length
                        ? `${mediaFiles.length} file${mediaFiles.length > 1 ? "s" : ""} selected`
                        : `Select ${inputType === "images" ? "images" : "text files"}`}
                    </div>
                    {mediaFiles.length > 0 && (
                      <div className="upload-filename" style={{ maxHeight: 64, overflowY: "auto" }}>
                        {mediaFiles.slice(0, 5).map((f) => f.name).join(", ")}
                        {mediaFiles.length > 5 && ` +${mediaFiles.length - 5} more`}
                      </div>
                    )}
                  </div>
                </div>

                {inputType === "text" && textCsvColumns.length > 0 && (
                  <div className="section">
                    <div className="section-title">Text Column</div>
                    <select
                      value={textColumn}
                      onChange={e => setTextColumn(e.target.value)}
                      style={{ width: "100%", padding: "6px 8px", borderRadius: 6, border: "1.5px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontFamily: "var(--mono)", fontSize: 11, marginBottom: 10 }}
                    >
                      {textCsvColumns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                    <div className="section-title">ID Column <span style={{ color: MUTED, fontWeight: 400 }}>(optional)</span></div>
                    <select
                      value={textIdColumn}
                      onChange={e => setTextIdColumn(e.target.value)}
                      style={{ width: "100%", padding: "6px 8px", borderRadius: 6, border: "1.5px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontFamily: "var(--mono)", fontSize: 11 }}
                    >
                      <option value="">— none (use row index) —</option>
                      {textCsvColumns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                )}

                <div className="section">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: "'Space Mono',monospace", fontSize: 10, color: MUTED, marginBottom: 6 }}>
                    <div className={`status-dot ${embGenStatus === "loading" ? "active" : embGenStatus === "done" ? "done" : embGenStatus === "error" ? "error" : ""}`} />
                    <span style={{ color: embGenStatus === "error" ? "#E83A5A" : MUTED }}>{embGenMsg}</span>
                  </div>
                  {embGenStatus === "loading" && (
                    <div className="progress-bar" style={{ marginBottom: 8 }}>
                      <div className="progress-fill" style={{ width: `${embGenProgress}%` }} />
                    </div>
                  )}
                </div>

                <div className="section">
                  <button
                    className="run-btn"
                    onClick={handleGenerateEmbeddings}
                    disabled={!mediaFiles.length || embGenStatus === "loading"}
                  >
                    {embGenStatus === "loading"
                      ? <><span className="shimmer" />Generating…</>
                      : `⬡ Generate ${inputType === "images" ? "CLIP" : "Text"} Embeddings`}
                  </button>
                </div>

                {embGenStatus === "done" && embResultJobId && (
                  <div className="section">
                    <div className="section-title">Actions</div>
                    <div style={{ background: "var(--surface2)", border: `1px solid var(--border)`, borderRadius: 6, padding: "8px 10px", marginBottom: 10 }}>
                      <div style={{ fontSize: 9, color: MUTED, fontFamily: "'Space Mono',monospace", letterSpacing: 1, marginBottom: 3 }}>JOB ID</div>
                      <div style={{ fontSize: 9, color: ACCENT, fontFamily: "'Space Mono',monospace", wordBreak: "break-all" }}>{embResultJobId}</div>
                    </div>
                    <button
                      className="run-btn"
                      onClick={handleDownloadEmbeddings}
                      style={{ marginBottom: 8, background: "var(--surface2)", border: `1.5px solid ${ACCENT}`, color: ACCENT }}
                    >
                      ⬇ Download CSV
                    </button>
                    <button
                      className="run-btn"
                      onClick={handleUseInClustering}
                      style={{ background: ACCENT, color: "#fff" }}
                    >
                      ▶ Use in Clustering
                    </button>
                  </div>
                )}

                {embJobId && (
                  <>

                    <div className="section">
                      <div className="section-title">Clustering Parameters</div>
                      <div className="form-group">
                        <div className="form-label">TARGET CARDINALITY <span className="hint">comma-separated</span></div>
                        <input type="text" value={targetCard} onChange={(e) => setTargetCard(e.target.value)} placeholder="50,50,50" />
                      </div>
                      <div className="form-group">
                        <div className="form-label">DELTA (δ) <span className="hint">tolerance</span></div>
                        <div className="range-row">
                          <input type="range" min="0" max="0.5" step="0.01" value={delta} onChange={(e) => setDelta(+e.target.value)} />
                          <span className="range-val">{delta.toFixed(2)}</span>
                        </div>
                      </div>
                      <div className="form-group">
                        <div className="form-label">MAX ITERATIONS <span className="hint">auto if empty</span></div>
                        <input type="number" value={maxIter} onChange={(e) => setMaxIter(e.target.value)} placeholder="auto" />
                      </div>
                    </div>
                    <div className="section">
                      <button className="run-btn" onClick={handleRun} disabled={status === "loading"}>
                        {status === "loading" ? <><span className="shimmer" />Running…</> : "▶ Run CapFlex"}
                      </button>
                    {clustered && clusterJobId && (
                      <button
                        className="run-btn"
                        onClick={handleDownloadClustering}
                        style={{ marginTop: 8, background: "var(--surface2)", border: `1.5px solid ${ACCENT}`, color: ACCENT }}
                      >
                        Download Results
                      </button>
                    )}
                    </div>

                  </>
                )}
              </>
            )}
            {(isTabularMode || isLoadEmbMode) && (
              <>
                {embJobId ? (
                  <div className="section">
                    <div style={{ background: "#E8F0FF", border: `1px solid ${ACCENT}40`, borderRadius: 8, padding: "12px 14px" }}>
                      <div style={{ fontSize: 9, color: ACCENT, fontFamily: "'Space Mono',monospace", letterSpacing: 1, marginBottom: 4 }}>EMBEDDING JOB LOADED</div>
                      <div style={{ fontSize: 10, color: MUTED, fontFamily: "'Space Mono',monospace", wordBreak: "break-all", marginBottom: 8 }}>{embJobId.slice(0, 18)}…</div>
                      <button
                        onClick={() => { setEmbJobId(null); setPoints([]); setClustered(false); setStatusMsg("Embedding job cleared"); }}
                        style={{ fontSize: 9, color: MUTED, background: "none", border: "none", cursor: "pointer", fontFamily: "'Space Mono',monospace", textDecoration: "underline", padding: 0 }}
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                ) : (
                <div className="section">
                  <div className="section-title">Upload CSV</div>
                  <div className={`upload-zone ${fileName ? "has-file" : ""}`}>
                    <input type="file" accept=".csv" onChange={handleFileChange} />
                    <div className="upload-icon">{fileName ? "📄" : "⬆"}</div>
                    <div className="upload-text">{fileName ? "File loaded" : "Drop CSV here or click to browse"}</div>
                    {fileName && <div className="upload-filename">{fileName}</div>}
                  </div>
                </div>
                )}

                <div className="section">
                  <div className="section-title">Parameters</div>

                  {csvColumns.length > 0 && isTabularMode && (
                    <>
                      <div className="form-group">
                        <div className="form-label">LABEL COLUMN</div>
                        <select
                          value={labelCol}
                          onChange={(e) => setLabelCol(e.target.value)}
                          style={{
                            width: "100%", background: "var(--surface2)", border: "1.5px solid var(--border)",
                            borderRadius: 6, padding: "9px 12px", color: "var(--text)",
                            fontFamily: "var(--mono)", fontSize: 12, outline: "none", cursor: "pointer",
                          }}
                        >
                          <option value="">— None —</option>
                          {csvColumns.map((col) => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                        </select>
                      </div>

                      <div className="form-group">
                        <div className="form-label">FEATURE COLUMNS</div>
                        <div style={{
                          maxHeight: 160, overflowY: "auto",
                          background: "var(--surface2)", border: "1.5px solid var(--border)",
                          borderRadius: 6, padding: "8px 10px",
                        }}>
                          {csvColumns.filter((col) => col !== labelCol).map((col) => (
                            <label key={col} style={{
                              display: "flex", alignItems: "center", gap: 8,
                              fontFamily: "var(--mono)", fontSize: 11, padding: "4px 0", cursor: "pointer",
                              color: excludedCols.includes(col) ? MUTED : "var(--text)",
                            }}>
                              <input
                                type="checkbox"
                                checked={!excludedCols.includes(col)}
                                onChange={(e) => setExcludedCols((prev) =>
                                  e.target.checked ? prev.filter((c) => c !== col) : [...prev, col]
                                )}
                                style={{ accentColor: ACCENT, width: 14, height: 14, cursor: "pointer" }}
                              />
                              {col}
                            </label>
                          ))}
                        </div>
                      </div>
                    </>
                  )}

                  {!csvColumns.length && isTabularMode && (
                    <div className="form-group">
                      <div className="form-label">LABEL COLUMN <span className="hint">optional</span></div>
                      <input type="text" value={labelCol} onChange={(e) => setLabelCol(e.target.value)} placeholder="e.g. Species" />
                    </div>
                  )}

                  {isLoadEmbMode && !embJobId && (
                    <div className="form-group">
                      <div className="form-label">EMBEDDING PREFIX</div>
                      <input type="text" value={embPrefix} onChange={(e) => setEmbPrefix(e.target.value)} placeholder="emb" />
                    </div>
                  )}

                  <div className="form-group">
                    <div className="form-label">TARGET CARDINALITY <span className="hint">comma-separated</span></div>
                    <input type="text" value={targetCard} onChange={(e) => setTargetCard(e.target.value)} placeholder="50,50,50" />
                  </div>

                  <div className="form-group">
                    <div className="form-label">DELTA (δ) <span className="hint">tolerance</span></div>
                    <div className="range-row">
                      <input type="range" min="0" max="0.5" step="0.01" value={delta} onChange={(e) => setDelta(+e.target.value)} />
                      <span className="range-val">{delta.toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="form-group">
                    <div className="form-label">MAX ITERATIONS <span className="hint">auto if empty</span></div>
                    <input type="number" value={maxIter} onChange={(e) => setMaxIter(e.target.value)} placeholder="auto" />
                  </div>
                </div>

                <div className="section">
                  <button className="run-btn" onClick={handleRun} disabled={!canRunClustering || status === "loading"}>
                    {status === "loading" ? <><span className="shimmer" />Running…</> : "▶ Run CapFlex"}
                  </button>
                    {clustered && clusterJobId && (
                      <button
                        className="run-btn"
                        onClick={handleDownloadClustering}
                        style={{ marginTop: 8, background: "var(--surface2)", border: `1.5px solid ${ACCENT}`, color: ACCENT }}
                      >
                        Download Results
                      </button>
                    )}
                </div>


              </>
            )}
          </aside>

          <div className="content">

            {isEmbGenMode && !embJobId ? (
              <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
                <div className="status-bar">
                  <div className={`status-dot ${embGenStatus === "loading" ? "active" : embGenStatus === "done" ? "done" : embGenStatus === "error" ? "error" : ""}`} />
                  <span style={{ color: embGenStatus === "error" ? "#E83A5A" : "var(--text-2)" }}>{embGenMsg}</span>
                  {embGenStatus === "loading" && (
                    <span style={{ marginLeft: "auto", color: ACCENT, fontFamily: "'Space Mono',monospace", fontSize: 11 }}>{Math.round(embGenProgress)}%</span>
                  )}
                </div>
                <div className="progress-bar">
                  {embGenStatus === "loading"
                    ? embGenProgress < 20
                      ? <div className="progress-fill indeterminate" />
                      : <div className="progress-fill" style={{ width: `${embGenProgress}%` }} />
                    : <div className="progress-fill" style={{ width: embGenStatus === "done" ? "100%" : "0%", opacity: 0.3 }} />}
                </div>

                {mediaFiles.length === 0 ? (
                  <div className="pca-overlay" style={{ position: "relative", flex: 1 }}>
                    <div className="pca-placeholder">
                      <div className="big">{inputType === "images" ? "🖼" : "T"}</div>
                      <p>Select {inputType === "images" ? "images" : "text files"} in the sidebar to preview them here</p>
                    </div>
                  </div>
                ) : inputType === "images" ? (
                  <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-2)" }}>
                        {mediaFiles.length} image{mediaFiles.length > 1 ? "s" : ""} selected
                      </div>
                      {embGenStatus === "done" && (
                        <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#00C48C", background: "#E6FAF4", border: "1px solid #00C48C", borderRadius: 4, padding: "3px 10px" }}>
                          ✓ {mediaFiles.length} embeddings generated
                        </div>
                      )}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 12 }}>
                      {mediaFiles.map((f, i) => {
                        const url = URL.createObjectURL(f);
                        return (
                          <div key={i} style={{ position: "relative", borderRadius: 8, overflow: "hidden", border: "1.5px solid var(--border)", background: "var(--surface)", aspectRatio: "1" }}>
                            <img src={url} alt={f.name} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} onLoad={() => URL.revokeObjectURL(url)} />
                            {embGenStatus === "loading" && (
                              <div style={{ position: "absolute", inset: 0, background: "rgba(255,255,255,0.6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                <div style={{ width: 24, height: 24, border: "3px solid var(--border)", borderTopColor: ACCENT, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                              </div>
                            )}
                            {embGenStatus === "done" && (
                              <div style={{ position: "absolute", inset: 0, background: "rgba(0,196,140,0.10)", display: "flex", alignItems: "flex-start", justifyContent: "flex-end", padding: 6, pointerEvents: "none" }}>
                                <div style={{ background: "#00C48C", color: "#fff", borderRadius: "50%", width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--mono)", fontSize: 12, fontWeight: 700 }}>✓</div>
                              </div>
                            )}
                            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "rgba(13,27,46,0.7)", color: "#fff", fontFamily: "var(--mono)", fontSize: 9, padding: "3px 6px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                              {f.name}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-2)" }}>
                        {mediaFiles.length} file{mediaFiles.length > 1 ? "s" : ""} selected
                      </div>
                      {embGenStatus === "done" && (
                        <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#00C48C", background: "#E6FAF4", border: "1px solid #00C48C", borderRadius: 4, padding: "3px 10px" }}>
                          ✓ {mediaFiles.length} embeddings generated
                        </div>
                      )}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {mediaFiles.map((f, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, background: "var(--surface2)", border: "1.5px solid var(--border)", borderRadius: 8, padding: "10px 14px" }}>
                          <span style={{ fontSize: 20 }}>📄</span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{f.name}</div>
                            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: MUTED, marginTop: 2 }}>{(f.size / 1024).toFixed(1)} KB</div>
                          </div>
                          {embGenStatus === "done" && <div style={{ color: "#00C48C", fontFamily: "var(--mono)", fontSize: 12 }}>✓</div>}
                          {embGenStatus === "loading" && <div style={{ width: 16, height: 16, border: "2px solid var(--border)", borderTopColor: ACCENT, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <div className="status-bar">
                  <div className={`status-dot ${status === "loading" ? "active" : status === "done" ? "done" : status === "error" ? "error" : ""}`} />
                  <span>{statusMsg}</span>
                  {status === "loading" && (
                    <span style={{ marginLeft: "auto", color: ACCENT, fontFamily: "'Space Mono',monospace", fontSize: 11 }}>{Math.round(progress)}%</span>
                  )}
                </div>
                <div className="progress-bar">
                  {status === "loading"
                    ? progress < 20 ? <div className="progress-fill indeterminate" /> : <div className="progress-fill" style={{ width: `${progress}%` }} />
                    : <div className="progress-fill" style={{ width: status === "done" ? "100%" : "0%", opacity: 0.3 }} />}
                </div>

                {activeTab === "pca" && (
                  <>
                    <div className="pca-area">
                      <canvas ref={canvasRef} className="pca-canvas" onMouseMove={handleCanvasMouseMove} onMouseLeave={() => setTooltip((t) => ({ ...t, visible: false }))} />
                      {points.length > 0 && (
                        <>
                          <div className="axis-label" style={{ bottom: 8, left: "50%", transform: "translateX(-50%)" }}>Feature axis 1</div>
                          <div className="axis-label" style={{ left: 4, top: "50%", transform: "rotate(-90deg) translateX(-50%)", transformOrigin: "left center" }}>Feature axis 2</div>
                        </>
                      )}
                      {!points.length && (
                        <div className="pca-overlay">
                          <div className="pca-placeholder">
                            <div className="big">◎</div>
                            <p>Load a CSV or generate embeddings, then run clustering</p>
                          </div>
                        </div>
                      )}
                      {clustered && clusters.length > 0 && (
                        <div className="legend">
                          <div className="legend-title">Clusters</div>
                          {clusters.map((c) => (
                            <div key={c} className="legend-item" style={{ cursor: "pointer", opacity: clusterFilter === null || clusterFilter === c ? 1 : 0.35 }}
                              onClick={() => setClusterFilter(clusterFilter === c ? null : c)}>
                              <div className="legend-dot" style={{ background: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                              Cluster {c + 1}
                              <span style={{ color: MUTED, fontSize: 10, marginLeft: 4 }}>({points.filter((p) => p.cluster === c).length})</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {points.length > 0 && !clustered && (
                        <div className="legend">
                          <div className="legend-title">State</div>
                          <div className="legend-item">
                            <div className="legend-dot" style={{ background: "#CBD5E1" }} />
                            Unclustered ({points.length})
                          </div>
                        </div>
                      )}
                    </div>

                    {pareto.length > 0 && (
                      <div className="pareto-panel">
                        <div className="pareto-header">
                          <div className="pareto-title">Pareto Front — Non-Dominated Solutions</div>
                          {knee && <div className="pareto-knee">Knee: Sol. {knee.solution_id} · Sil {(+knee.silhouette).toFixed(3)} · CSVI {(+knee.CSVI).toFixed(3)}</div>}
                        </div>
                        <div className="pareto-body">
                          {pareto.map((sol) => (
                            <div key={sol.solution_id} className={`pareto-card ${sol.isKnee ? "knee" : ""}`}>
                              <div className="pareto-card-id">{sol.isKnee ? "★ KNEE" : `SOL. ${sol.solution_id}`}</div>
                              <div className="pareto-metric"><span className="k">Sil</span><span className="v good">{(+sol.silhouette).toFixed(3)}</span></div>
                              <div className="pareto-metric"><span className="k">CSVI</span><span className="v">{(+sol.CSVI).toFixed(3)}</span></div>
                              <div className="pareto-metric"><span className="k">AMI</span><span className="v">{sol.AMI != null ? (+sol.AMI).toFixed(3) : "N/A"}</span></div>
                              <div className="pareto-metric"><span className="k">ILVC / CLVC</span><span className="v">{sol.ILVC} / {sol.CLVC}</span></div>
                              <div className="cardinality">Card: {sol.cardinality}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {activeTab === "table" && (
                  <>
                    {clustered && kneeMetrics && (
                      <div className="metrics-row">
                        {[
                          ["Silhouette", typeof kneeMetrics.silhouette === "number" ? kneeMetrics.silhouette.toFixed(4) : "—", true],
                          ["CSVI",       typeof kneeMetrics.CSVI       === "number" ? kneeMetrics.CSVI.toFixed(4)       : "—", false],
                          ["ILVC",       kneeMetrics.ILVC ?? "—", false],
                          ["CLVC",       kneeMetrics.CLVC ?? "—", false],
                          ["AMI",        kneeMetrics.AMI != null ? (+kneeMetrics.AMI).toFixed(4) : "N/A", true],
                        ].map(([k, v, good]) => (
                          <div key={k} className="metric-cell">
                            <span className="mk">{k}</span>
                            <span className={`mv ${good ? "" : "neutral"}`}>{v}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {clustered ? (
                      <>
                        <div className="filter-bar">
                          <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: MUTED, letterSpacing: 1, flexShrink: 0 }}>FILTER:</span>
                          <div className={`filter-chip ${clusterFilter === null ? "active" : ""}`}
                            style={clusterFilter === null ? { background: "#3A4A60", borderColor: "#3A4A60", color: "#fff" } : {}}
                            onClick={() => setClusterFilter(null)}>
                            All ({points.length})
                          </div>
                          {clusters.map((c) => (
                            <div key={c} className={`filter-chip ${clusterFilter === c ? "active" : ""}`}
                              style={clusterFilter === c ? { background: CLUSTER_COLORS[c % CLUSTER_COLORS.length], borderColor: CLUSTER_COLORS[c % CLUSTER_COLORS.length] } : {}}
                              onClick={() => setClusterFilter(clusterFilter === c ? null : c)}>
                              <div className="dot" style={{ background: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                              Cluster {c + 1} ({points.filter((p) => p.cluster === c).length})
                            </div>
                          ))}
                        </div>
                        <div className="table-area">
                          <table>
                            <thead>
                              <tr>
                                <th onClick={() => handleSort("id")} className={sortCol === "id" ? "sorted" : ""}>ID <span className="sort-icon">{sortCol === "id" ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span></th>
                                {Object.keys(points[0]?.features || {}).map((f) => (
                                  <th key={f} onClick={() => handleSort(f)} className={sortCol === f ? "sorted" : ""}>{f.toUpperCase()} <span className="sort-icon">{sortCol === f ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span></th>
                                ))}
                                <th onClick={() => handleSort("cluster")} className={sortCol === "cluster" ? "sorted" : ""}>CLUSTER <span className="sort-icon">{sortCol === "cluster" ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span></th>
                              </tr>
                            </thead>
                            <tbody>
                              {tableData.slice(0, 300).map((pt) => (
                                <tr key={pt.id}>
                                  <td>
                                    {pt.filename && mediaFiles.length > 0 ? (() => {
                                      const imgFile = mediaFiles.find(f => f.name === pt.filename);
                                      return imgFile ? (
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                          <img src={URL.createObjectURL(imgFile)} alt={pt.filename} style={{ width: 36, height: 36, objectFit: "cover", borderRadius: 4, border: "1px solid var(--border)", flexShrink: 0 }} />
                                          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-2)", wordBreak: "break-all" }}>{pt.filename}</span>
                                        </div>
                                      ) : <span>{pt.filename}</span>;
                                    })() : pt.id}
                                  </td>
                                  {Object.values(pt.features).map((v, i) => <td key={i}>{v}</td>)}
                                  <td>
                                    <span className="cluster-badge" style={{ background: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] + "20", color: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] }}>
                                      <span className="dot" style={{ background: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] }} />{pt.cluster + 1}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {tableData.length > 300 && (
                            <div style={{ padding: "12px 0", textAlign: "center", fontFamily: "'Space Mono',monospace", fontSize: 11, color: MUTED }}>
                              Showing 300 of {tableData.length} rows
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="empty-state"><div className="big">⬡</div><div>Run clustering first to see assignments</div></div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      <div className={`tooltip ${tooltip.visible ? "visible" : ""}`} style={{ left: tooltip.x, top: tooltip.y }}>
        {tooltip.data && (
          <>
            <div className="tcluster">Cluster {(tooltip.data.cluster ?? "?") + 1}</div>
            <div>ID: {tooltip.data.id}</div>
            {Object.entries(tooltip.data.features || {}).map(([k, v]) => <div key={k}>{k}: {v}</div>)}
          </>
        )}
      </div>
    </>
  );
}