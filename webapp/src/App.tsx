import { Activity, Download, Moon, ScanLine, Sun, UploadCloud } from "lucide-react";
import { ChangeEvent, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [dark, setDark] = useState(true);
  const [status, setStatus] = useState("Ready");
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const fileLabel = useMemo(() => file?.name ?? "Select MRI volume", [file]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setResultUrl(null);
    setStatus(selected ? "Volume staged" : "Ready");
  }

  async function submit() {
    if (!file) return;
    setStatus("Uploading and running inference");
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${apiUrl}/predict`, { method: "POST", body: form });
    if (!response.ok) {
      const text = await response.text();
      setStatus(`Inference failed: ${text}`);
      return;
    }
    const blob = await response.blob();
    setResultUrl(URL.createObjectURL(blob));
    setStatus("Prediction ready");
  }

  return (
    <main className={dark ? "app dark" : "app"}>
      <nav className="topbar">
        <div className="brand">
          <ScanLine size={24} />
          <span>Brain Tumor Segmentation</span>
        </div>
        <button className="iconButton" onClick={() => setDark(!dark)} title="Toggle theme">
          {dark ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </nav>

      <section className="workspace">
        <div className="panel uploadPanel">
          <div className="panelHeader">
            <UploadCloud size={22} />
            <h1>MRI Inference</h1>
          </div>
          <label className="dropZone">
            <input
              type="file"
              accept=".npy,.npz,.nii,.gz"
              onChange={onFileChange}
              aria-label="Upload MRI volume"
            />
            <UploadCloud size={38} />
            <strong>{fileLabel}</strong>
            <span>Supports NPY, NPZ, NIfTI, and NIfTI.GZ volumes</span>
          </label>
          <button className="primaryButton" disabled={!file} onClick={submit}>
            <Activity size={18} />
            Run Segmentation
          </button>
        </div>

        <div className="panel viewerPanel">
          <div className="panelHeader">
            <Activity size={22} />
            <h2>Result</h2>
          </div>
          <div className="viewer">
            <div className="scanGrid" aria-hidden="true">
              {Array.from({ length: 64 }, (_, index) => (
                <span key={index} className={index % 7 === 0 ? "tumorVoxel" : ""} />
              ))}
            </div>
            <div className="status">{status}</div>
          </div>
          {resultUrl && (
            <a className="downloadButton" href={resultUrl} download="prediction.npz">
              <Download size={18} />
              Download NPZ Output
            </a>
          )}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);

