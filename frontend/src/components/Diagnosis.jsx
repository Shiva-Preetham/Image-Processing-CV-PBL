import { useEffect, useRef, useState } from "react";
import { api, pct } from "../api";
import Histogram from "./Histogram";
import ModuleBadge from "./ModuleBadge";

const FILTERS = ["Median", "Gaussian", "Average"];

function Segmented({ label, hint, options, value, onChange }) {
  return (
    <div className="field">
      <div className="field-label">
        {label} {hint && <ModuleBadge module={hint} />}
      </div>
      <div className="segmented" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o} role="radio" aria-checked={value === o}
                  className={value === o ? "seg active" : "seg"} onClick={() => onChange(o)}>
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}

function SampleGallery({ samples, selected, onPick }) {
  return (
    <div className="gallery">
      {samples.map((s) => (
        <button key={s.name} title={`${s.label} #${s.number}`} onClick={() => onPick(s.name)}
                className={selected === s.name ? "thumb active" : "thumb"}>
          <img src={api.url(s.url)} alt={`${s.label} sample ${s.number}`} />
          <span className={`thumb-tag ${s.label === "Parasitized" ? "tag-p" : "tag-u"}`}>
            {s.label === "Parasitized" ? "P" : "U"}{s.number}
          </span>
        </button>
      ))}
    </div>
  );
}

function UploadZone({ onFile, active }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);
  const take = (files) => files?.[0] && onFile(files[0]);
  return (
    <div className={`drop ${dragging ? "dragging" : ""} ${active ? "active" : ""}`}
         onClick={() => input.current.click()}
         onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
         onDragLeave={() => setDragging(false)}
         onDrop={(e) => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files); }}>
      <input ref={input} type="file" accept="image/png,image/jpeg" hidden onChange={(e) => take(e.target.files)} />
      <span className="drop-icon" aria-hidden="true">⬆</span>
      <span><strong>Upload a cell image</strong><br />drag and drop, or click (PNG / JPG, max 5 MB)</span>
    </div>
  );
}

function Verdict({ result }) {
  const v = result.verdict;
  const infected = v.label === "Parasitized";
  const gt = result.ground_truth;
  return (
    <div className="verdict-col">
      <div className={`verdict ${infected ? "infected" : "healthy"}`}>
        <span className="kicker">CNN verdict · Stage 9</span>
        <h2>{infected ? "🦠" : "✅"} {v.label}</h2>
        <span>Confidence {pct(v.confidence)}</span>
      </div>
      <div className="prob">
        <div className="prob-head"><span>Probability of infection</span><b>{pct(v.p_parasitized)}</b></div>
        <div className="prob-track"><div className="prob-fill" style={{ width: pct(v.p_parasitized) }} /></div>
      </div>
      <div className="mini-stats">
        <div className="mini">
          <span>Stained spots found</span>
          <b>{result.segmentation.spots}</b>
          <small>Stages 6-8</small>
        </div>
        <div className="mini">
          <span>Dataset label</span>
          <b className={gt ? (gt === v.label ? "match" : "miss") : ""}>{gt || "Uploaded"}</b>
          <small>{gt ? (gt === v.label ? "CNN agrees" : "CNN disagrees") : "no label"}</small>
        </div>
      </div>
    </div>
  );
}

export default function Diagnosis() {
  const [samples, setSamples] = useState([]);
  const [source, setSource] = useState({ type: "sample", name: "parasitized_2" });
  const [settings, setSettings] = useState({ filter: "Median", ksize: 3, morph: 3 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const requestId = useRef(0);

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    const request = source.type === "sample"
      ? api.analyzeSample(source.name, settings)
      : api.analyzeUpload(source.file, settings);
    request
      .then((r) => id === requestId.current && setResult({ ...r, id }))
      .catch((e) => id === requestId.current && setError(e.message))
      .finally(() => id === requestId.current && setLoading(false));
  }, [source, settings]);

  const set = (key) => (value) => setSettings((s) => ({ ...s, [key]: value }));
  const steps = result?.steps ?? [];
  const seg = result?.segmentation;

  return (
    <div className="diagnosis">
      <aside className="sidebar">
        <div className="card">
          <h3>1 · Choose a cell</h3>
          <p className="muted">12 test images the model never saw in training.</p>
          <SampleGallery samples={samples} selected={source.type === "sample" ? source.name : null}
                         onPick={(name) => setSource({ type: "sample", name })} />
          <UploadZone active={source.type === "upload"}
                      onFile={(file) => setSource({ type: "upload", file, name: file.name })} />
          {source.type === "upload" && <p className="muted small">Analysing: {source.name}</p>}
        </div>
        <div className="card">
          <h3>2 · Tune the pipeline</h3>
          <Segmented label="Noise filter" hint="Module 3" options={FILTERS} value={settings.filter} onChange={set("filter")} />
          <Segmented label="Filter kernel size" options={[3, 5, 7]} value={settings.ksize} onChange={set("ksize")} />
          <Segmented label="Morphology kernel" hint="Module 2" options={[3, 5]} value={settings.morph} onChange={set("morph")} />
          <p className="muted small">The CNN was trained with Median 3×3. Change the settings to see how each stage reacts.</p>
        </div>
      </aside>

      <section className="results">
        {error && <div className="card error">⚠️ {error}</div>}
        {result && (
          <div className={`card hero ${loading ? "is-loading" : ""}`}>
            <figure className="hero-img">
              <img className="pixel" src={steps[1]?.image} alt="Input cell after crop and resize" />
              <figcaption>Input cell</figcaption>
            </figure>
            <figure className="hero-img">
              <img className="pixel" src={steps[9]?.image} alt="Cell outline and stained spots" />
              <figcaption><span className="key-green">■</span> cell outline · <span className="key-red">■</span> stained spots</figcaption>
            </figure>
            <Verdict result={result} />
            {loading && <div className="hero-spinner"><div className="spinner" /></div>}
          </div>
        )}
        {!result && loading && <div className="card wait"><div className="spinner" /><p>Analysing the cell…</p></div>}

        {result && (
          <>
            <h2 className="section-title">How the computer got there, step by step</h2>
            <div className="steps">
              {steps.map((s, i) => (
                <figure className="step" key={`${result.id}-${s.key}`} style={{ animationDelay: `${i * 55}ms` }}>
                  <img className="pixel" src={s.image} alt={s.title} />
                  <figcaption>
                    <div className="step-title"><span>{s.title}</span><ModuleBadge module={s.module} /></div>
                    <p>{s.note}</p>
                  </figcaption>
                </figure>
              ))}
            </div>

            <div className="details">
              <div className="card">
                <h3>Stage 4 · Brightness histogram <ModuleBadge module="Module 3" /></h3>
                <p className="muted">CLAHE spreads the brightness values out, so faint details inside the cell become visible.</p>
                <Histogram before={result.histogram.before} after={result.histogram.after} />
              </div>
              <div className="card">
                <h3>Stage 6 · Thresholds chosen by Otsu <ModuleBadge module="Module 4" /></h3>
                <p className="muted">Otsu picks each threshold automatically from the image's own histogram.</p>
                <dl className="facts">
                  <div><dt>Cell vs. black background (grayscale)</dt><dd>{seg.otsu_cell.toFixed(0)}</dd></div>
                  <div><dt>Stained vs. normal cell pixels (saturation)</dt><dd>{seg.otsu_stain.toFixed(0)}</dd></div>
                  <div>
                    <dt>Stain contrast, spot vs. cell body (needs ≥ {seg.min_stain_gap})</dt>
                    <dd className={seg.stain_gap >= seg.min_stain_gap ? "match" : ""}>{seg.stain_gap.toFixed(0)}</dd>
                  </div>
                  <div><dt>Total stained area after erosion + dilation</dt><dd>{seg.spot_area} px</dd></div>
                </dl>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
