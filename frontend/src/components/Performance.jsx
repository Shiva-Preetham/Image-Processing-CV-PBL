import { useEffect, useState } from "react";
import { api, pct } from "../api";
import LineChart from "./LineChart";

// First training run, kept for comparison (see README "What changed since the first run").
const BASELINE = { accuracy: 0.9267, precision: 0.955, recall: 0.8956, f1: 0.9243, FN: 47 };

const METRICS = [
  { key: "accuracy", name: "Accuracy", explain: "Correct answers out of all test cells." },
  { key: "precision", name: "Precision", explain: "When it says “infected”, how often it is right." },
  { key: "recall", name: "Recall", explain: "Out of all truly infected cells, how many it caught." },
  { key: "f1", name: "F1-score", explain: "One number balancing precision and recall." },
];

function ConfusionMatrix({ cm }) {
  const cells = [
    { v: cm.TN, kind: "good", text: "Healthy, called healthy" },
    { v: cm.FP, kind: "bad", text: "False alarm" },
    { v: cm.FN, kind: "worst", text: "Missed infection" },
    { v: cm.TP, kind: "good", text: "Infected, caught" },
  ];
  const total = cm.TN + cm.FP + cm.FN + cm.TP;
  return (
    <div className="cm">
      <div className="cm-grid">
        <span />
        <span className="cm-head">Predicted<br />Uninfected</span>
        <span className="cm-head">Predicted<br />Parasitized</span>
        {[0, 1].map((row) => (
          <div className="cm-row" key={row}>
            <span className="cm-head cm-side">Actually<br />{row ? "Parasitized" : "Uninfected"}</span>
            {cells.slice(row * 2, row * 2 + 2).map((c) => (
              <div key={c.text} className={`cm-cell ${c.kind}`} style={{ "--a": 0.15 + 0.85 * (c.v / (total / 2)) }}>
                <b>{c.v}</b>
                <span>{c.text}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Performance() {
  const [m, setM] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.metrics().then(setM).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="card error">⚠️ {error}</div>;
  if (!m) return <div className="card wait"><div className="spinner" /></div>;

  const cm = m.confusion_matrix;
  const h = m.history;
  const headEpochs = h.filter((e) => e.phase.startsWith("Phase A")).length;

  return (
    <div className="performance">
      <p className="lead">
        Tested on <strong>{m.dataset.test.toLocaleString()} cell images the model never saw</strong> during training
        (trained on {m.dataset.train.toLocaleString()}, validated on {m.dataset.val.toLocaleString()}; {m.train_minutes} minutes on a laptop CPU).
      </p>

      <div className="metric-grid">
        {METRICS.map(({ key, name, explain }) => {
          const delta = m[key] - BASELINE[key];
          return (
            <div className="card metric" key={key}>
              <span className="metric-name">{name}</span>
              <span className="metric-value">{pct(m[key])}</span>
              <span className={delta >= 0 ? "delta" : "delta down"}>
                {delta >= 0 ? "▲" : "▼"} {Math.abs(delta * 100).toFixed(1)} pts vs. first run
              </span>
              <p className="muted small">{explain}</p>
            </div>
          );
        })}
      </div>

      <div className="perf-row">
        <div className="card">
          <h3>Confusion matrix</h3>
          <p className="muted">
            <strong>{cm.FN} missed infections</strong> out of {cm.FN + cm.TP} infected cells, and {cm.FP} false alarms.
            In diagnosis a missed infection is the costlier mistake, so we watch recall, not just accuracy.
          </p>
          <ConfusionMatrix cm={cm} />
        </div>
        <div className="card">
          <h3>Training curves</h3>
          <p className="muted">
            Phase A trains only the new output layer; Phase B fine-tunes the last blocks of MobileNetV2.
            Training and validation stay close, so the model is not just memorising.
          </p>
          <div className="charts">
            <LineChart title="Accuracy (higher is better)" markerAfter={headEpochs} format={(v) => pct(v, 0)}
                       series={[{ name: "Train", color: "var(--c-train)", values: h.map((e) => e.train_acc) },
                                { name: "Validation", color: "var(--c-val)", values: h.map((e) => e.val_acc) }]} />
            <LineChart title="Loss (lower is better)" markerAfter={headEpochs}
                       series={[{ name: "Train", color: "var(--c-train)", values: h.map((e) => e.train_loss) },
                                { name: "Validation", color: "var(--c-val)", values: h.map((e) => e.val_loss) }]} />
          </div>
        </div>
      </div>

      <div className="card">
        <h3>What changed since the first run</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Version</th><th>Stage 4 enhancement</th><th>Fine-tuning</th><th>Accuracy</th><th>Recall</th><th>Missed infections</th></tr>
            </thead>
            <tbody>
              <tr><td>Baseline</td><td>Plain histogram equalisation</td><td>3 epochs, last 5 blocks</td>
                <td>{pct(BASELINE.accuracy)}</td><td>{pct(BASELINE.recall)}</td><td>{BASELINE.FN}</td></tr>
              <tr className="highlight"><td>v0.1</td><td>CLAHE</td><td>5 epochs, last 8 blocks</td>
                <td>{pct(m.accuracy)}</td><td>{pct(m.recall)}</td><td>{cm.FN}</td></tr>
            </tbody>
          </table>
        </div>
        <p className="muted small">
          Plain equalisation stretched each cell's darkest pixels to almost black, which hurt both Otsu and the CNN.
          CLAHE (still Module 3) limits the contrast boost.
        </p>
      </div>

      <div className="card">
        <h3>Sample test predictions</h3>
        <img className="wide" src={api.url("/static/outputs/sample_predictions.png")}
             alt="Grid of test cells with predicted and true labels" />
      </div>
    </div>
  );
}
