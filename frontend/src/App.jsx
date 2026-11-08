import { useEffect, useState } from "react";
import { api } from "./api";
import Diagnosis from "./components/Diagnosis";
import Performance from "./components/Performance";
import SyllabusMap from "./components/SyllabusMap";

const TABS = [
  { id: "diagnosis", label: "Live diagnosis", icon: "🔍" },
  { id: "performance", label: "Model performance", icon: "📊" },
  { id: "syllabus", label: "Syllabus map", icon: "🗺️" },
];

// The free Render server sleeps when idle, so the first request can take
// up to a minute. Keep pinging /api/health and tell the user what is going on.
function useServerStatus() {
  const [status, setStatus] = useState("connecting");
  useEffect(() => {
    let cancelled = false;
    let tries = 0;
    const slow = setTimeout(() => setStatus((s) => (s === "connecting" ? "waking" : s)), 2500);
    async function ping() {
      try {
        await api.health();
        if (!cancelled) setStatus("online");
      } catch {
        tries += 1;
        if (cancelled) return;
        if (tries >= 20) setStatus("offline");
        else {
          setStatus("waking");
          setTimeout(ping, 5000);
        }
      }
    }
    ping();
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);
  return status;
}

function StatusPill({ status }) {
  const text = { connecting: "Connecting", waking: "Waking server", online: "Model online", offline: "Server offline" }[status];
  return (
    <span className={`status status-${status}`}>
      <span className="status-dot" /> {text}
    </span>
  );
}

function WaitScreen({ status }) {
  if (status === "offline") {
    return (
      <div className="card wait">
        <h2>Can't reach the analysis server</h2>
        <p>The Python server that runs the image processing and the CNN is not responding.</p>
        <button className="btn" onClick={() => window.location.reload()}>Try again</button>
      </div>
    );
  }
  return (
    <div className="card wait">
      <div className="spinner" />
      <h2>{status === "waking" ? "Waking up the analysis server…" : "Connecting…"}</h2>
      <p>
        The image processing and the CNN run on a free cloud server that sleeps when nobody uses it.
        The first visit can take up to a minute; after that every analysis takes about a second.
      </p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("diagnosis");
  const status = useServerStatus();

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width="34" height="34" />
          <div>
            <h1>MalariaVision</h1>
            <p>Malaria parasite detection from blood-smear cell images</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Sections">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
              <span aria-hidden="true">{t.icon}</span> {t.label}
            </button>
          ))}
        </nav>
        <StatusPill status={status} />
      </header>

      <main>
        <section className="intro">
          <p>
            <strong>Is this red blood cell infected with the malaria parasite?</strong> Classical image processing
            cleans the cell and outlines any dark-stained spot, then a pre-trained CNN gives the final answer.
            Every step uses a technique from our Image Processing and Computer Vision lab syllabus.
          </p>
        </section>
        {/* Tabs stay mounted (just hidden) so the chosen cell and settings survive tab switches. */}
        <div hidden={tab !== "syllabus"}><SyllabusMap /></div>
        {tab !== "syllabus" && status !== "online" && <WaitScreen status={status} />}
        {status === "online" && (
          <>
            <div hidden={tab !== "diagnosis"}><Diagnosis /></div>
            <div hidden={tab !== "performance"}><Performance /></div>
          </>
        )}
      </main>

      <footer className="footer">
        <div>
          <strong>MalariaVision</strong> · Project-based learning, Fundamental of Image Processing and Computer Vision Lab
          (26B16CS316), 2026-27
        </div>
        <div>
          Lakshita Gautam · Ranjeet Kaur · Navya Banga · Shiva Preetham Chelimila · Faculty: Ms. Neha, Dr. Pawan Kumar
          Upadhyay
        </div>
        <div>
          <a href="https://github.com/Shiva-Preetham/Image-Processing-CV-PBL" target="_blank" rel="noreferrer">
            Source code on GitHub
          </a>{" "}
          · Dataset: NIH Malaria Cell Images · For education, not for medical diagnosis
        </div>
      </footer>
    </div>
  );
}
