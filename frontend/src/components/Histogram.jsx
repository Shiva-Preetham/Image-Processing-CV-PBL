// Before/after brightness histograms (32 bins over 0-255) drawn as SVG bars.
function Bars({ counts, title, color }) {
  const max = Math.max(...counts, 1);
  const w = 300, h = 110, gap = 1.5, bar = w / counts.length;
  return (
    <figure className="hist">
      <svg viewBox={`0 0 ${w} ${h + 18}`} role="img" aria-label={`${title} brightness histogram`}>
        {counts.map((c, i) => {
          const bh = (c / max) * h;
          return <rect key={i} x={i * bar + gap / 2} y={h - bh} width={bar - gap} height={bh} rx="1.5" fill={color} />;
        })}
        <line x1="0" x2={w} y1={h} y2={h} className="axis" />
        <text x="0" y={h + 14} className="tick">0 (dark)</text>
        <text x={w} y={h + 14} className="tick" textAnchor="end">255 (bright)</text>
      </svg>
      <figcaption>{title}</figcaption>
    </figure>
  );
}

export default function Histogram({ before, after }) {
  return (
    <div className="hist-pair">
      <Bars counts={before} title="Before" color="var(--muted-strong)" />
      <Bars counts={after} title="After CLAHE" color="var(--primary)" />
    </div>
  );
}
