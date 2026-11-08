// Minimal SVG line chart for the training curves (one point per epoch).
export default function LineChart({ title, series, markerAfter, format = (v) => v.toFixed(2) }) {
  const w = 420, h = 240, pad = { l: 46, r: 14, t: 12, b: 42 };
  const all = series.flatMap((s) => s.values);
  let lo = Math.min(...all), hi = Math.max(...all);
  const margin = (hi - lo) * 0.12 || 0.05;
  lo -= margin;
  hi += margin;
  const n = series[0].values.length;
  const x = (i) => pad.l + (i / Math.max(n - 1, 1)) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const ticks = Array.from({ length: 4 }, (_, i) => lo + ((i + 0.5) / 4) * (hi - lo));

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="tick">{format(t)}</text>
          </g>
        ))}
        {series[0].values.map((_, i) => (
          <text key={i} x={x(i)} y={h - pad.b + 17} textAnchor="middle" className="tick">{i + 1}</text>
        ))}
        <text x={w - pad.r} y={h - 2} textAnchor="end" className="tick">epoch →</text>
        {markerAfter != null && (
          <g>
            <line x1={(x(markerAfter - 1) + x(markerAfter)) / 2} x2={(x(markerAfter - 1) + x(markerAfter)) / 2}
                  y1={pad.t} y2={h - pad.b} className="marker" />
            <text x={(x(markerAfter - 1) + x(markerAfter)) / 2 + 4} y={pad.t + 10} className="tick">fine-tuning →</text>
          </g>
        )}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth="2.5" strokeLinejoin="round"
                      points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
            {s.values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="3.5" fill={s.color} />)}
          </g>
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}
      </div>
    </figure>
  );
}
