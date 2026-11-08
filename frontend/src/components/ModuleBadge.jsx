// Small coloured tag showing which syllabus module a step comes from.
export default function ModuleBadge({ module }) {
  const n = String(module).match(/\d/)?.[0];
  const label = n ? `M${n}` : module;
  return <span className={`badge badge-m${n || "x"}`}>{label}</span>;
}
