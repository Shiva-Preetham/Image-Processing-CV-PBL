import ModuleBadge from "./ModuleBadge";

const STAGES = [
  ["Load, display, save", "cv2.imread, Matplotlib", "Module 1", "CO1"],
  ["Crop + resize", "Bounding-box crop, cv2.resize to 128 × 128", "Module 2", "CO1"],
  ["Colour spaces", "RGB → Grayscale, RGB → HSV", "Module 2", "CO1"],
  ["Enhancement", "CLAHE histogram equalisation (V channel)", "Module 3", "CO2"],
  ["Noise removal", "Median / Gaussian / Average filter", "Module 3", "CO2"],
  ["Segmentation", "Otsu thresholding (cell, then stain)", "Module 4", "CO2"],
  ["Mask cleanup", "Erosion + dilation", "Module 2", "CO2"],
  ["Region detection", "Contour detection, convex hull", "Module 5", "CO3"],
  ["Classification", "MobileNetV2 transfer learning, flip/rotate augmentation", "Module 5", "CO3"],
  ["Evaluation", "Accuracy, precision, recall, F1, confusion matrix", "Project", "CO4"],
];

const OUTCOMES = [
  ["CO1", "Image acquisition and basic manipulation", "Loading, display, resizing, cropping, colour spaces"],
  ["CO2", "Enhancement, filtering, segmentation", "CLAHE, median filter, Otsu, erosion and dilation"],
  ["CO3", "Detection and recognition with ML/DL", "Contours and CNN classification with transfer learning"],
  ["CO4", "Real-world solution, evaluated", "A healthcare application measured on unseen test images"],
];

export default function SyllabusMap() {
  return (
    <div className="syllabus">
      <div className="card">
        <h3>The pipeline at a glance</h3>
        <p className="muted">Every stage uses a technique from the lab syllabus, Modules 1 to 5. Nothing outside it.</p>
        <ol className="flow">
          {STAGES.map(([stage, , module], i) => (
            <li key={stage}>
              <span className="flow-num">{i + 1}</span>
              <span className="flow-name">{stage}</span>
              <ModuleBadge module={module} />
            </li>
          ))}
        </ol>
      </div>

      <div className="card">
        <h3>Stage by stage</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Stage</th><th>Technique</th><th>Module</th><th>CO</th></tr></thead>
            <tbody>
              {STAGES.map(([stage, technique, module, co], i) => (
                <tr key={stage}>
                  <td>{i + 1}</td><td>{stage}</td><td>{technique}</td>
                  <td><ModuleBadge module={module} /></td><td>{co}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="co-grid">
        {OUTCOMES.map(([co, title, how]) => (
          <div className="card co" key={co}>
            <span className="co-id">{co}</span>
            <h4>{title}</h4>
            <p className="muted small">{how}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
