import { useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const uploadVCF = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData
    });

    setResult(await res.json());
  };

  return (
    <div className="container">
      <h2>Pathogenic Variant Classifier</h2>

      <input type="file" accept=".vcf" onChange={e => setFile(e.target.files[0])} />
      <button onClick={uploadVCF}>Analyze</button>

      {result && (
        <div className="result">
          <h3>{result.prediction}</h3>
          <p>Confidence: {result.confidence}</p>

          <h4>Model Explanation</h4>
          <ul>
            {Object.entries(result.explanation).map(([k, v]) => (
              <li key={k}>{k}: {v.toFixed(3)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}