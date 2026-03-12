import React, { useState, useRef } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    setFile(e.target.files[0]);
  };

  const openFileDialog = () => {
    fileInputRef.current.click();
  };

  const uploadVCF = async () => {
    if (!file) {
      setError("Please upload a VCF file.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Server error. Check backend logs.");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <div className="card">

        <h1 className="title">Variant Pathogenicity Classifier</h1>
        <p className="subtitle">
          Upload a VCF file to predict whether the variant is pathogenic.
        </p>

        {/* Hidden file input */}
        <input
          type="file"
          accept=".vcf"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />

        <div className="upload-section">

          <button className="upload-btn" onClick={openFileDialog}>
            Upload VCF File
          </button>

          {file && (
            <p className="filename">
              Selected file: <strong>{file.name}</strong>
            </p>
          )}

          <button
            className="analyze-btn"
            onClick={uploadVCF}
            disabled={!file || loading}
          >
            {loading ? "Analyzing Variant..." : "Analyze Variant"}
          </button>

        </div>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">

            <h2
              className={
                result.prediction === "Pathogenic"
                  ? "prediction pathogenic"
                  : "prediction benign"
              }
            >
              {result.prediction}
            </h2>

            <p className="confidence">
              Confidence Score: <strong>{result.confidence}</strong>
            </p>

            {result.model_breakdown && (
              <div className="model-box">
                <h3>Model Breakdown</h3>
                <p>XGBoost: {result.model_breakdown.xgboost}</p>
              </div>
            )}

            {result.explanation && (
              <div className="shap-box">
                <h3>Feature Contribution (SHAP)</h3>

                {Object.entries(result.explanation).map(([key, value]) => (
                  <div key={key} className="shap-item">
                    <span>{key}</span>
                    <span>{Number(value).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}

          </div>
        )}

      </div>
    </div>
  );
}