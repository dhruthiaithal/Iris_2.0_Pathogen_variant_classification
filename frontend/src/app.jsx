import React, { useState, useRef } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedVariant, setSelectedVariant] = useState(null);

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

  const toggleVariant = (index) => {
    setSelectedVariant(selectedVariant === index ? null : index);
  };

  // -------------------------
  // CSV DOWNLOAD FUNCTION
  // -------------------------

  const downloadCSV = () => {
    if (!result || !result.results) return;

    const headers = [
      "Chr",
      "Position",
      "Ref",
      "Alt",
      "Prediction",
      "Confidence",
      "Disease",
    ];

    const rows = result.results.map((variant) => [
      variant.Chr,
      variant.Start,
      variant.Ref,
      variant.Alt,
      variant.prediction,
      variant.confidence,
      variant.clinvar_disease || "",
    ]);

    const csvContent =
      [headers, ...rows].map((row) => row.join(",")).join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "variant_predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app">
      <div className="card">

        <h1 className="title">Variant Pathogenicity Classifier</h1>
        <p className="subtitle">
          Upload a VCF file to predict whether variants are pathogenic.
        </p>

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
            {loading ? "Analyzing Variants..." : "Analyze Variants"}
          </button>

        </div>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">

            <h2 className="results-title">
              Total Variants: {result.total_variants}
            </h2>

            {/* CSV Download Button */}

            <button
              className="download-btn"
              onClick={downloadCSV}
              style={{
                marginBottom: "15px",
                padding: "10px 18px",
                backgroundColor: "#2c7be5",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              Download Results (CSV)
            </button>

            <div className="table-container">

              <table className="variant-table">

                <thead>
                  <tr>
                    <th>Chr</th>
                    <th>Position</th>
                    <th>Ref</th>
                    <th>Alt</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Disease</th>
                  </tr>
                </thead>

                <tbody>
                  {result.results.map((variant, index) => (
                    <React.Fragment key={index}>
                      <tr
                        className="variant-row"
                        onClick={() => toggleVariant(index)}
                      >
                        <td>{variant.Chr}</td>
                        <td>{variant.Start}</td>
                        <td>{variant.Ref}</td>
                        <td>{variant.Alt}</td>

                        <td
                          className={
                            variant.prediction === "Pathogenic"
                              ? "pathogenic"
                              : "benign"
                          }
                        >
                          {variant.prediction}
                        </td>

                        <td>{variant.confidence}%</td>

                        <td>
                          {variant.clinvar_disease
                            ? variant.clinvar_disease
                            : "-"}
                        </td>
                      </tr>

                      {selectedVariant === index && (
                        <tr className="shap-row">
                          <td colSpan="7">

                            <div className="shap-box">

                              <h3>Feature Contribution (SHAP)</h3>

                              {Object.entries(variant.explanation).map(
                                ([key, value]) => (
                                  <div key={key} className="shap-item">
                                    <span>{key}</span>
                                    <span>{Number(value).toFixed(3)}</span>
                                  </div>
                                )
                              )}

                              {variant.clinvar_disease && (
                                <div className="clinvar-disease">
                                  <strong>Associated Disease(s):</strong>{" "}
                                  {variant.clinvar_disease}
                                </div>
                              )}

                            </div>

                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>

              </table>

            </div>

          </div>
        )}

      </div>
    </div>
  );
}