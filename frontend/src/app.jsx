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

      const res = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) {
        throw new Error(
          "Server error. Check backend logs."
        );
      }

      const data = await res.json();

      setResult(data);

    } catch (err) {

      setError(err.message);

    }

    setLoading(false);
  };

  const toggleVariant = (index) => {

    setSelectedVariant(
      selectedVariant === index ? null : index
    );
  };

  const downloadCSV = async () => {

    try {

      const response = await fetch(
        result.important_variants_csv
      );

      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");

      a.href = url;

      a.download = "important_variants.csv";

      document.body.appendChild(a);

      a.click();

      a.remove();

    } catch (err) {

      console.error(err);

    }
  };

  return (

    <div className="app">

      <div className="card">

        <h1 className="title">
          Variant Pathogenicity Classifier
        </h1>

        <p className="subtitle">
          Upload a VCF file to predict whether
          variants are pathogenic.
        </p>

        <input
          type="file"
          accept=".vcf"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />

        <div className="upload-section">

          <button
            className="upload-btn"
            onClick={openFileDialog}
          >
            Upload VCF File
          </button>

          {file && (
            <p className="filename">
              Selected file:
              <strong> {file.name}</strong>
            </p>
          )}

          <button
            className="analyze-btn"
            onClick={uploadVCF}
            disabled={!file || loading}
          >
            {loading
              ? "Analyzing Variants..."
              : "Analyze Variants"}
          </button>

        </div>

        {error && (
          <p className="error">{error}</p>
        )}

        {result && (

          <div className="result">

            <h2 className="results-title">
              Total Variants:
              {" "}
              {result.total_variants}
            </h2>

            {/* GRS Section */}

            <div className="grs-box">

              <h2>
                Breast Cancer Genetic Risk Score
              </h2>

              <div className="grs-score">
                {result.breast_cancer_genetic_risk.grs}
              </div>

              <div className="risk-tier">

                Risk Tier:

                <span
                  className={
                    result.breast_cancer_genetic_risk
                      .risk_tier === "Very High Risk"
                      ? "very-high"
                      : result.breast_cancer_genetic_risk
                          .risk_tier === "High Risk"
                      ? "high"
                      : result.breast_cancer_genetic_risk
                          .risk_tier === "Moderate Risk"
                      ? "moderate"
                      : "low"
                  }
                >
                  {" "}
                  {
                    result
                      .breast_cancer_genetic_risk
                      .risk_tier
                  }
                </span>

              </div>

              <p className="grs-interpretation">

                {
                  result
                    .breast_cancer_genetic_risk
                    .interpretation
                }

              </p>

            </div>

            <button
              className="download-btn"
              onClick={downloadCSV}
            >
              Download Important Variants CSV
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
                    <th>GRS Contribution</th>
                  </tr>

                </thead>

                <tbody>

                  {result.results.map(
                    (variant, index) => (

                    <React.Fragment key={index}>

                      <tr
                        className="variant-row"
                        onClick={() =>
                          toggleVariant(index)
                        }
                      >

                        <td>{variant.Chr}</td>
                        <td>{variant.Start}</td>
                        <td>{variant.Ref}</td>
                        <td>{variant.Alt}</td>

                        <td
                          className={
                            variant.prediction ===
                            "Pathogenic"
                              ? "pathogenic"
                              : variant.prediction ===
                                "Benign"
                              ? "benign"
                              : "uncertain"
                          }
                        >
                          {variant.prediction}
                        </td>

                        <td>
                          {variant.confidence}%
                        </td>

                        <td>
                          {
                            variant.clinvar_disease
                            ? variant.clinvar_disease
                            : "-"
                          }
                        </td>

                        <td>
                          {
                            variant.grs_contribution
                          }
                        </td>

                      </tr>

                      {selectedVariant === index && (

                        <tr className="shap-row">

                          <td colSpan="8">

                            <div className="shap-box">

                              <div className="variant-meta">

                                <p>
                                  <strong>Gene:</strong>
                                  {" "}
                                  {variant.gene}
                                </p>

                                <p>
                                  <strong>
                                    Consequence:
                                  </strong>
                                  {" "}
                                  {
                                    variant.consequence
                                  }
                                </p>

                                <p>
                                  <strong>
                                    Pathogenicity Probability:
                                  </strong>
                                  {" "}
                                  {
                                    variant.pathogenicity_probability
                                  }
                                </p>

                              </div>

                              <h3>
                                Feature Contribution
                                (SHAP)
                              </h3>

                              {Object.entries(
                                variant.explanation
                              ).map(
                                ([key, value]) => (

                                <div
                                  key={key}
                                  className="shap-item"
                                >

                                  <span>{key}</span>

                                  <span>
                                    {
                                      Number(value)
                                      .toFixed(3)
                                    }
                                  </span>

                                </div>
                              ))}

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