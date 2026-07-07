import React, { useState } from 'react';
import { predictTrafficAudio } from '../api';
import MicRecorder from './MicRecorder';

const PredictionCard = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const data = await predictTrafficAudio(file);
    if (data.success) setResult(data);
    else setError(data.error || "Failed to analyze audio");
    
    setLoading(false);
  };

  return (
    <div className="card">
      <h2>AI Traffic Analyzer</h2>
      <p>Upload a 4-second .wav or .raw file, or record live audio to classify the vehicle.</p>
      
      <div className="file-upload-wrapper">
        <input 
          type="file" 
          accept=".wav,.raw" 
          id="audio-file"
          onChange={(e) => {
            setFile(e.target.files[0]);
            setResult(null);
            setError(null);
          }} 
        />
        <label htmlFor="audio-file" className="file-label">
          {file ? "Change File" : "Choose File"}
        </label>
        {file && (
          <p className="file-name">{file.name}</p>
        )}
      </div>

      <div className="mode-buttons">
        <button onClick={handlePredict} disabled={!file || loading}>
          {loading ? "Analyzing..." : "Analyze File"}
        </button>
        <MicRecorder 
          setResult={setResult} 
          setError={setError} 
          setLoading={setLoading}
        />
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-box">
          <h1 className="prediction">{result.prediction.toUpperCase()}</h1>
          <p className="confidence">Confidence: {result.confidence.toFixed(2)}%</p>
          <div className="top-3">
            <h4>Top 3 Matches:</h4>
            {result.top_3.map((item, i) => (
              <div key={i} className="top-3-item">
                <span className="label">{item.label}</span>
                <span className="confidence">{item.confidence.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionCard;