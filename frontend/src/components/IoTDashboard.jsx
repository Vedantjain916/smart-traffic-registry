import React, { useState, useEffect } from 'react';
import { getPredictionHistory } from '../api';

const IoTDashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      const data = await getPredictionHistory();
      if (data.success) {
        setPredictions(data.data);
      }
    };

    // Fetch initially
    fetchHistory();

    // Set up polling every 1 second
    const interval = setInterval(() => {
      fetchHistory();
    }, 1000);

    setIsListening(true);

    return () => {
      clearInterval(interval);
      setIsListening(false);
    };
  }, []);

  return (
    <div className="card iot-card iot-dashboard">
      <h2>📡 IoT Live Stream Dashboard</h2>
      <p style={{ textAlign: 'center' }}>Listening for vehicle detections from deployed hardware sensors...</p>
      
      <div className="iot-status">
        <span>Listening Live</span>
      </div>

      <div className="prediction-list">
        <div className="prediction-list-header">
          <span>Time</span>
          <span>Vehicle</span>
          <span>Confidence</span>
        </div>
        {predictions.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#718096', padding: '32px' }}>
            Waiting for predictions from IoT sensors...
          </p>
        ) : (
          predictions.map((item, index) => (
            <div key={`${item.id}-${item.timestamp}-${index}`} className="prediction-item">
              <span className="time">{item.timestamp}</span>
              <span className="vehicle">{item.vehicle}</span>
              <span className="conf">{item.confidence.toFixed(1)}%</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default IoTDashboard;
