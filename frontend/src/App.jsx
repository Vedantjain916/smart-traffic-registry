import React, { useState } from 'react';
import PredictionCard from './components/PredictionCard';
import ModeToggle from './components/ModeToggle';
import IoTDashboard from './components/IoTDashboard';
import './index.css';

function App() {
  const [mode, setMode] = useState('software');

  return (
    <div className="app-container">
      <header>
        <h1>Smart Traffic Registry</h1>
        <p>Classify vehicle types using audio analysis</p>
      </header>

      <ModeToggle mode={mode} setMode={setMode} />

      <main>
        {mode === 'software' ? (
          <PredictionCard />
        ) : (
          <IoTDashboard />
        )}
      </main>
    </div>
  )
}

export default App;
