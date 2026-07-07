import React from 'react';

const ModeToggle = ({ mode, setMode }) => {
  return (
    <div className="mode-toggle-container">
      <div className="mode-toggle">
        <button 
          className={mode === 'software' ? 'active' : ''} 
          onClick={() => setMode('software')}
        >
          📁 Software / Testing Mode
        </button>
        <button 
          className={mode === 'iot' ? 'active' : ''} 
          onClick={() => setMode('iot')}
        >
          📡 Hardware / IoT Mode
        </button>
      </div>
    </div>
  );
};

export default ModeToggle;
