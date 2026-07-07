export const predictTrafficAudio = async (audioFile) => {
  const formData = new FormData();
  formData.append('audio', audioFile);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    return { success: false, error: error.message };
  }
};

export const getPredictionHistory = async () => {
  try {
    const response = await fetch('/api/history', {
      method: 'GET',
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    return { success: false, error: error.message };
  }
};
