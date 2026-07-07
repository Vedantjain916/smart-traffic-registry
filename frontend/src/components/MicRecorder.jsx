import React, { useState, useRef } from 'react';
import { predictTrafficAudio } from '../api';

// Helper to resample audio buffer to target sample rate
const resampleAudioBuffer = async (buffer, targetSampleRate) => {
  const offlineCtx = new OfflineAudioContext(1, buffer.duration * targetSampleRate, targetSampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(offlineCtx.destination);
  source.start();
  return await offlineCtx.startRendering();
};

// Helper to convert audio buffer to WAV
const audioBufferToWav = (buffer) => {
  const numChannels = 1; // Force mono
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;

  const resultLength = 44 + buffer.length * blockAlign;
  const resultBuffer = new ArrayBuffer(resultLength);
  const view = new DataView(resultBuffer);

  // Write WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + buffer.length * blockAlign, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, buffer.length * blockAlign, true);

  // Write audio data (mono only)
  const channelData = buffer.getChannelData(0);
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([resultBuffer], { type: 'audio/wav' });
};

const MicRecorder = ({ setResult, setError, setLoading }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioContextRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current);
        const arrayBuffer = await audioBlob.arrayBuffer();
        
        if (!audioContextRef.current) {
          audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
        // Resample to 16000 Hz (model expects this)
        const resampledBuffer = await resampleAudioBuffer(audioBuffer, 16000);
        const wavBlob = audioBufferToWav(resampledBuffer);
        const audioFile = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });

        setLoading(true);
        const result = await predictTrafficAudio(audioFile);
        if (result.success) {
          setResult(result);
          setError(null);
        } else {
          setError(result.error || 'Failed to analyze audio');
        }
        setLoading(false);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Stop after 4 seconds (matches dataset)
      setTimeout(() => {
        stopRecording();
      }, 4000);

    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Could not access microphone. Please allow microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
  };

  return (
    <button 
      onClick={isRecording ? stopRecording : startRecording} 
      className={isRecording ? 'recording' : ''}
    >
      {isRecording 
        ? `🔴 Recording... (${recordingTime}s)` 
        : '🎤 Record Live Audio (4s)'}
    </button>
  );
};

export default MicRecorder;
