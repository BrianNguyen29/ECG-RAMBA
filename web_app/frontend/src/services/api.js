import axios from 'axios';

// In development, Vite proxy forwards /api to backend (see vite.config.js)
// In production, set VITE_API_URL to the actual backend URL
const API_URL = import.meta.env.VITE_API_URL || '/api';

export const api = {
  /**
   * Get list of available model checkpoints
   */
  getModels: async () => {
    const response = await axios.get(`${API_URL}/models`);
    return response.data;
  },

  /**
   * Get API and model information
   */
  getInfo: async () => {
    const response = await axios.get(`${API_URL}/info`);
    return response.data;
  },

  /**
   * Get list of classification classes
   */
  getClasses: async () => {
    const response = await axios.get(`${API_URL}/classes`);
    return response.data;
  },

  /**
   * Upload ECG file and parse signal data
   * @param {File} file - ECG file (JSON, CSV, MAT, ZIP, NPY)
   * @param {Object} options - Optional parameters
   * @param {number} options.sampleRate - Sampling rate in Hz (required for CSV without header)
   * @param {number} options.durationSeconds - Known duration for auto-detect Fs
   * @returns {Object} { signal: [[lead1], [lead2], ...], num_leads: 12, samples: 5000, ... }
   */
  uploadRecord: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    
    if (options.sampleRate) {
      formData.append('sample_rate', options.sampleRate);
    }
    if (options.durationSeconds) {
      formData.append('duration_seconds', options.durationSeconds);
    }
    
    const response = await axios.post(`${API_URL}/upload`, formData);
    return response.data;
  },

  /**
   * Run ECG-RAMBA inference on 12-lead signal
   * @param {string} modelName - Checkpoint name e.g. "fold1_best.pt"
   * @param {Array} signalData - 12-lead signal [[lead1...], [lead2...], ...]
   * @returns {Object} Prediction result with diagnoses and insights
   */
  predict: async (modelName, signalData) => {
    const response = await axios.post(`${API_URL}/predict`, {
      model_name: modelName,
      signal_data: signalData
    });
    return response.data;
  },

  /**
   * Simple prediction for single-lead data (backward compatible)
   * @param {string} modelName 
   * @param {Array} signalData - Single lead [v1, v2, v3, ...]
   */
  predictSimple: async (modelName, signalData) => {
    const response = await axios.post(`${API_URL}/predict/simple`, {
      model_name: modelName,
      signal_data: signalData
    });
    return response.data;
  },

  /**
   * Run ECG inference on 12-lead ECG signal
   * @param {Array} signalData - 12-lead normalized signal [[lead1...], [lead2...], ...]
   * @param {Array} rawSignalData - Optional raw signal (before normalization) for amplitude features
   * @param {boolean} explain - If true, include saliency map and disentanglement
   * @param {Array} activeLeads - Optional array of 12 booleans for lead dropout
   * @param {string} mode - 'fast' (single fold) or 'accurate' (5-fold ensemble)
   * @returns {Object} Prediction result with diagnoses and insights
   */
  predictEnsemble: async (signalData, rawSignalData = null, explain = false, activeLeads = null, mode = 'accurate') => {
    const payload = {
      signal_data: signalData
    };
    if (rawSignalData) {
      payload.raw_signal_data = rawSignalData;
    }
    if (activeLeads) {
      payload.active_leads = activeLeads;
    }
    const response = await axios.post(`${API_URL}/predict/ensemble?explain=${explain}&mode=${mode}`, payload);
    return response.data;
  },

  /**
   * AI LAB: Analyze signal frequency components (PSD)
   */
  analyzeSignal: async (signal, fs = 250) => {
    const response = await axios.post(`${API_URL}/lab/analyze`, { signal, fs });
    return response.data;
  },

  /**
   * AI LAB: Apply digital filters
   */
  processSignal: async (signal, type, low, high, fs = 250) => {
    const response = await axios.post(`${API_URL}/lab/process`, { signal, fs, type, low, high });
    return response.data;
  }
};
