import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const api = {
  getModels: async () => {
    const response = await axios.get(`${API_URL}/models`);
    return response.data;
  },

  uploadRecord: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_URL}/upload`, formData);
    return response.data;
  },

  predict: async (modelName, signalData) => {
    const response = await axios.post(`${API_URL}/predict`, {
      model_name: modelName,
      signal_data: signalData
    });
    return response.data;
  }
};
