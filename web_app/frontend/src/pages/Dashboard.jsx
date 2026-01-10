import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import ECGGraph from '../components/ECGGraph';
import DiagnosisReport from '../components/DiagnosisReport';
import PatientForm from '../components/PatientForm';
import { Upload, Activity, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

function Dashboard() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [signalData, setSignalData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // New: Patient State
  const [patientInfo, setPatientInfo] = useState({ name: '', id: '' });

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(err => console.error("Failed to fetch models", err));
  }, []);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0]);
    }
  }, [models]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const result = await api.uploadRecord(file);
      if (result.signal) {
        setSignalData(result.signal);
      } else {
        setError("Invalid file format: No signal data found.");
      }
    } catch (err) {
      setError("Failed to process file.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!signalData.length || !selectedModel) return;

    // Validation: Require Patient ID or Name
    if (!patientInfo.id && !patientInfo.name) {
       setError("Please enter Patient Name or ID before analysis.");
       return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const result = await api.predict(selectedModel, signalData);
      setPrediction(result);
      
      // Save to History (Local Storage)
      const newRecord = {
          id: Date.now().toString(36),
          timestamp: new Date().toISOString(),
          patient: patientInfo,
          model: selectedModel,
          diagnosis: result.diagnosis,
          confidence: result.confidence
      };
      
      try {
          const history = JSON.parse(localStorage.getItem('ecg_history') || '[]');
          const updatedHistory = [newRecord, ...history].slice(0, 50); // Keep last 50
          localStorage.setItem('ecg_history', JSON.stringify(updatedHistory));
      } catch (e) {
          console.error("Failed to save history", e);
      }
      

      

    } catch (err) {
      setError("Analysis failed. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Patient Intake Section */}
      <PatientForm onUpdate={setPatientInfo} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Controls */}
        <div className="space-y-6">
          
          {/* Upload Card */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 relative overflow-hidden">
             
             {/* Decorative background circle */}
             <div className="absolute top-0 right-0 -mr-8 -mt-8 w-24 h-24 rounded-full bg-blue-50 opacity-50 pointer-events-none"></div>

            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 relative z-10">
              <Upload className="w-5 h-5 text-blue-600" />
              Upload ECG Record
            </h2>
            <div className="w-full relative z-10">
              <label 
                className={clsx(
                  "flex flex-col items-center justify-center w-full h-36 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-200 group",
                  signalData.length > 0 
                    ? "border-green-300 bg-green-50" 
                    : "border-gray-300 bg-gray-50 hover:bg-blue-50 hover:border-blue-300"
                )}
              >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
                      {signalData.length > 0 ? (
                        <>
                          <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mb-2">
                             <Activity className="w-5 h-5 text-green-600" />
                          </div>
                          <p className="text-sm text-green-700 font-medium">Signal Loaded Successfully</p>
                          <p className="text-xs text-green-600 mt-1">{signalData.length} samples</p>
                        </>
                      ) : (
                        <>
                          <Upload className="w-8 h-8 text-gray-400 mb-2 group-hover:text-blue-500 transition-colors" />
                          <p className="text-sm text-gray-500 group-hover:text-blue-600">
                             <span className="font-semibold">Click to upload</span>
                          </p>
                          <p className="text-xs text-gray-400 mt-1">JSON or CSV</p>
                        </>
                      )}
                  </div>
                  <input type="file" className="hidden" onChange={handleFileUpload} accept=".json,.csv" />
              </label>
            </div>
          </div>

          {/* Model Select Card */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-600" />
              AI Diagnosis
            </h2>
            
            <label className="block text-xs font-medium text-gray-500 mb-2 uppercase">Select Model</label>
            <div className="relative">
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-3 bg-gray-50 border border-gray-200 text-gray-900 text-sm rounded-lg focus:ring-2 focus:ring-purple-100 focus:border-purple-500 block outline-none transition-all"
              >
                {models.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <button 
              onClick={handleAnalyze}
              disabled={!signalData.length || loading}
              className={clsx(
                "w-full mt-6 text-white font-medium rounded-lg text-sm px-5 py-3.5 text-center transition-all shadow-lg flex items-center justify-center gap-2",
                !signalData.length || loading 
                  ? "bg-gray-300 cursor-not-allowed shadow-none"
                  : "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0"
              )}
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing Signal...
                </>
              ) : (
                'Run Diagnosis'
              )}
            </button>
            <p className="text-xs text-gray-400 mt-3 text-center">
               Estimated time: ~200ms
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-4 text-sm text-red-800 rounded-lg bg-red-50 border border-red-200 flex items-start gap-2" role="alert">
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <div>
                <span className="font-bold block">Action Required</span>
                {error}
              </div>
            </div>
          )}
          
        </div>

        {/* Right Column: Visualization & Report */}
        <div className="lg:col-span-2 space-y-6">
          <ECGGraph data={signalData} />
          
          {prediction && (
              <DiagnosisReport prediction={prediction} />
          )}
        </div>

      </div>
    </div>
  );
}

export default Dashboard;
