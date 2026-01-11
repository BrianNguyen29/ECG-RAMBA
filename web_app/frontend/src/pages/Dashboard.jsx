import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../services/api';
import ECGGraph from '../components/ECGGraph';
import DiagnosisReport from '../components/DiagnosisReport';
import StatisticsDashboard from '../components/StatisticsDashboard';
import FoldResults from '../components/FoldResults';
import PreprocessingInfo from '../components/PreprocessingInfo';
import ScientificLoader from '../components/ScientificLoader';
import { Card, CardContent, CardHeader, CardTitle, AnimatedCard, GlassCard } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input, Label } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { Upload, Activity, Zap, Heart, FileText, ChevronDown, Loader2, Database, CheckCircle, AlertTriangle, Sparkles } from 'lucide-react';
import clsx from 'clsx';

// Sample ECG data for demo (simulated 12-lead, 2 seconds at 500Hz = 1000 samples)
const generateSampleECG = () => {
  const samples = 5000;
  const leads = 12;
  const signal = [];
  
  for (let lead = 0; lead < leads; lead++) {
    const leadData = [];
    for (let i = 0; i < samples; i++) {
      const t = i / 500;
      // Simulate ECG-like waveform
      const heartRate = 72;
      const beatPeriod = 60 / heartRate;
      const phase = (t % beatPeriod) / beatPeriod;
      
      let value = 0;
      // P wave
      if (phase > 0.1 && phase < 0.2) {
        value += 0.15 * Math.sin((phase - 0.1) * Math.PI / 0.1);
      }
      // QRS complex
      if (phase > 0.25 && phase < 0.35) {
        const qrsPhase = (phase - 0.25) / 0.1;
        if (qrsPhase < 0.2) value -= 0.1;
        else if (qrsPhase < 0.5) value += 1.2 * (qrsPhase - 0.2) / 0.3;
        else if (qrsPhase < 0.7) value += 1.2 - 1.5 * (qrsPhase - 0.5) / 0.2;
        else value -= 0.3 * (1 - (qrsPhase - 0.7) / 0.3);
      }
      // T wave
      if (phase > 0.45 && phase < 0.65) {
        value += 0.3 * Math.sin((phase - 0.45) * Math.PI / 0.2);
      }
      // Add noise and lead variation
      value += (Math.random() - 0.5) * 0.05;
      value *= (1 + lead * 0.1);
      leadData.push(value);
    }
    signal.push(leadData);
  }
  return signal;
};

function Dashboard() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [signalData, setSignalData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'results' | 'statistics'
  const [patientInfo, setPatientInfo] = useState({ name: '', id: '' });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);  // Store full upload response with preprocessing info
  const [inputSampleRate, setInputSampleRate] = useState('');  // Sample rate for CSV files (empty = auto-detect)
  const [activeLeads, setActiveLeads] = useState(Array(12).fill(true)); // [Deep RAMBA] Lead states


  useEffect(() => {
    api.getModels()
      .then(m => {
        setModels(m);
        // Default to ensemble mode (first model)
        if (m.length > 0) setSelectedModel(m[0]);
      })
      .catch(err => console.error('Failed to fetch models', err));
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);
    setUploadResult(null);
    
    try {
      // Pass sample rate if specified (important for CSV files)
      const options = {};
      if (inputSampleRate && !isNaN(parseInt(inputSampleRate))) {
        options.sampleRate = parseInt(inputSampleRate);
      }
      const result = await api.uploadRecord(file, options);
      if (result.signal) {
        setSignalData(result.signal);
        setUploadResult(result);  // Save full result with preprocessing info
        setActiveTab('upload');
      } else if (result.error) {
        setError(result.error);
        if (result.details) {
          setError(`${result.error}: ${result.details.join(', ')}`);
        }
      } else {
        setError('Invalid file format: No signal data found.');
      }
    } catch (err) {
      setError('Failed to process file. Please check the format.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = () => {
    setSignalData(generateSampleECG());
    setPatientInfo({ name: 'Demo Patient', id: 'DEMO-001' });
    setError(null);
    setPrediction(null);
  };

  const handleAnalyze = async () => {
    if (!signalData.length) return;

    setLoading(true);
    setError(null);
    
    try {
      // Use 5-fold ensemble prediction for robust results
      // Pass raw_for_amplitude for accurate amplitude feature extraction (matches training pipeline)
      const rawSignal = uploadResult?.raw_for_amplitude || null;
      // Request explanation (Saliency Map) for "Scientific Principles" visualization
      // [Deep RAMBA] Pass activeLeads for robustness testing
      const result = await api.predictEnsemble(signalData, rawSignal, true, activeLeads);
      setPrediction(result);
      setActiveTab('results');
      
      // Save to history
      const newRecord = {
        id: Date.now().toString(36),
        timestamp: new Date().toISOString(),
        patient: patientInfo,
        model: selectedModel,
        diagnosis: result.top_diagnosis,
        confidence: result.confidence,
        predictions: result.predictions
      };
      
      try {
        const history = JSON.parse(localStorage.getItem('ecg_history') || '[]');
        const updatedHistory = [newRecord, ...history].slice(0, 50);
        localStorage.setItem('ecg_history', JSON.stringify(updatedHistory));
      } catch (e) {
        console.error('Failed to save history', e);
      }

    } catch (err) {
      setError('Analysis failed. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const numLeads = Array.isArray(signalData[0]) ? signalData.length : 1;
  const numSamples = Array.isArray(signalData[0]) ? signalData[0].length : signalData.length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-lg border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">ECG-RAMBA</h1>
                <p className="text-xs text-gray-500">AI-Powered ECG Analysis</p>
              </div>
            </div>
            
            {/* Quick Stats */}
            {signalData.length > 0 && (
              <div className="hidden md:flex items-center gap-6">
                <div className="text-right">
                  <p className="text-xs text-gray-500">Leads</p>
                  <p className="text-lg font-semibold text-gray-800">{numLeads}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500">Samples</p>
                  <p className="text-lg font-semibold text-gray-800">{numSamples.toLocaleString()}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500">Duration</p>
                  <p className="text-lg font-semibold text-gray-800">{(numSamples / 500).toFixed(1)}s</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Main Content Grid - Balanced Split Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
          
          {/* LEFT PANEL - INPUTS & CONFIGURATION */}
          <div className="space-y-6 flex flex-col">
            
            {/* 1. Upload Section */}
            <Card>
              <CardHeader>
                <CardTitle>
                  <Upload className="w-5 h-5 text-blue-600" />
                  1. Input Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                 <label className={clsx(
                  "flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300",
                  signalData.length > 0 
                    ? "border-green-300 bg-green-50/50" 
                    : "border-gray-300 bg-gray-50 hover:bg-blue-50 hover:border-blue-400"
                )}>
                   <div className="flex flex-col items-center justify-center py-6">
                    {signalData.length > 0 ? (
                      <>
                        <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mb-3">
                          <CheckCircle className="w-6 h-6 text-green-600" />
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-green-700">Signal Loaded Successfully</p>
                          <div className="flex items-center justify-center gap-1 mt-1">
                             <Badge variant={uploadResult?.sqi_passed ? "success" : "warning"}>
                               {uploadResult?.sqi_passed ? "High Quality (SQI Passed)" : "Noise Detected"}
                             </Badge>
                          </div>
                          <p className="text-xs text-gray-400 mt-1">{numLeads} leads × {numSamples} samples ({(numSamples/500).toFixed(1)}s)</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <Upload className="w-10 h-10 text-gray-400 mb-3" />
                        <p className="text-sm font-medium text-gray-600">Drag & Drop ECG File</p>
                        <p className="text-xs text-gray-400 mt-1">Support: JSON, CSV, MAT, ZIP</p>
                      </>
                    )}
                  </div>
                  <input 
                    type="file" 
                    className="hidden" 
                    onChange={handleFileUpload} 
                    accept=".json,.csv,.mat,.zip" 
                  />
                </label>

                 <div className="flex items-center gap-4 my-4">
                  <div className="flex-1 h-px bg-gray-200"></div>
                  <span className="text-xs text-gray-400 font-medium">QUICK START</span>
                  <div className="flex-1 h-px bg-gray-200"></div>
                </div>

                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={handleLoadSample}
                >
                  <Database className="w-4 h-4 mr-2" />
                  Load Demo Data (Normal Sinus)
                </Button>
              </CardContent>
            </Card>

            {/* 2. Configuration & Analysis */}
            <Card className="flex-1">
              <CardHeader>
                <CardTitle>
                    <Zap className="w-5 h-5 text-yellow-500" />
                    2. Analysis Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                 {/* Model Selection */}
                 <div>
                    <Label>AI Engine</Label>
                    <select 
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full h-11 rounded-xl border border-gray-200 bg-gray-50/50 px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-blue-500/20"
                    >
                      {models.map(m => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                 </div>

                 {/* Patient Metadata */}
                 <div>
                    <Label>Patient ID</Label>
                    <Input
                      type="text"
                      value={patientInfo.id}
                      onChange={(e) => setPatientInfo({ ...patientInfo, id: e.target.value })}
                      placeholder="Optional identifier..."
                    />
                 </div>

                 {/* Run Button */}
                 <Button 
                  onClick={handleAnalyze}
                  disabled={!signalData.length || loading}
                  variant={!signalData.length || loading ? "secondary" : "gradient"}
                  className="w-full h-12 text-base mt-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin mr-2" />
                      Processing Pipeline...
                    </>
                  ) : (
                    <>
                      <Activity className="w-5 h-5 mr-2" />
                      RUN DIAGNOSIS
                    </>
                  )}
                </Button>
                
                {error && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* RIGHT PANEL - VISUALIZATION & RESULTS */}
          <div className="space-y-6 flex flex-col h-full">
            
            {/* Visualization Card (Higher Priority) */}
            <Card className="flex flex-col min-h-[400px] border-0 ring-1 ring-gray-200">
              <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50 rounded-t-2xl">
                <div className="flex items-center gap-4">
                    <h3 className="font-semibold text-gray-800 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-blue-500" />
                        ECG Visualization
                    </h3>
                    {/* Active Lead Selectors (Compact) */}
                    {signalData.length > 0 && (
                        <div className="hidden xl:flex gap-1">
                             {['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'].map((lead, idx) => (
                                <button
                                    key={lead}
                                    onClick={() => {
                                        const newLeads = [...activeLeads];
                                        newLeads[idx] = !newLeads[idx];
                                        setActiveLeads(newLeads);
                                    }}
                                    className={clsx(
                                        "w-6 h-6 text-[9px] rounded flex items-center justify-center transition-all font-bold",
                                        activeLeads[idx] 
                                            ? "bg-blue-100 text-blue-700 hover:bg-blue-200" 
                                            : "bg-gray-100 text-gray-400 line-through"
                                    )}
                                >
                                    {lead}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
                
                <div className="flex items-center gap-2">
                     <span className="text-[10px] text-gray-400 uppercase font-semibold tracking-wider">Standard 25mm/s</span>
                     {prediction?.inference_time_s && (
                        <Badge variant="success" className="font-mono">
                            {prediction.inference_time_s}s
                        </Badge>
                    )}
                </div>
              </div>
              
              <div className="flex-1 p-0 relative bg-white rounded-b-2xl">
                <div className="absolute inset-0 p-4">
                     <ECGGraph 
                        data={signalData} 
                        saliencyMap={prediction?.saliency_map} 
                        activeLeads={activeLeads}
                    />
                </div>
              </div>
            </Card>

            {/* Results Section */}
            <div className="flex-1 min-h-[300px]">
                {loading ? (
                    <div className="h-full">
                        <ScientificLoader />
                    </div>
                ) : prediction ? (
                     <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {/* Tab Toggle for Results */}
                        <div className="flex gap-2 bg-gray-100/50 p-1 rounded-xl w-fit mx-auto mb-4 border border-gray-200/50">
                           {[
                              { id: 'results', label: 'Clinical Report', icon: FileText },
                              { id: 'statistics', label: 'Model Statistics', icon: Activity },
                            ].map(tab => (
                              <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={clsx(
                                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                                  activeTab === tab.id 
                                    ? "bg-white shadow-sm text-blue-600 ring-1 ring-black/5" 
                                    : "text-gray-500 hover:text-gray-700"
                                )}
                              >
                                <tab.icon className="w-3.5 h-3.5" />
                                {tab.label}
                              </button>
                            ))}
                        </div>

                        {activeTab === 'results' ? (
                            <DiagnosisReport prediction={prediction} />
                        ) : (
                            <div className="space-y-6">
                                <FoldResults prediction={prediction} />
                                <StatisticsDashboard prediction={prediction} />
                            </div>
                        )}
                     </div>
                ) : (
                    <Card className="h-full flex flex-col items-center justify-center p-8 text-center border-dashed bg-gray-50/50">
                        <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-4">
                            <Activity className="w-8 h-8 text-blue-300" />
                        </div>
                        <h3 className="text-gray-900 font-medium">Ready for Diagnostics</h3>
                        <p className="text-gray-500 text-sm mt-1 max-w-xs mx-auto">
                            Import ECG data on the left to generate detailed AI analysis.
                        </p>
                    </Card>
                )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Dashboard;
