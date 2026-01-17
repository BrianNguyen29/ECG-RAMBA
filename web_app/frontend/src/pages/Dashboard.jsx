import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../services/api';
import ECGGraph from '../components/ECGGraph';
import EEGViewer from '../components/EEGViewer';
import DiagnosisReport from '../components/DiagnosisReport';
import ScientificLoader from '../components/ScientificLoader';
import ResearchLab from '../components/ResearchLab';
import EducationMode from '../components/EducationMode';
import AIInsightPanel from '../components/AIInsightPanel';
import TutorChat from '../components/TutorChat';
import ErrorBoundary from '../components/ErrorBoundary';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Upload, Activity, Zap, Heart, FileText, Loader2, Database, Search, User, Filter, AlertTriangle, Layers, GraduationCap, Microscope, LayoutDashboard, Moon, Sun, Eye, EyeOff, Cpu, Brain, Printer } from 'lucide-react';
import { useToast } from '../components/ui/use-toast';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import clsx from 'clsx';

// --- MOCK DATA FOR PATIENT QUEUE ---
const MOCK_QUEUE = [
  { id: 'PT-1092', name: 'Nguyen Van A', age: 54, time: '10:30', status: 'pending' },
  { id: 'PT-1093', name: 'Le Thi B', age: 62, time: '10:45', status: 'processed' },
  { id: 'PT-1094', name: 'Tran Van C', age: 41, time: '11:00', status: 'urgent' },
];

const generateSampleECG = () => {
  // Realistic ECG simulator with proper PQRST morphology
  const samples = 2500; // 5 seconds @ 500Hz
  const leads = 12;
  const sampleRate = 500;
  const heartRate = 72; // BPM
  const samplesPerBeat = Math.floor(60 / heartRate * sampleRate);
  
  // Gaussian function for smooth wave shapes
  const gaussian = (x, amp, mu, sigma) => {
    return amp * Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma, 2)));
  };
  
  // Generate a single realistic ECG beat (0-1 normalized position)
  const generateBeat = (t) => {
    // P wave: small positive bump at ~0.1-0.2
    const pWave = gaussian(t, 0.15, 0.12, 0.035);
    
    // QRS complex: sharp spike at ~0.25-0.35
    const qWave = gaussian(t, -0.12, 0.24, 0.012); // Q dip
    const rWave = gaussian(t, 1.2, 0.28, 0.018);   // R peak (tall)
    const sWave = gaussian(t, -0.25, 0.32, 0.015); // S dip
    
    // T wave: rounded positive bump at ~0.5-0.7
    const tWave = gaussian(t, 0.35, 0.55, 0.06);
    
    // Small U wave (optional, subtle)
    const uWave = gaussian(t, 0.05, 0.72, 0.03);
    
    return pWave + qWave + rWave + sWave + tWave + uWave;
  };
  
  // Lead-specific morphology variations
  const leadFactors = [
    { amp: 1.0, inv: false },  // I
    { amp: 1.2, inv: false },  // II (tallest in limb leads)
    { amp: 0.9, inv: false },  // III
    { amp: 0.8, inv: true },   // aVR (inverted)
    { amp: 0.7, inv: false },  // aVL
    { amp: 1.0, inv: false },  // aVF
    { amp: 0.6, inv: false },  // V1 (small R, deep S)
    { amp: 0.8, inv: false },  // V2
    { amp: 1.0, inv: false },  // V3
    { amp: 1.3, inv: false },  // V4 (tallest precordial)
    { amp: 1.2, inv: false },  // V5
    { amp: 1.0, inv: false },  // V6
  ];
  
  const signal = [];
  for (let lead = 0; lead < leads; lead++) {
    const leadData = [];
    const factor = leadFactors[lead] || { amp: 1, inv: false };
    
    for (let i = 0; i < samples; i++) {
      // Calculate position within current beat cycle (0-1)
      const beatPosition = (i % samplesPerBeat) / samplesPerBeat;
      
      // Generate ECG value
      let val = generateBeat(beatPosition) * factor.amp;
      
      // Invert for aVR
      if (factor.inv) val = -val;
      
      // Add small baseline wander (low frequency drift)
      val += Math.sin(i / 1000) * 0.02;
      
      // Add minimal high-frequency noise
      val += (Math.random() - 0.5) * 0.015;
      
      leadData.push(val);
    }
    signal.push(leadData);
  }
  return signal;
};

export default function Dashboard() {
  const { toast } = useToast();
  // -- STATE --
  const [activeTab, setActiveTab] = useState('clinical'); // 'clinical' | 'education' | 'research'
  const [signalData, setSignalData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [patientInfo, setPatientInfo] = useState({ name: '', id: '' });
  const [uploadResult, setUploadResult] = useState(null);
  const [activeLeads, setActiveLeads] = useState(Array(12).fill(true));
  const [inferenceMode, setInferenceMode] = useState('accurate'); // 'fast' | 'accurate'
  const [ecgDarkMode, setEcgDarkMode] = useState(false); // Toggle ECG dark/light
  const [showAIAttention, setShowAIAttention] = useState(true); // AI overlay on ECG
  const [showWaveAnnotations, setShowWaveAnnotations] = useState(true); // P/QRS/T wave markers

  // -- HANDLERS --
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);
    setUploadResult(null);

    try {
      const result = await api.uploadRecord(file);
      if (result.signal) {
        setSignalData(result.signal);
        setUploadResult(result);
        setPatientInfo(prev => ({ ...prev, id: file.name.split('.')[0] }));
      } else {
        setError('Invalid file format. Please upload a valid ECG record.');
      }
    } catch (err) {
      setError('Upload failed. Check network connection.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = useCallback(() => {
    setSignalData(generateSampleECG());
    setPatientInfo({ name: 'Demo Patient', id: 'DEMO-001' });
    setError(null);
    setPrediction(null);
  }, []);

  const handleAnalyze = async () => {
    if (!signalData.length) return;
    
    // Skip analysis for EEG for now (Phase 9.3)
    if (uploadResult?.modality === 'EEG') {
        setError("AI Analysis for EEG is coming in Phase 9.3!");
        return;
    }

    setLoading(true);
    setError(null);
    try {
      const rawSignal = uploadResult?.raw_for_amplitude || null;
      // Pass inferenceMode to control speed vs accuracy
      const result = await api.predictEnsemble(signalData, rawSignal, true, activeLeads, inferenceMode);
      if (result.error) {
           setError(result.error); // Capture backend error
      } else {
           setPrediction(result);
      }
    } catch (err) {
      setError('Analysis request failed. Please ensure Backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const toggleLead = useCallback((idx) => {
    setActiveLeads(prev => {
      const newLeads = [...prev];
      newLeads[idx] = !newLeads[idx];
      return newLeads;
    });
  }, []);

  const hasData = signalData.length > 0;
  // Determine Modality
  const isEEG = uploadResult?.modality === 'EEG';
  const modalityTitle = isEEG ? 'EEG Viewer' : 'ECG Viewer';
  const modalityIcon = isEEG ? Brain : Activity;

    // PDF Export Handler
    const handleExportPDF = async () => {
        const input = document.getElementById('ecg-cockpit-container'); // We will ID the main container
        if (!input) {
            toast({
                title: "Error",
                description: "Could not find ECG view to export.",
                variant: "destructive"
            });
            return;
        }

        try {
            toast({ title: "Generating Report...", description: "Please wait while we prepare the PDF." });
            
            // Capture the ECG Cockpit area
            const canvas = await html2canvas(input, {
                scale: 2, // High res
                useCORS: true,
                logging: false,
                backgroundColor: '#ffffff'
            });

            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('l', 'mm', 'a4'); // Landscape, mm, A4
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();
            
            // Add Header
            pdf.setFontSize(18);
            pdf.text("ECG-RAMBA Clinical Report", 10, 15);
            
            pdf.setFontSize(10);
            pdf.text(`Patient ID: ${patientInfo.id || 'Unknown'}`, 10, 25);
            pdf.text(`Date: ${new Date().toLocaleString()}`, 10, 30);
            if (prediction) {
                 pdf.text(`AI Diagnosis: ${prediction.class}`, 200, 25);
            }
            
            // Add Image (Fit to page)
            const imgProps = pdf.getImageProperties(imgData);
            const ratio = imgProps.width / imgProps.height;
            const printWidth = pdfWidth - 20;
            const printHeight = printWidth / ratio;
            
            pdf.addImage(imgData, 'PNG', 10, 40, printWidth, printHeight);
            
            // Add Footer
            pdf.setFontSize(8);
            pdf.setTextColor(150);
            pdf.text("Generated by ECG-RAMBA Clinical Decision Support System. Not for definitive diagnosis.", 10, pdfHeight - 10);
            
            pdf.save(`ECG_Report_${patientInfo.id || 'DEMO'}.pdf`);
            
            toast({ title: "Success", description: "Report downloaded successfully." });

        } catch (err) {
            console.error(err);
            toast({ title: "Export Failed", description: "Something went wrong.", variant: "destructive" });
        }
    };

  return (
    <div className="h-full w-full bg-slate-50 text-slate-900 font-sans flex flex-col overflow-hidden selection:bg-blue-100">
      
      {/* --- PERSPECTIVE CONTROLLER (TAB BAR) --- */}
      <header className="h-14 border-b border-slate-200 bg-white/80 backdrop-blur flex items-center justify-between px-4 z-20 shrink-0 shadow-sm">
          <div className="flex items-center gap-4">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-widest hidden md:block">Clinical Cockpit</span>
              <div className="flex bg-slate-100 border border-slate-200 rounded-lg p-1">
                  {[
                      { id: 'clinical', label: 'Clinical View', icon: LayoutDashboard },
                      { id: 'education', label: 'Tutor', icon: GraduationCap },
                      { id: 'research', label: 'Research Lab', icon: Microscope },
                  ].map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all",
                            activeTab === tab.id 
                                ? "bg-white text-blue-600 shadow-sm border border-slate-200" 
                                : "text-slate-500 hover:text-slate-700 hover:bg-slate-200"
                        )}
                      >
                          <tab.icon className="w-3.5 h-3.5" />
                          {tab.label}
                      </button>
                  ))}
              </div>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
              <button 
                  onClick={() => setActiveTab(activeTab === 'education' ? 'clinical' : 'education')}
                  className={clsx(
                      "px-2 py-1 rounded border flex items-center gap-1 transition-colors",
                      activeTab === 'education' 
                          ? "bg-indigo-600 text-white border-indigo-700" 
                          : "bg-white text-slate-500 border-slate-200 hover:text-indigo-600"
                  )}
              >
                  <GraduationCap className="w-3 h-3" />
                  {activeTab === 'education' ? 'Education Mode On' : 'Enable Tutor'}
              </button>
              <div className="w-px h-4 bg-slate-300 mx-1" />
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              System Ready
          </div>
      </header>

      {/* --- MAIN CONTENT AREA --- */}
      <main className="flex-1 overflow-hidden relative">
          
          <AnimatePresence mode="wait">
              {activeTab === 'clinical' && (
                  <motion.div 
                    key="clinical"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="h-full grid grid-cols-24 gap-4 p-4 overflow-hidden"
                  >
                      {/* --- COL 1: SIDEBAR / QUEUE (SPAN 4) --- */}
                      <section className="col-span-24 xl:col-span-4 flex flex-col gap-4 h-full">
                        {/* Profile / Input Card */}
                        <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col gap-3 shadow-sm">
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <User className="w-3 h-3" /> Current Patient
                            </h3>
                            <Input 
                                placeholder="Patient ID" 
                                className="bg-slate-50 border-slate-200 text-xs h-8 text-slate-900 placeholder:text-slate-400"
                                value={patientInfo.id}
                                onChange={e => setPatientInfo({...patientInfo, id: e.target.value})}
                            />
                            
                             <label className="cursor-pointer bg-blue-50 hover:bg-blue-100 border border-blue-200 border-dashed rounded-lg p-3 flex flex-col items-center justify-center transition-all group">
                                <Upload className="w-5 h-5 text-blue-500 mb-1 group-hover:scale-110 transition-transform" />
                                <span className="text-[10px] text-blue-600 font-medium">Upload ECG</span>
                                <input type="file" className="hidden" onChange={handleFileUpload} accept=".json,.csv,.mat,.zip" />
                            </label>
                            
                            {/* Inference Mode Toggle */}
                            <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
                                <p className="text-[9px] text-slate-400 uppercase font-bold tracking-wider mb-1.5 text-center">Analysis Mode</p>
                                <div className="flex gap-1">
                                    <button
                                        onClick={() => setInferenceMode('fast')}
                                        className={clsx(
                                            "flex-1 py-1.5 px-2 rounded-md text-[10px] font-semibold transition-all flex items-center justify-center gap-1",
                                            inferenceMode === 'fast' 
                                                ? "bg-amber-100 text-amber-700 border border-amber-300 shadow-sm" 
                                                : "bg-white text-slate-500 border border-slate-200 hover:bg-slate-100"
                                        )}
                                    >
                                        <Zap className="w-3 h-3" />
                                        Fast
                                    </button>
                                    <button
                                        onClick={() => setInferenceMode('accurate')}
                                        className={clsx(
                                            "flex-1 py-1.5 px-2 rounded-md text-[10px] font-semibold transition-all flex items-center justify-center gap-1",
                                            inferenceMode === 'accurate' 
                                                ? "bg-blue-100 text-blue-700 border border-blue-300 shadow-sm" 
                                                : "bg-white text-slate-500 border border-slate-200 hover:bg-slate-100"
                                        )}
                                    >
                                        <Layers className="w-3 h-3" />
                                        Accurate
                                    </button>
                                </div>
                                <p className="text-[8px] text-slate-400 text-center mt-1">
                                    {inferenceMode === 'fast' ? '1 model (~5s)' : '5-fold parallel (~8s)'}
                                </p>
                            </div>
                            
                            <Button variant="ghost" className="text-xs h-7 text-slate-500 hover:text-blue-600 hover:bg-blue-50" onClick={handleExportPDF} disabled={signalData.length === 0}>
                                <Printer className="w-3 h-3 mr-2" /> Export PDF Report
                            </Button>
                        </div>

                        {/* Patient Queue List */}
                        <div className="flex-1 bg-white border border-slate-200 rounded-xl overflow-hidden flex flex-col shadow-sm">
                            <div className="p-3 border-b border-slate-200 flex justify-between items-center bg-slate-50">
                                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Worklist</h3>
                                <Filter className="w-3 h-3 text-slate-400" />
                            </div>
                            <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin">
                                {MOCK_QUEUE.map((pt) => (
                                    <div key={pt.id} className="p-2 rounded-lg hover:bg-slate-100 cursor-pointer flex items-center justify-between group transition-colors">
                                        <div>
                                            <p className="text-xs font-bold text-slate-700 group-hover:text-blue-600">{pt.id}</p>
                                            <p className="text-[10px] text-slate-500">{pt.name}</p>
                                        </div>
                                        <Badge className={clsx("text-[9px] h-4 px-1 border shadow-none", 
                                            pt.status === 'urgent' ? 'bg-rose-50 text-rose-600 border-rose-200' : 
                                            pt.status === 'processed' ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : 'bg-slate-100 text-slate-500 border-slate-200'
                                        )}>
                                            {pt.status}
                                        </Badge>
                                    </div>
                                ))}
                            </div>
                        </div>
                      </section>

                      {/* --- COL 2: EXPANDED ECG COCKPIT (SPAN 15) --- */}
                      <section 
                          id="ecg-cockpit-container"
                          className={clsx(
                          "col-span-24 xl:col-span-15 flex flex-col h-full rounded-2xl shadow-md overflow-hidden relative border",
                          ecgDarkMode ? "bg-slate-900 border-slate-700" : "bg-white border-slate-200"
                      )}>
                         {/* ECG HEADER with Controls */}
                         <header className={clsx(
                             "h-12 border-b flex items-center justify-between px-4",
                             ecgDarkMode ? "bg-slate-800 border-slate-700" : "bg-slate-50 border-slate-200"
                         )}>
                             <div className="flex items-center gap-2">
                                 {React.createElement(modalityIcon, { className: clsx("w-4 h-4", ecgDarkMode ? "text-cyan-400" : "text-blue-600") })}
                                 <span className={clsx(
                                     "text-xs font-bold uppercase tracking-widest",
                                     ecgDarkMode ? "text-slate-400" : "text-slate-500"
                                 )}>{modalityTitle}</span>
                                 <Badge variant="outline" className={clsx(
                                     "text-[9px] ml-2",
                                     ecgDarkMode ? "border-slate-600 text-slate-400 bg-slate-800" : "border-slate-300 text-slate-500 bg-white"
                                 )}>
                                     {isEEG ? (uploadResult?.num_channels ? `${uploadResult.num_channels} Channels` : 'EEG Signal') : '12-Lead Stack'}
                                 </Badge>
                             </div>
                             
                             {/* ECG Controls */}
                             <div className="flex items-center gap-2">
                                 {/* Dark Mode Toggle */}
                                 <button
                                     onClick={() => setEcgDarkMode(!ecgDarkMode)}
                                     className={clsx(
                                         "p-1.5 rounded-md transition-all border",
                                         ecgDarkMode 
                                             ? "bg-cyan-900/50 border-cyan-700 text-cyan-400 hover:bg-cyan-900" 
                                             : "bg-slate-100 border-slate-200 text-slate-500 hover:bg-slate-200"
                                     )}
                                     title={ecgDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                                 >
                                     {ecgDarkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                                 </button>
                                 
                                 {/* AI Attention Toggle */}
                                 <button
                                     onClick={() => setShowAIAttention(!showAIAttention)}
                                     className={clsx(
                                         "p-1.5 rounded-md transition-all border flex items-center gap-1",
                                         showAIAttention 
                                             ? "bg-rose-50 border-rose-200 text-rose-600" 
                                             : "bg-slate-100 border-slate-200 text-slate-500"
                                     )}
                                     title={showAIAttention ? "Hide AI Attention" : "Show AI Attention"}
                                 >
                                     {showAIAttention ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                                     <span className="text-[9px] font-semibold hidden sm:inline">AI</span>
                                 </button>
                                 
                                 {/* Wave Annotations Toggle (P/QRS/T) */}
                                 <button
                                     onClick={() => setShowWaveAnnotations(!showWaveAnnotations)}
                                     className={clsx(
                                         "p-1.5 rounded-md transition-all border flex items-center gap-1",
                                         showWaveAnnotations 
                                             ? (ecgDarkMode ? "bg-purple-900/50 border-purple-700 text-purple-400" : "bg-purple-50 border-purple-200 text-purple-600")
                                             : "bg-slate-100 border-slate-200 text-slate-500"
                                     )}
                                     title={showWaveAnnotations ? "Hide Wave Markers" : "Show P/QRS/T Markers"}
                                 >
                                     <Activity className="w-3.5 h-3.5" />
                                     <span className="text-[9px] font-semibold hidden sm:inline">Waves</span>
                                 </button>
                                 
                                 {/* Inference Time Badge */}
                                 {prediction?.inference_time_s && (
                                     <div className={clsx(
                                         "flex items-center gap-1.5 text-[10px] font-mono px-2 py-1 rounded-md border",
                                         ecgDarkMode 
                                             ? "bg-emerald-900/30 border-emerald-700 text-emerald-400" 
                                             : "bg-emerald-50 border-emerald-200 text-emerald-600"
                                     )}>
                                         <Zap className="w-3 h-3" /> 
                                         {prediction.inference_time_s}s
                                     </div>
                                 )}
                             </div>
                         </header>

                         {/* GRAPH AREA - Wrapped in ErrorBoundary */}
                         <div className={clsx(
                             "flex-1 relative",
                             ecgDarkMode ? "ecg-grid-dark" : "ecg-grid-light"
                         )}>
                             <ErrorBoundary>
                                {isEEG ? (
                                    <EEGViewer 
                                        data={signalData}
                                        channels={uploadResult?.channels}
                                        sampleRate={uploadResult?.sample_rate || 250}
                                        warnings={uploadResult?.warnings}
                                        darkMode={ecgDarkMode}
                                    />
                                ) : (
                                    <ECGGraph 
                                        data={signalData} 
                                        saliencyMap={showAIAttention ? prediction?.saliency_map : null} 
                                        activeLeads={activeLeads}
                                        annotations={showWaveAnnotations ? prediction?.clinical_features : null}
                                        darkMode={ecgDarkMode}
                                    />
                                )}
                             </ErrorBoundary>
                         </div>

                         {/* LEAD CONTROLS FOOTER */}
                         {hasData && !isEEG && (
                             <div className={clsx(
                                 "h-14 border-t flex items-center px-4 gap-4 overflow-x-auto scrollbar-thin",
                                 ecgDarkMode ? "bg-slate-800 border-slate-700" : "bg-slate-50 border-slate-200"
                             )}>
                                 <span className={clsx(
                                     "text-[10px] font-bold uppercase whitespace-nowrap",
                                     ecgDarkMode ? "text-slate-500" : "text-slate-400"
                                 )}>Active Leads:</span>
                                 <div className="flex gap-1">
                                     {['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'].map((lead, idx) => (
                                         <button
                                            key={lead}
                                            onClick={() => toggleLead(idx)}
                                            className={clsx(
                                                "h-6 px-2 rounded text-[10px] font-mono font-bold transition-all border",
                                                activeLeads[idx] 
                                                    ? (ecgDarkMode 
                                                        ? "bg-cyan-900/50 border-cyan-700 text-cyan-400" 
                                                        : "bg-blue-50 border-blue-200 text-blue-600")
                                                    : (ecgDarkMode 
                                                        ? "bg-transparent border-slate-700 text-slate-600 line-through opacity-50" 
                                                        : "bg-transparent border-slate-200 text-slate-400 line-through opacity-50")
                                            )}
                                         >
                                             {lead}
                                         </button>
                                     ))}
                                 </div>
                                  <div className={clsx("h-4 w-px mx-2", ecgDarkMode ? "bg-slate-700" : "bg-slate-300")} />
                                 
                                 {/* Preset Buttons */}
                                 <div className="flex gap-1">
                                     <Button 
                                         size="sm" 
                                         variant="outline" 
                                         className={clsx(
                                             "text-[9px] h-5 px-2",
                                             ecgDarkMode ? "border-slate-600 text-slate-400 hover:bg-slate-700" : "border-slate-200 text-slate-500 hover:bg-slate-100"
                                         )} 
                                         onClick={() => setActiveLeads([true, true, true, true, true, true, false, false, false, false, false, false])}
                                     >
                                        Limb
                                     </Button>
                                     <Button 
                                         size="sm" 
                                         variant="outline" 
                                         className={clsx(
                                             "text-[9px] h-5 px-2",
                                             ecgDarkMode ? "border-cyan-700 text-cyan-400 hover:bg-cyan-900/50" : "border-blue-200 text-blue-600 hover:bg-blue-50"
                                         )} 
                                         onClick={() => setActiveLeads([false, false, false, false, false, false, true, true, true, true, true, true])}
                                     >
                                        V1-V6
                                     </Button>
                                     <Button 
                                         size="sm" 
                                         variant="ghost" 
                                         className={clsx(
                                             "text-[9px] h-5 px-2",
                                             ecgDarkMode ? "text-slate-400 hover:text-cyan-400" : "text-slate-500 hover:text-blue-600"
                                         )} 
                                         onClick={() => setActiveLeads(Array(12).fill(true))}
                                     >
                                        All
                                     </Button>
                                 </div>
                             </div>
                         )}
                         
                         {/* PERFORMANCE FOOTER BAR */}
                         <div className={clsx(
                             "h-8 border-t flex items-center justify-between px-4 text-[9px] font-mono",
                             ecgDarkMode 
                                 ? "bg-slate-900 border-slate-800 text-slate-500" 
                                 : "bg-slate-100 border-slate-200 text-slate-400"
                         )}>
                             <div className="flex items-center gap-4">
                                 <span className="flex items-center gap-1"><Cpu className="w-3 h-3" /> Backend: {isEEG ? 'MNE-Python' : 'Mamba2-SSD'}</span>
                                 <span>|</span>
                                 <span>Device: CPU (ONNX Optimized)</span>
                             </div>
                             <div className="flex items-center gap-1">
                                 <div className={clsx(
                                     "w-1.5 h-1.5 rounded-full animate-pulse",
                                     ecgDarkMode ? "bg-emerald-400" : "bg-emerald-500"
                                 )} />
                                 <span>Real-time</span>
                             </div>
                         </div>
                      </section>

                      {/* --- COL 3: AI INSIGHT PANEL (SPAN 3) --- */}
                      <section className={clsx(
                          "col-span-24 xl:col-span-5 flex flex-col h-full rounded-2xl shadow-md overflow-hidden border min-w-[220px]",
                          ecgDarkMode ? "bg-slate-800 border-slate-700" : "bg-white border-slate-200"
                      )}>
                         {/* ERROR ALERT (if any) */}
                         {error && (
                             <div className="bg-rose-50 border-b border-rose-200 text-rose-800 px-4 py-3 flex items-start gap-3">
                                 <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                                 <div>
                                     <p className="font-bold text-xs">Error</p>
                                     <p className="text-[10px] opacity-90">{error}</p>
                                 </div>
                             </div>
                         )}
                         
                         {/* Ready State - Show Analyze Button */}
                         {hasData && !prediction && !loading && (
                             <div className={clsx(
                                 "flex-1 flex flex-col items-center justify-center gap-4 p-6 text-center",
                                 ecgDarkMode ? "bg-slate-800" : "bg-slate-50"
                             )}>
                                 <div className={clsx(
                                     "w-16 h-16 rounded-full flex items-center justify-center ring-1",
                                     ecgDarkMode ? "bg-cyan-900/50 ring-cyan-700" : "bg-blue-50 ring-blue-100"
                                 )}>
                                     <Activity className={clsx("w-8 h-8", ecgDarkMode ? "text-cyan-400" : "text-blue-500")} />
                                 </div>
                                 <div>
                                     <h3 className={clsx("text-lg font-bold", ecgDarkMode ? "text-white" : "text-slate-900")}>Ready to Analyze</h3>
                                     <p className={clsx("text-xs mt-1", ecgDarkMode ? "text-slate-400" : "text-slate-500")}>
                                         {patientInfo.id ? `Record: ${patientInfo.id}` : 'Signal loaded'}<br/>12 Leads • 500Hz
                                     </p>
                                 </div>
                                 <Button 
                                     onClick={handleAnalyze} 
                                     disabled={isEEG}
                                     className={clsx(
                                         "w-full max-w-[200px] text-white shadow-lg h-10",
                                         isEEG ? "bg-slate-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 shadow-blue-500/20"
                                     )}
                                 >
                                     <Zap className="w-4 h-4 mr-2 fill-current" /> 
                                     {isEEG ? 'Analysis (Coming 9.3)' : 'Analyze ECG'}
                                 </Button>
                                 <p className={clsx("text-[10px]", ecgDarkMode ? "text-slate-500" : "text-slate-400")}>
                                     {isEEG ? 'Visualization Only' : (inferenceMode === 'fast' ? 'Fast Mode (~5s)' : 'Accurate Mode (~8s)')}
                                 </p>
                             </div>
                         )}
                         
                         {/* Loading State */}
                         {loading && (
                             <div className="flex-1 flex items-center justify-center">
                                 <ScientificLoader />
                             </div>
                         )}
                         
                         {/* AI Insight Panel (Results) */}
                         {prediction && !loading && (
                             <AIInsightPanel prediction={prediction} darkMode={ecgDarkMode} />
                         )}
                         
                         {/* Empty State */}
                         {!hasData && (
                             <div className={clsx(
                                 "flex-1 flex flex-col items-center justify-center gap-4 p-6 text-center",
                                 ecgDarkMode ? "bg-slate-800" : "bg-slate-50"
                             )}>
                                 <div className={clsx(
                                     "w-16 h-16 rounded-full flex items-center justify-center",
                                     ecgDarkMode ? "bg-slate-700" : "bg-slate-100"
                                 )}>
                                     <Database className={clsx("w-8 h-8", ecgDarkMode ? "text-slate-600" : "text-slate-400")} />
                                 </div>
                                 <p className={clsx("text-xs", ecgDarkMode ? "text-slate-500" : "text-slate-400")}>
                                     Select patient or upload ECG<br/>to see AI insights
                                 </p>
                             </div>
                         )}
                      </section>
                  </motion.div>
              )}

              {activeTab === 'education' && (
                  <motion.div 
                    key="education"
                    initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.2 }}
                    className="h-full p-4"
                  >
                      <EducationMode signalData={signalData} prediction={prediction} />
                  </motion.div>
              )}

              {activeTab === 'research' && (
                  <motion.div 
                    key="research"
                    initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }}
                    transition={{ duration: 0.2 }}
                    className="h-full"
                  >
                      <ResearchLab 
    originalSignal={signalData} 
    patientId={patientInfo.id} 
    sampleRate={uploadResult?.input_sample_rate || 500}
/>
                  </motion.div>
              )}
          </AnimatePresence>

          {/* DEEP TUTOR CHAT (Global Floating) */}
          {(activeTab === 'education' || activeTab === 'clinical') && (
              <TutorChat 
                  context={prediction}
                  minimized={activeTab !== 'education'} 
                  setMinimized={(val) => {
                       if (val === false) setActiveTab('education');
                  }}
                  onAction={(action) => {
                      console.log("Tutor Action:", action);
                      if (action.type === 'highlight_lead') {
                          const leadName = action.value;
                          const leadIndex = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'].indexOf(leadName);
                          if (leadIndex >= 0) {
                              const newLeads = Array(12).fill(false);
                              newLeads[leadIndex] = true;
                              setActiveLeads(newLeads);
                          }
                      }
                      if (action.type === 'show_all') setActiveLeads(Array(12).fill(true));
                  }}
              />
          )}
      </main>

    </div>
  );
}
