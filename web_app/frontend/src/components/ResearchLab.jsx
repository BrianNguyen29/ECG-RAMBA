import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Activity, Zap, Sliders, Play, RotateCcw, Brain, Waves } from 'lucide-react';
import ECGGraph from './ECGGraph';
import FrequencyGraph from './FrequencyGraph';
import { api } from '../services/api';
import clsx from 'clsx';

const ResearchLab = ({ originalSignal, patientId, sampleRate = 500 }) => {
    // ------------------------------------------------------------------------
    // STATE
    // ------------------------------------------------------------------------
    // DEBUG: Inject dummy signal if originalSignal is missing
    const initialSignal = useMemo(() => {
        if (originalSignal && originalSignal.length > 0) {
            console.log("ResearchLab received signal:", {
                type: Array.isArray(originalSignal) ? "Array" : typeof originalSignal,
                length: originalSignal.length,
                elem0Type: Array.isArray(originalSignal[0]) ? "Array" : typeof originalSignal[0],
                elem0Len: originalSignal[0]?.length,
                sampleValues: originalSignal[0]?.slice(0, 5)
            });
            return originalSignal;
        }
        console.log("ResearchLab: Using Dummy Signal");
        // Generate 12 leads of sine waves
        return Array(12).fill(0).map((_, leadIdx) => 
            Array(5000).fill(0).map((__, t) => Math.sin(t/10) + (leadIdx * 0.5))
        );
    }, [originalSignal]);

    const [processedSignal, setProcessedSignal] = useState(initialSignal);
    const [freqData, setFreqData] = useState(null);
    const [loading, setLoading] = useState(false);
    
    // Filter Params
    const [filterType, setFilterType] = useState('none'); // none, bandpass, notch
    const [lowCut, setLowCut] = useState(0.5);
    const [highCut, setHighCut] = useState(50);
    const [notchFreq, setNotchFreq] = useState(50);

    // ------------------------------------------------------------------------
    // EFFECTS
    // ------------------------------------------------------------------------
    // 1. Initial Load: Compute PSD of raw signal
    useEffect(() => {
        if (originalSignal && originalSignal.length > 0) {
            handleAnalyze(originalSignal);
        }
    }, [originalSignal]);

    // ------------------------------------------------------------------------
    // HANDLERS
    // ------------------------------------------------------------------------
    const handleAnalyze = async (signalData) => {
        try {
            // Take lead 2 (index 1) or Lead 0 if unavailable
            const leadData = signalData[1] || signalData[0];
            const res = await api.analyzeSignal(leadData, sampleRate); // Dynamic Sample Rate from Props
            setFreqData(res);
        } catch (e) {
            console.error("Analysis Failed", e);
        }
    };

    const handleApplyFilter = async () => {
        if (!originalSignal) return;
        setLoading(true);
        try {
            // Backend expects 2D array: [channels, samples]
            const type = filterType;
            let low = undefined;
            let high = undefined;

            if (type === 'bandpass') {
                low = lowCut;
                high = highCut;
            } else if (type === 'notch') {
                low = notchFreq; // API uses 'low' param for notch freq
            }

            // Using Dynamic Sample Rate
            const res = await api.processSignal(originalSignal, type, low, high, sampleRate);
            
            if (res.filtered_signal) {
                setProcessedSignal(res.filtered_signal);
                // Re-calculate PSD for the new signal
                await handleAnalyze(res.filtered_signal);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setProcessedSignal(originalSignal);
        setFilterType('none');
        handleAnalyze(originalSignal);
    };

    // ------------------------------------------------------------------------
    // UI RENDER
    // ------------------------------------------------------------------------
    const getFilterBadge = () => {
        if (filterType === 'none') return <Badge variant="outline">Raw Signal</Badge>;
        if (filterType === 'bandpass') return <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-200">Bandpass {lowCut}-{highCut}Hz</Badge>;
        if (filterType === 'notch') return <Badge className="bg-purple-100 text-purple-700 hover:bg-purple-200">Notch {notchFreq}Hz</Badge>;
    };

    return (
        <div className="h-full grid grid-cols-12 gap-4 p-2">
            
            {/* LEFT SIDEBAR: DSP CONTROLS */}
            <div className="col-span-12 lg:col-span-3 space-y-4">
                <Card className="border-slate-200 shadow-sm h-full flex flex-col">
                    <CardHeader className="pb-2 bg-slate-50 rounded-t-xl border-b border-slate-100">
                        <CardTitle className="text-sm font-bold flex items-center gap-2 text-slate-700">
                            <Sliders className="w-4 h-4 text-indigo-500" />
                            DSP Toolbox
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6 pt-4 flex-1">
                        
                        {/* Filter Selection */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 uppercase">Filter Type</label>
                            <div className="grid grid-cols-3 gap-2">
                                <Button 
                                    variant={filterType === 'none' ? "default" : "outline"} 
                                    onClick={() => setFilterType('none')}
                                    size="sm" className={filterType === 'none' ? "bg-slate-700" : ""}
                                >Raw</Button>
                                <Button 
                                    variant={filterType === 'bandpass' ? "default" : "outline"} 
                                    onClick={() => setFilterType('bandpass')}
                                    size="sm" className={filterType === 'bandpass' ? "bg-blue-600" : ""}
                                >Bandpass</Button>
                                <Button 
                                    variant={filterType === 'notch' ? "default" : "outline"} 
                                    onClick={() => setFilterType('notch')}
                                    size="sm" className={filterType === 'notch' ? "bg-purple-600" : ""}
                                >Notch</Button>
                            </div>
                        </div>

                        {/* Sliders based on Type */}
                        {filterType === 'bandpass' && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
                                <div className="space-y-1">
                                    <div className="flex justify-between text-xs text-slate-500">
                                        <span>Low Cut (Hz)</span>
                                        <span className="font-mono text-blue-600">{lowCut} Hz</span>
                                    </div>
                                    <input 
                                        type="range" min="0.1" max="10" step="0.1"
                                        value={lowCut} onChange={(e) => setLowCut(parseFloat(e.target.value))}
                                        className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <div className="flex justify-between text-xs text-slate-500">
                                        <span>High Cut (Hz)</span>
                                        <span className="font-mono text-blue-600">{highCut} Hz</span>
                                    </div>
                                    <input 
                                        type="range" min="15" max="100" step="1"
                                        value={highCut} onChange={(e) => setHighCut(parseFloat(e.target.value))}
                                        className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                    />
                                </div>
                            </div>
                        )}

                        {filterType === 'notch' && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-top-2">
                                 <div className="space-y-1">
                                    <div className="flex justify-between text-xs text-slate-500">
                                        <span>Notch Freq (Mains)</span>
                                        <span className="font-mono text-purple-600">{notchFreq} Hz</span>
                                    </div>
                                    <div className="flex gap-2">
                                        <Button size="sm" variant={notchFreq === 50 ? "default" : "outline"} onClick={() => setNotchFreq(50)} className={notchFreq === 50 ? "bg-purple-600 w-1/2" : "w-1/2"}>50 Hz (EU)</Button>
                                        <Button size="sm" variant={notchFreq === 60 ? "default" : "outline"} onClick={() => setNotchFreq(60)} className={notchFreq === 60 ? "bg-purple-600 w-1/2" : "w-1/2"}>60 Hz (US)</Button>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="pt-4 flex gap-2">
                            <Button onClick={handleApplyFilter} disabled={filterType === 'none' || loading} className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white">
                                {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                                Apply DSP
                            </Button>
                            <Button variant="outline" size="icon" onClick={handleReset} title="Reset Original" className="border-slate-200">
                                <RotateCcw className="w-4 h-4 text-slate-500" />
                            </Button>
                        </div>

                    </CardContent>

                    {/* Quick Stats */}
                    {freqData && (
                        <div className="p-4 bg-slate-50 border-t border-slate-100 rounded-b-xl text-xs space-y-2">
                            <h4 className="font-bold text-slate-700 flex items-center gap-1 uppercase tracking-wider">
                                <Brain className="w-3 h-3" /> Spectral Power
                            </h4>
                            <div className="grid grid-cols-2 gap-2">
                                <div className="bg-white p-2 rounded border border-slate-200">
                                    <span className="block text-slate-400 text-[10px]">Alpha (8-13Hz)</span>
                                    <span className="font-mono font-bold text-emerald-600">{(freqData.features.alpha * 100).toFixed(1)}%</span>
                                </div>
                                <div className="bg-white p-2 rounded border border-slate-200">
                                    <span className="block text-slate-400 text-[10px]">Delta (0.5-4Hz)</span>
                                    <span className="font-mono font-bold text-blue-600">{(freqData.features.delta * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    )}
                </Card>
            </div>

            {/* RIGHT MAIN: VISUALIZATION */}
            <div className="col-span-12 lg:col-span-9 flex flex-col gap-4 h-full min-h-[500px]">
                
                {/* 1. Time Domain (ECG/EEG Graph) */}
                <div className="flex-[2] bg-white border border-slate-200 rounded-2xl overflow-hidden relative shadow-sm flex flex-col">
                    <div className="absolute top-2 left-4 z-10 flex items-center gap-2">
                        <Badge variant="outline" className="border-indigo-200 text-indigo-700 bg-indigo-50">
                            Time Domain
                        </Badge>
                        <Badge variant="outline" className="text-slate-500">
                            Fs: {sampleRate} Hz
                        </Badge>
                        {getFilterBadge()}
                    </div>
                    {/* Updated ECGGraph with dynamic sampleRate */}
                    <ECGGraph data={processedSignal} activeLeads={null} sampleRate={sampleRate} />
                </div>

                {/* 2. Frequency Domain (PSD) */}
                <div className="flex-1 bg-slate-900 rounded-2xl overflow-hidden relative shadow-sm flex flex-col p-4">
                    <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                            <Waves className="w-4 h-4 text-emerald-400" />
                            <span className="text-sm font-bold text-slate-200">Frequency Domain (PSD)</span>
                        </div>
                        <Badge className="bg-slate-800 text-slate-400 hover:bg-slate-800">0 Hz - 60 Hz</Badge>
                    </div>
                    {freqData ? (
                        <FrequencyGraph data={freqData} color="#34d399" />
                    ) : (
                        <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                            Run DSP analysis to view spectrum
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResearchLab;
