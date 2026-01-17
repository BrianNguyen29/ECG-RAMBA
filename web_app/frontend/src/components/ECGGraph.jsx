import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceArea } from 'recharts';
import { ZoomIn, ZoomOut, RotateCcw, Activity, Maximize2, Minimize2, Ruler, X } from 'lucide-react';
import clsx from 'clsx';

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
// Neon Palette for Dark Mode
const LEAD_COLORS_DARK = [
  '#F87171', '#60A5FA', '#34D399', '#FBBF24', '#A78BFA', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#818CF8', '#2DD4BF', '#C084FC'
];
// Medical Palette for Light Mode
const LEAD_COLORS_LIGHT = [
  '#dc2626', '#2563eb', '#059669', '#d97706', '#7c3aed', '#db2777',
  '#0891b2', '#65a30d', '#ea580c', '#4f46e5', '#0d9488', '#9333ea'
];

const ECGGraph = React.memo(({ data, saliencyMap, activeLeads, annotations, darkMode = false, sampleRate = 500 }) => {
  const [selectedLead, setSelectedLead] = useState(1);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isStacked, setIsStacked] = useState(true); // Default to Medical Stacked View
  const [paperSpeed, setPaperSpeed] = useState(25); // mm/s: 25 or 50
  
  // Caliper State
  const [showCallipers, setShowCallipers] = useState(false);
  const [caliperStart, setCaliperStart] = useState(null);
  const [caliperEnd, setCaliperEnd] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Reset calipers when view changes
  useEffect(() => {
    setCaliperStart(null);
    setCaliperEnd(null);
  }, [selectedLead, isStacked, zoomLevel]);

  // Handle Caliper Mouse Events
  const handleMouseDown = (e) => {
    if (!showCallipers) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Map pixels to logic units
    // X logic: Time
    // Y logic: Voltage
    // Note: Recharts logic mapping is tricky without API access.
    // Simpler approach: Use proportional positioning (0-100%) and map to visible range.
    
    const relX = x / rect.width;
    const relY = y / rect.height; 

    setCaliperStart({ x, y, relX, relY });
    setCaliperEnd({ x, y, relX, relY });
    setIsDragging(true);
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !showCallipers) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const relX = x / rect.width;
    const relY = y / rect.height; 

    setCaliperEnd({ x, y, relX, relY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };
   
  // Select color palette based on dark mode
  
  // Select color palette based on dark mode
  const LEAD_COLORS = darkMode ? LEAD_COLORS_DARK : LEAD_COLORS_LIGHT;

  // Derive lead-specific data from props
  const numLeads = data?.length || 12;
  const leadData = data?.[selectedLead] || [];
  const leadSaliency = saliencyMap?.[selectedLead] || [];



  // Single Lead View: Visible Samples
  const visibleSamples = useMemo(() => {
    if (!leadData || leadData.length === 0) return [];
    
    // Calculate samples to show
    const samplesPerView = Math.floor(leadData.length / zoomLevel);
    const startIdx = 0;
    const endIdx = Math.min(startIdx + samplesPerView, leadData.length);
    
    // Downsample for performance
    const maxPoints = 1500;
    const step = Math.max(1, Math.floor((endIdx - startIdx) / maxPoints));
    
    const samples = [];
    for (let i = startIdx; i < endIdx; i += step) {
      samples.push({
        time: ((i / sampleRate) * 1000).toFixed(0), // Use dynamic sampleRate
        index: i,
        value: leadData[i]
      });
    }
    return samples;
  }, [leadData, zoomLevel, sampleRate]);

  // Signal statistics and R-peak detection for Focus View
  const signalStats = useMemo(() => {
    if (!leadData || leadData.length === 0) return null;
    
    const minVal = Math.min(...leadData);
    const maxVal = Math.max(...leadData);
    const amplitude = maxVal - minVal;
    
    // Simple R-peak detection (find local maxima above threshold)
    const threshold = minVal + amplitude * 0.6;
    const rPeaks = [];
    const minDistance = Math.floor(sampleRate * 0.3); // Min 300ms between beats
    
    for (let i = 1; i < leadData.length - 1; i++) {
      if (leadData[i] > threshold && 
          leadData[i] > leadData[i-1] && 
          leadData[i] > leadData[i+1]) {
        // Check minimum distance from last peak
        if (rPeaks.length === 0 || (i - rPeaks[rPeaks.length - 1]) >= minDistance) {
          rPeaks.push(i);
        }
      }
    }
    
    // Calculate R-R intervals and heart rate
    const rrIntervals = [];
    for (let i = 1; i < rPeaks.length; i++) {
      rrIntervals.push((rPeaks[i] - rPeaks[i-1]) / sampleRate * 1000); // ms
    }
    
    const avgRR = rrIntervals.length > 0 
      ? rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length 
      : 0;
    const heartRate = avgRR > 0 ? Math.round(60000 / avgRR) : 0;
    
    return {
      min: minVal.toFixed(2),
      max: maxVal.toFixed(2),
      amplitude: amplitude.toFixed(2),
      rPeakCount: rPeaks.length,
      avgRR: avgRR.toFixed(0),
      heartRate,
      duration: (leadData.length / sampleRate).toFixed(1)
    };
  }, [leadData, sampleRate]);


  // Saliency Gradient
  const saliencyGradient = useMemo(() => {
    if (!leadSaliency || leadSaliency.length === 0) return null;
    const segments = 100;
    const step = Math.floor(leadSaliency.length / segments);
    let stops = [];
    
    for (let i = 0; i < segments; i++) {
        const start = i * step;
        const end = Math.min(start + step, leadSaliency.length);
        const segment = leadSaliency.slice(start, end);
        const avg = segment.reduce((a, b) => a + b, 0) / segment.length;
        const pct = (i / segments) * 100;
        // Use Red with varying opacity
        stops.push(`rgba(239, 68, 68, ${Math.min(avg * 8, 0.8)}) ${pct}%`);
    }
    return `linear-gradient(90deg, ${stops.join(', ')})`;
  }, [leadSaliency]);

  if (!data || data.length === 0) {
    return (
      <div className={clsx(
        "flex flex-col items-center justify-center h-full relative overflow-hidden",
        darkMode ? "bg-slate-900 text-slate-500" : "bg-slate-50 text-slate-400"
      )}>
        <div className={clsx(
          "absolute inset-0 opacity-100",
          darkMode ? "ecg-grid-dark" : "ecg-grid-light"
        )} />
        <Activity className={clsx("w-12 h-12 mb-4 opacity-50", darkMode ? "text-cyan-500" : "text-slate-400")} />
        <p className="font-mono text-sm">NO SIGNAL</p>
        {darkMode && <p className="font-mono text-xs mt-2 text-slate-600">Dark Mode Active</p>}
      </div>
    );
  }

  return (
    <div className="w-full h-full relative flex flex-col">
      {/* Controls Overlay (Floating) */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
          {/* View Toggle */}
          {numLeads > 1 && (
             <button
                onClick={() => setIsStacked(!isStacked)}
                className="flex items-center gap-2 px-3 py-1.5 bg-white/90 backdrop-blur border border-slate-200 rounded-lg text-xs font-bold text-slate-600 hover:bg-slate-100 hover:text-blue-600 transition-all shadow-sm"
             >
                {isStacked ? <Maximize2 className="w-3 h-3" /> : <Minimize2 className="w-3 h-3" />}
                {isStacked ? "Focus View" : "12-Lead View"}
             </button>
          )}

          {/* Zoom Controls (Only for Single View) */}
          {!isStacked && (
            <div className="flex items-center gap-1 bg-white/90 backdrop-blur border border-slate-200 rounded-lg p-1 shadow-sm">
                <button 
                  onClick={() => setShowCallipers(!showCallipers)} 
                  className={clsx(
                    "p-1 rounded transition-colors",
                    showCallipers ? "bg-rose-100 text-rose-600" : "text-slate-500 hover:text-blue-600"
                  )}
                  title="Digital Calipers"
                >
                  <Ruler className="w-3 h-3" />
                </button>
                <div className="w-px h-3 bg-slate-200 mx-0.5" />
                <button onClick={() => setZoomLevel(Math.min(zoomLevel * 2, 8))} className="p-1 text-slate-500 hover:text-blue-600" aria-label="Zoom in"><ZoomIn className="w-3 h-3" /></button>
                <button onClick={() => setZoomLevel(1)} className="p-1 text-slate-500 hover:text-blue-600" aria-label="Reset zoom"><RotateCcw className="w-3 h-3" /></button>
                <button onClick={() => setZoomLevel(Math.max(zoomLevel / 2, 0.5))} className="p-1 text-slate-500 hover:text-blue-600" aria-label="Zoom out"><ZoomOut className="w-3 h-3" /></button>
            </div>
          )}
      </div>

      {/* CHART AREA */}
      <div className={clsx(
        "flex-1 overflow-hidden relative",
        darkMode ? "bg-slate-900" : "bg-white"
      )}>
         {/* Background Grid - Medical Paper Style */}
         <div className={clsx(
           "absolute inset-0 opacity-100 pointer-events-none",
           darkMode ? "ecg-grid-dark" : "ecg-grid-light"
         )} />
         
         {/* AI Saliency Overlay (Red gradient) */}
         {saliencyMap && (
           <div 
             className="absolute inset-0 pointer-events-none opacity-60 z-5" 
             style={{ background: saliencyGradient }}
           />
         )}

         {isStacked ? (
             /* STACKED 12-LEAD VIEW */
             <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2 p-4 h-full overflow-y-auto scrollbar-thin">
                 {LEAD_NAMES.slice(0, numLeads).map((leadName, idx) => {
                     const leadSig = data[idx] || [];
                     const isActive = activeLeads ? activeLeads[idx] : true;
                     const displaySig = isActive ? leadSig : new Array(leadSig.length).fill(0);

                     // Min-Max Downsampling with higher resolution
                     const chartData = [];
                     const targetPoints = 500; // Increased for better peak visibility 
                     const step = Math.ceil(displaySig.length / targetPoints);
                     for (let i = 0; i < displaySig.length; i += step) {
                        let min = displaySig[i], max = displaySig[i];
                        let minIdx = i, maxIdx = i;
                        const end = Math.min(i + step, displaySig.length);
                        for (let j = i + 1; j < end; j++) {
                            const val = displaySig[j];
                            if (val < min) { min = val; minIdx = j; }
                            if (val > max) { max = val; maxIdx = j; }
                        }
                        if (minIdx < maxIdx) {
                            chartData.push({ v: min, i: minIdx });
                            chartData.push({ v: max, i: maxIdx });
                        } else {
                            chartData.push({ v: max, i: maxIdx });
                            chartData.push({ v: min, i: minIdx });
                        }
                     }

                     return (
                         <div key={leadName} className={clsx(
                           "h-24 relative border-b last:border-0 transition-colors",
                           darkMode 
                             ? "border-slate-700 hover:bg-slate-800/50" 
                             : "border-slate-100 hover:bg-slate-50"
                         )}>
                             {/* Lead Label */}
                             <span className={clsx(
                               "absolute top-1 left-1 text-[11px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none border",
                               darkMode 
                                 ? "text-cyan-400 bg-slate-800/90 border-slate-700" 
                                 : "text-slate-700 bg-white/90 border-slate-200 shadow-sm"
                             )}>
                                {leadName}
                             </span>
                             {/* mV Scale Reference (1mV marker) */}
                             <div className={clsx(
                               "absolute right-2 top-1/2 -translate-y-1/2 flex flex-col items-center z-10 pointer-events-none",
                               darkMode ? "text-slate-500" : "text-slate-400"
                             )}>
                               <div className={clsx(
                                 "w-0.5 h-6 mb-0.5",
                                 darkMode ? "bg-cyan-500/50" : "bg-red-400/50"
                               )} />
                               <span className="text-[7px] font-mono">1mV</span>
                             </div>
                             <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <Line 
                                        type="monotone" 
                                        dataKey="v" 
                                        stroke={isActive ? (LEAD_COLORS[idx] || '#fff') : '#334155'} 
                                        strokeWidth={1.5} 
                                        dot={false}
                                        isAnimationActive={false}
                                    />
                                    <YAxis domain={['auto', 'auto']} hide />
                                </LineChart>
                             </ResponsiveContainer>
                         </div>
                     );
                 })}
             </div>
         ) : (
             /* SINGLE LEAD FOCUS VIEW */
             <div className="h-full w-full relative">
                 <div className="absolute top-4 left-4 z-10">
                    <select 
                        value={selectedLead} 
                        onChange={(e) => setSelectedLead(parseInt(e.target.value))}
                        className={clsx(
                            "text-xs border rounded p-1.5 outline-none focus:ring-1 focus:ring-cyan-500 font-medium",
                            darkMode 
                                ? "bg-slate-800 text-slate-200 border-slate-700" 
                                : "bg-white text-slate-700 border-slate-300 shadow-sm"
                        )}
                    >
                        {LEAD_NAMES.slice(0, numLeads).map((n, i) => <option key={i} value={i}>Lead {n}</option>)}
                    </select>
                 </div>

                 {/* Measurement Info Panel - positioned below controls */}
                 {signalStats && (
                   <div className={clsx(
                     "absolute top-14 right-4 z-10 px-3 py-2 rounded-lg shadow-lg border text-[10px] font-mono",
                     darkMode 
                       ? "bg-slate-800/95 border-slate-700 text-slate-300" 
                       : "bg-white/95 border-slate-200 text-slate-600"
                   )}>
                     <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                       <div className="flex items-center gap-1">
                         <span className="text-rose-500 font-bold">♥</span>
                         <span className="font-bold text-base text-rose-500">{signalStats.heartRate}</span>
                         <span className="text-[8px]">BPM</span>
                       </div>
                       <div>
                         <span className={darkMode ? "text-slate-500" : "text-slate-400"}>R-R:</span> 
                         <span className="font-medium ml-1">{signalStats.avgRR}ms</span>
                       </div>
                       <div>
                         <span className={darkMode ? "text-slate-500" : "text-slate-400"}>Peaks:</span> 
                         <span className="font-medium ml-1">{signalStats.rPeakCount}</span>
                       </div>
                       <div>
                         <span className={darkMode ? "text-slate-500" : "text-slate-400"}>Dur:</span> 
                         <span className="font-medium ml-1">{signalStats.duration}s</span>
                       </div>
                     </div>
                   </div>
                 )}

                 <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={visibleSamples}>
                        <XAxis dataKey="time" hide />
                        <YAxis domain={['auto', 'auto']} hide />
                        {/* ANNOTATION LAYERS (P/QRS/T) */}
                        {annotations && annotations.lead_used === LEAD_NAMES[selectedLead] && annotations.wave_boundaries?.map((wave, idx) => (
                             <React.Fragment key={idx}>
                                {wave.p_onset > 0 && (
                                    <ReferenceArea 
                                        x1={wave.p_onset} x2={wave.p_offset} 
                                        fill="#60A5FA" fillOpacity={0.1} 
                                        ifOverflow="extendDomain"
                                        label={{ value: 'P', fill: '#60A5FA', fontSize: 10, position: 'top' }}
                                    />
                                )}
                                {wave.qrs_onset > 0 && (
                                    <ReferenceArea 
                                        x1={wave.qrs_onset} x2={wave.qrs_offset} 
                                        fill="#F87171" fillOpacity={0.1}
                                        label={{ value: 'QRS', fill: '#F87171', fontSize: 10, position: 'top' }}
                                    />
                                )}
                                {wave.st_onset > 0 && (
                                    <ReferenceArea 
                                        x1={wave.st_onset} x2={wave.st_offset} 
                                        fill="#FBBF24" fillOpacity={0.1}
                                    />
                                )}
                             </React.Fragment>
                        ))}

                        <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={LEAD_COLORS[selectedLead]} 
                            strokeWidth={2} 
                            dot={false} 
                            isAnimationActive={false}
                        />
                        {/* Custom Tooltip with Mamba Attention Score */}
                        <Tooltip 
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const dataPoint = payload[0].payload;
                                    const saliencyValue = leadSaliency && dataPoint.index < leadSaliency.length 
                                        ? leadSaliency[dataPoint.index] 
                                        : null;
                                    const attentionLevel = saliencyValue 
                                        ? saliencyValue > 0.7 ? 'High' : saliencyValue > 0.4 ? 'Medium' : 'Low'
                                        : 'N/A';
                                    const attentionColor = saliencyValue
                                        ? saliencyValue > 0.7 ? 'text-rose-500' : saliencyValue > 0.4 ? 'text-amber-500' : 'text-slate-400'
                                        : 'text-slate-400';
                                    
                                    return (
                                        <div className={clsx(
                                            "px-3 py-2 rounded-lg shadow-lg border text-xs",
                                            darkMode 
                                                ? "bg-slate-800 border-slate-700 text-slate-200" 
                                                : "bg-white border-slate-200 text-slate-800"
                                        )}>
                                            <div className="font-mono">
                                                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Sample:</span> {dataPoint.index}
                                            </div>
                                            <div className="font-mono">
                                                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Value:</span> {dataPoint.value?.toFixed(3)}
                                            </div>
                                            {saliencyValue !== null && (
                                                <div className="mt-1 pt-1 border-t border-slate-700">
                                                    <div className={clsx("font-bold", attentionColor)}>
                                                        Mamba Attention: {attentionLevel}
                                                    </div>
                                                    <div className="text-[10px] text-slate-500">
                                                        Score: {(saliencyValue * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <YAxis domain={['auto', 'auto']} hide />
                         {/* XAxis required for ReferenceArea to map indices correctly? 
                             No, index is implicit in data array if not specified.
                             Wait, visibleSamples has 'index' property. 
                             We need XAxis with dataKey="index" to map wave boundaries (indices) to chart X.
                         */}
                        <XAxis dataKey="index" hide />

                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', color: '#f1f5f9', fontSize: '12px' }}
                            itemStyle={{ color: '#bae6fd' }}
                            formatter={(val) => val.toFixed(3)}
                            labelFormatter={() => ''}
                        />
                    </LineChart>
                 </ResponsiveContainer>

                 {/* Saliency Overlay */}
                 {saliencyGradient && (
                     <div className="absolute bottom-4 left-4 right-4 h-2 rounded-full overflow-hidden bg-slate-800/50">
                         <div className="w-full h-full opacity-80" style={{ background: saliencyGradient }} />
                     </div>
                 )}

                 {/* Digital Calipers Overlay Layer */}
                 {showCallipers && (
                   <div 
                     className="absolute inset-0 z-30 cursor-crosshair select-none"
                     onMouseDown={handleMouseDown}
                     onMouseMove={handleMouseMove}
                     onMouseUp={handleMouseUp}
                     onMouseLeave={handleMouseUp}
                   >
                      {caliperStart && caliperEnd && (
                        <>
                          {/* Selection Box */}
                          <div 
                            className="absolute border border-rose-400 bg-rose-400/10"
                            style={{
                              left: Math.min(caliperStart.x, caliperEnd.x),
                              top: Math.min(caliperStart.y, caliperEnd.y),
                              width: Math.abs(caliperEnd.x - caliperStart.x),
                              height: Math.abs(caliperEnd.y - caliperStart.y)
                            }}
                          />
                          
                          {/* Measurements Badge */}
                          <div 
                            className="absolute bg-slate-900/90 text-white text-[10px] font-mono p-1 rounded shadow-xl border border-rose-500/50 flex flex-col items-start gap-0.5 pointer-events-none whitespace-nowrap z-40"
                            style={{
                              left: Math.max(caliperStart.x, caliperEnd.x) + 10,
                              top: Math.min(caliperStart.y, caliperEnd.y),
                            }}
                          >
                            {(() => {
                                // Calculate Values
                                const totalMs = (visibleSamples.length / sampleRate) * 1000; // Total ms in view
                                const deltaX = Math.abs(caliperEnd.relX - caliperStart.relX);
                                const ms = Math.round(deltaX * totalMs);
                                
                                // Voltage is tricky without fixed scale. 
                                // Assuming typical view range fits roughly 2-3mV
                                // Better approach: Use grid size. width=100%
                                // We know total duration. 
                                // For voltage: we rely on standard 10mm/mV scale.
                                // 10mm ~ 50px (based on grid CSS 5mm=25px).
                                // So 1mV = 50px vertical.
                                const deltaYPx = Math.abs(caliperEnd.y - caliperStart.y);
                                const mV = (deltaYPx / 50).toFixed(2);

                                return (
                                  <>
                                    <span className="text-rose-300 font-bold">Δt: {ms} ms</span>
                                    <span className="text-cyan-300">ΔV: {mV} mV</span>
                                  </>
                                );
                            })()}
                          </div>
                        </>
                      )}
                      
                      {/* Helper hint */}
                      {!caliperStart && (
                         <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/50 text-white px-2 py-1 rounded text-xs pointer-events-none">
                            Click and drag to measure
                         </div>
                      )}
                      
                      {/* Close Button */}
                      <button 
                        onClick={(e) => {
                            e.stopPropagation(); // Prevent drawing
                            setShowCallipers(false);
                            setCaliperStart(null);
                            setCaliperEnd(null);
                        }}
                        className="absolute top-2 right-2 bg-white/90 text-slate-500 hover:text-rose-500 hover:bg-rose-50 p-1 rounded-full shadow-sm z-50 transition-colors pointer-events-auto"
                        title="Close Calipers"
                      >
                        <X className="w-3 h-3" />
                      </button>
                   </div>
                 )}
             </div>
         )}
      </div>

       <div className={clsx(
           "h-7 border-t flex justify-between items-center px-4 text-[9px] uppercase font-mono",
           darkMode ? "bg-slate-800 border-slate-700 text-slate-400" : "bg-slate-50 border-slate-200 text-slate-500"
       )}>
           <div className="flex items-center gap-4">
               <span className="flex items-center gap-1">
                   <span className="font-bold text-blue-500">{numLeads}</span> Leads
               </span>
               <span>{sampleRate} Hz</span>
           </div>
           <div className="flex items-center gap-4">
               <button 
                   onClick={() => setPaperSpeed(paperSpeed === 25 ? 50 : 25)}
                   className={clsx(
                       "px-1.5 py-0.5 rounded text-[8px] font-bold transition-colors",
                       darkMode 
                           ? "bg-cyan-900/50 hover:bg-cyan-800/50 text-cyan-300" 
                           : "bg-red-100 hover:bg-red-200 text-red-700"
                   )}
                   title="Click to toggle paper speed"
               >
                   Speed: {paperSpeed}mm/s
               </button>
               <span>Gain: 10mm/mV</span>
           </div>
           <div className="flex items-center gap-1">
               {LEAD_NAMES.slice(0, Math.min(numLeads, 6)).map((name, idx) => (
                   <span 
                       key={name} 
                       className={clsx(
                           "px-1 rounded text-[8px] font-bold",
                           activeLeads?.[idx] !== false 
                               ? "text-white" 
                               : "text-slate-400 line-through"
                       )}
                       style={{ backgroundColor: activeLeads?.[idx] !== false ? LEAD_COLORS[idx] : 'transparent' }}
                   >
                       {name}
                   </span>
               ))}
               {numLeads > 6 && <span className="text-slate-400">+{numLeads - 6}</span>}
           </div>
       </div>
    </div>
  );
});

export default ECGGraph;
