import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ZoomIn, ZoomOut, RotateCcw, Activity, ChevronDown } from 'lucide-react';
import clsx from 'clsx';

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
const LEAD_COLORS = [
  '#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899',
  '#06B6D4', '#84CC16', '#F97316', '#6366F1', '#14B8A6', '#A855F7'
];

const ECGGraph = ({ data, saliencyMap, activeLeads }) => {
  const [selectedLead, setSelectedLead] = useState(1); // Default to Lead II
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showLeadSelector, setShowLeadSelector] = useState(false);
  const [isStacked, setIsStacked] = useState(false); // [Deep RAMBA] Stacked view toggle

  // Determine if data is 12-lead (2D array) or single lead (1D array)
  const { leadData, numLeads, numSamples, leadSaliency } = useMemo(() => {
    if (!data || data.length === 0) {
      return { leadData: [], numLeads: 0, numSamples: 0, leadSaliency: null };
    }
    
    // Check if 2D array (12-lead format)
    if (Array.isArray(data[0])) {
      const rawData = data[selectedLead] || data[0];
      const isLeadActive = activeLeads ? activeLeads[selectedLead] : true;
      
      return {
        leadData: isLeadActive ? rawData : new Array(rawData.length).fill(0),
        numLeads: data.length,
        numSamples: data[0].length,
        leadSaliency: saliencyMap ? (saliencyMap[selectedLead] || saliencyMap[0]) : null
      };
    }
    
    // 1D array (single lead)
    return {
      leadData: data,
      numLeads: 1,
      numSamples: data.length,
      leadSaliency: saliencyMap // Assume 1D saliency if data is 1D
    };
  }, [data, selectedLead, saliencyMap, activeLeads]);

  // Calculate visible samples based on zoom
  const visibleSamples = useMemo(() => {
    if (!leadData || leadData.length === 0) return [];
    
    const samplesPerView = Math.floor(leadData.length / zoomLevel);
    const startIdx = 0;
    const endIdx = Math.min(startIdx + samplesPerView, leadData.length);
    
    // Downsample for performance if too many points
    const maxPoints = 1000;
    const step = Math.max(1, Math.floor((endIdx - startIdx) / maxPoints));
    
    const samples = [];
    for (let i = startIdx; i < endIdx; i += step) {
      samples.push({
        time: ((i / 500) * 1000).toFixed(0), // ms
        index: i,
        value: leadData[i]
      });
    }
    return samples;
  }, [leadData, zoomLevel]);

  // Generate CSS Gradient for Saliency Heatmap
  const saliencyGradient = useMemo(() => {
    if (!leadSaliency || leadSaliency.length === 0) return null;

    // Downsample to ~100 segments for CSS performance
    const segments = 100;
    const step = Math.floor(leadSaliency.length / segments);
    let gradientStops = [];
    
    for (let i = 0; i < segments; i++) {
        const start = i * step;
        const end = Math.min(start + step, leadSaliency.length);
        const segment = leadSaliency.slice(start, end);
        const avg = segment.reduce((a, b) => a + b, 0) / segment.length;
        
        // Map 0-1 to Heatmap Colors (White -> Yellow -> Red)
        // Simple: rgba(255, 0, 0, opacity)
        const pct = (i / segments) * 100;
        gradientStops.push(`rgba(255, 0, 0, ${Math.min(avg * 5, 1)}) ${pct}%`); // Multiply by 5 to boost visibility
    }
    
    return `linear-gradient(90deg, ${gradientStops.join(', ')})`;
  }, [leadSaliency]);

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl relative overflow-hidden">
        {/* ECG Grid Background */}
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(#FF9999 1px, transparent 1px),
              linear-gradient(90deg, #FF9999 1px, transparent 1px),
              linear-gradient(#FFCCCC 0.5px, transparent 0.5px),
              linear-gradient(90deg, #FFCCCC 0.5px, transparent 0.5px)
            `,
            backgroundSize: '40px 40px, 40px 40px, 8px 8px, 8px 8px'
          }}
        />
        <div className="z-10 text-center">
          <div className="w-16 h-16 bg-white/80 backdrop-blur rounded-2xl shadow-lg flex items-center justify-center mx-auto mb-4">
            <Activity className="w-8 h-8 text-gray-400" />
          </div>
          <p className="text-gray-600 font-medium">No Signal Loaded</p>
          <p className="text-xs text-gray-400 mt-1">Upload a record to visualize</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full relative">
      {/* Controls Bar */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {/* Lead Selector */}
          {numLeads > 1 && (
            <div className="relative">
              <button
                onClick={() => setShowLeadSelector(!showLeadSelector)}
                className="flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium text-gray-700 transition-colors"
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: LEAD_COLORS[selectedLead] }} />
                Lead {LEAD_NAMES[selectedLead] || selectedLead + 1}
                <ChevronDown className={clsx("w-4 h-4 transition-transform", showLeadSelector && "rotate-180")} />
              </button>
              
              {showLeadSelector && (
                <div className="absolute top-full left-0 mt-1 bg-white rounded-xl shadow-xl border border-gray-200 p-2 z-20 grid grid-cols-4 gap-1 min-w-[200px]">
                  {LEAD_NAMES.slice(0, numLeads).map((name, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedLead(idx);
                        setShowLeadSelector(false);
                      }}
                      className={clsx(
                        "px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                        selectedLead === idx 
                          ? "bg-blue-100 text-blue-700" 
                          : "hover:bg-gray-100 text-gray-600"
                      )}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Live Indicator */}
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            {numSamples.toLocaleString()} samples
          </div>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
          <button 
            onClick={() => setZoomLevel(Math.min(zoomLevel * 2, 8))} 
            className="p-1.5 hover:bg-white hover:shadow-sm rounded text-gray-500 hover:text-gray-900 transition-all"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button 
            onClick={() => setZoomLevel(1)} 
            className="p-1.5 hover:bg-white hover:shadow-sm rounded text-gray-500 hover:text-gray-900 transition-all"
            title="Reset"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button 
            onClick={() => setZoomLevel(Math.max(zoomLevel / 2, 0.5))} 
            className="p-1.5 hover:bg-white hover:shadow-sm rounded text-gray-500 hover:text-gray-900 transition-all"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ECG Chart Area */}
      <div className={clsx(
            "relative bg-gradient-to-br from-white to-gray-50 rounded-xl border border-gray-200 overflow-hidden",
            isStacked ? "h-[600px] overflow-y-auto" : "h-64"
      )}>
        {/* Toggle View Mode Button (Absolute Top Right) */}
        {!isStacked && numLeads > 1 && (
             <button
                onClick={() => setIsStacked(true)}
                className="absolute top-2 right-2 z-20 px-3 py-1 bg-white/90 backdrop-blur border border-gray-200 shadow-sm rounded-lg text-xs font-semibold text-blue-600 hover:bg-blue-50 transition-all flex items-center gap-2"
             >
                <Activity className="w-3 h-3" />
                View 12-Lead Stack
             </button>
        )}
         {isStacked && (
             <button
                onClick={() => setIsStacked(false)}
                className="sticky top-2 right-2 float-right z-20 px-3 py-1 bg-white/90 backdrop-blur border border-gray-200 shadow-sm rounded-lg text-xs font-semibold text-gray-600 hover:bg-gray-100 transition-all flex items-center gap-2 m-2"
             >
                <ZoomIn className="w-3 h-3" />
                Focus Single Lead
             </button>
        )}

        {isStacked ? (
            /* STACKED VIEW (12-Lead Grid) */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
                {LEAD_NAMES.slice(0, numLeads).map((leadName, idx) => {
                     const leadSignal = data[idx] || [];
                     const isActive = activeLeads ? activeLeads[idx] : true;
                     const displaySignal = isActive ? leadSignal : new Array(leadSignal.length).fill(0);
                     
                     // Robust Min-Max Downsampling to preserve QRS peaks
                     const gridSamples = [];
                     const targetPoints = 600; // 300 pairs of min/max
                     const step = Math.ceil(displaySignal.length / targetPoints);
                     
                     for (let i = 0; i < displaySignal.length; i += step) {
                        let min = displaySignal[i];
                        let max = displaySignal[i];
                        let minIdx = i;
                        let maxIdx = i;
                        
                        const end = Math.min(i + step, displaySignal.length);
                        for (let j = i + 1; j < end; j++) {
                            const val = displaySignal[j];
                            if (val < min) { min = val; minIdx = j; }
                            if (val > max) { max = val; maxIdx = j; }
                        }
                        
                        // Push min and max in temporal order
                        if (minIdx < maxIdx) {
                            gridSamples.push({ value: min, index: minIdx });
                            gridSamples.push({ value: max, index: maxIdx });
                        } else {
                            gridSamples.push({ value: max, index: maxIdx });
                            gridSamples.push({ value: min, index: minIdx });
                        }
                     }

                    return (
                        <div key={leadName} className="h-32 bg-white border border-gray-100 rounded-lg relative overflow-hidden">
                             <span className="absolute top-1 left-2 text-[10px] font-bold text-gray-500 bg-white/80 px-1 rounded z-10">
                                {leadName}
                             </span>
                             <div 
                                className="absolute inset-0 opacity-20 pointer-events-none"
                                style={{
                                    backgroundImage: `linear-gradient(#eee 1px, transparent 1px), linear-gradient(90deg, #eee 1px, transparent 1px)`,
                                    backgroundSize: '20px 20px'
                                }}
                             />
                             <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={gridSamples}>
                                    <Line 
                                        type="monotone" 
                                        dataKey="value" 
                                        stroke={LEAD_COLORS[idx] || '#000'} 
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
            /* SINGLE LEAD VIEW (Detailed) */
            <>
                {/* ECG Paper Grid */}
                <div 
                  className="absolute inset-0 pointer-events-none opacity-30"
                  style={{
                    backgroundImage: `
                      linear-gradient(#FFAAAA 1px, transparent 1px),
                      linear-gradient(90deg, #FFAAAA 1px, transparent 1px),
                      linear-gradient(#FFDDDD 0.5px, transparent 0.5px),
                      linear-gradient(90deg, #FFDDDD 0.5px, transparent 0.5px)
                    `,
                    backgroundSize: '40px 40px, 40px 40px, 8px 8px, 8px 8px'
                  }}
                />
                
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={visibleSamples} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: '12px', 
                        border: 'none', 
                        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.15)',
                        backdropFilter: 'blur(8px)',
                        backgroundColor: 'rgba(255,255,255,0.95)'
                      }}
                      labelFormatter={(_, payload) => payload[0] ? `Sample ${payload[0].payload.index}` : ''}
                      formatter={(value) => [value.toFixed(4), 'mV']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke={LEAD_COLORS[selectedLead] || '#EF4444'}
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>

                {/* Saliency Heatmap Overlay (Bottom) */}
                {saliencyGradient && (
                    <div className="absolute bottom-0 left-0 right-0 h-4 mx-5 mb-2 rounded-full overflow-hidden border border-gray-100/50 shadow-sm z-10">
                        <div 
                            className="w-full h-full opacity-80"
                            style={{ background: saliencyGradient }}
                        />
                    </div>
                )}
            </>
        )}
      </div>

      {/* Footer Info */}
      <div className="flex justify-between items-center mt-2 px-2 text-xs text-gray-400">
        <span className="font-mono">25 mm/s • 10 mm/mV</span>
        <span>Lead {LEAD_NAMES[selectedLead] || selectedLead + 1} • {numLeads}-lead ECG</span>
      </div>
    </div>
  );
};

export default ECGGraph;
