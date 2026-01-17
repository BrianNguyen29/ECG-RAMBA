import React, { useMemo, useState } from 'react';
import { LineChart, Line, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { Activity, Maximize2, Minimize2, Settings } from 'lucide-react';
import clsx from 'clsx';

const EEGViewer = ({ data, channels, sampleRate = 250, warnings = [], darkMode = false }) => {
  const [gain, setGain] = useState(1); // Scale factor
  const [timeWindow, setTimeWindow] = useState(10); // Seconds to show
  
  // Prepare data for rendering
  // Data comes as [channels, samples]
  // We need to subsample for performance if specific duration is long
  
  const processedChannels = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Default channels if not provided
    const channelNames = channels || data.map((_, i) => `Ch ${i+1}`);
    
    return data.map((channelData, idx) => {
      // Subsample for display (aim for ~500 points per visible window)
      // If full signal is passed, we might want to window it. 
      // For now, assume we render the whole chunk passed (usually 10s from backend?)
      // Check data length vs sampleRate
      
      const chartData = [];
      const step = Math.max(1, Math.floor(channelData.length / 1000));
      
      for (let i = 0; i < channelData.length; i += step) {
        chartData.push({
          i: i,
          v: channelData[i] * gain // Apply gain
        });
      }
      return {
        name: channelNames[idx],
        data: chartData
      };
    });
  }, [data, channels, gain]);

  if (!data || data.length === 0) {
    return (
       <div className={clsx(
        "flex flex-col items-center justify-center h-full relative overflow-hidden",
        darkMode ? "bg-slate-900 text-slate-500" : "bg-slate-50 text-slate-400"
      )}>
        <Activity className="w-12 h-12 mb-4 opacity-50" />
        <p>NO EEG SIGNAL</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col relative">
      {/* Controls Header */}
      <div className={clsx(
          "h-10 border-b flex items-center justify-between px-4 z-10",
           darkMode ? "bg-slate-800 border-slate-700" : "bg-slate-50 border-slate-200"
      )}>
          <div className="flex items-center gap-2">
            <span className={clsx("text-xs font-bold", darkMode ? "text-slate-300" : "text-slate-600")}>
                EEG / {processedChannels.length} Channels
            </span>
             {warnings.length > 0 && (
                <span className="text-[10px] text-amber-500 bg-amber-100 px-2 py-0.5 rounded">
                    {warnings.length} Warnings
                </span>
             )}
          </div>
          
          <div className="flex items-center gap-2">
             <button onClick={() => setGain(g => Math.max(0.5, g - 0.5))} className="p-1 px-2 border rounded text-xs" aria-label="Decrease gain">- Gain</button>
             <span className="text-xs font-mono w-12 text-center">{gain}x</span>
             <button onClick={() => setGain(g => Math.min(5, g + 0.5))} className="p-1 px-2 border rounded text-xs" aria-label="Increase gain">+ Gain</button>
          </div>
      </div>

      {/* Charts Area */}
      <div className={clsx(
          "flex-1 overflow-y-auto scrollbar-thin relative p-2 md:p-4", 
          darkMode ? "bg-slate-900" : "bg-white"
      )}>
         {/* Background Grid */}
          <div className={clsx(
           "absolute inset-0 opacity-100 pointer-events-none",
           darkMode ? "ecg-grid-dark" : "ecg-grid-light"
         )} />

         <div className="flex flex-col gap-2">
            {processedChannels.map((ch, idx) => (
                <div key={idx} className="h-16 relative flex items-center">
                    {/* Channel Label */}
                    <div className={clsx(
                        "w-12 text-[10px] font-bold text-right pr-2 shrink-0 z-10",
                        darkMode ? "text-slate-400" : "text-slate-500"
                    )}>
                        {ch.name}
                    </div>
                    
                    {/* Chart */}
                    <div className="flex-1 h-full border-b border-slate-100/10">
                         <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={ch.data}>
                                <Line 
                                    type="monotone" 
                                    dataKey="v" 
                                    stroke={darkMode ? "#a5b4fc" : "#4f46e5"} 
                                    strokeWidth={1} 
                                    dot={false}
                                    isAnimationActive={false} 
                                />
                                <YAxis domain={['dataMin', 'dataMax']} hide />
                            </LineChart>
                         </ResponsiveContainer>
                    </div>
                </div>
            ))}
         </div>
      </div>
      
      {/* Footer Info */}
      <div className={clsx(
          "h-6 border-t flex items-center justify-between px-4 text-[9px] font-mono",
          darkMode ? "bg-slate-950 border-slate-800 text-slate-500" : "bg-slate-100 border-slate-200 text-slate-400"
      )}>
          <span>Fs: {sampleRate} Hz</span>
          <span>Window: {timeWindow}s</span>
          <span>Gain: {gain * 10}uV/mm</span>
      </div>
    </div>
  );
};

export default EEGViewer;
