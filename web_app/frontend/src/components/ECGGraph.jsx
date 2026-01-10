import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ZoomIn, ZoomOut, RotateCcw, Activity } from 'lucide-react';

const ECGGraph = ({ data }) => {
  // Simple zoom state (view window)
  const [zoomLeft, setZoomLeft] = useState(0);
  const [zoomRight, setZoomRight] = useState(data && data.length > 0 ? data.length : 100);
  
  // Update zoom window when data loads
  React.useEffect(() => {
     if (data && data.length > 0) {
        setZoomRight(data.length);
        setZoomLeft(0);
     }
  }, [data]);

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-80 bg-white rounded-xl border border-gray-100 shadow-sm relative overflow-hidden">
        {/* Placeholder grid background */}
        <div className="absolute inset-0 opacity-[0.03]" 
             style={{ backgroundImage: 'linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
        </div>
        <div className="z-10 text-center">
           <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-3">
              <Activity className="w-8 h-8 text-gray-300" />
           </div>
           <p className="text-gray-500 font-medium">No Signal Loaded</p>
           <p className="text-xs text-gray-400 mt-1">Upload a record to visualize</p>
        </div>
      </div>
    );
  }

  // Slice data based on zoom
  // Optimization: Downsample if view window is huge
  const visibleData = data.slice(zoomLeft, zoomRight).map((val, i) => ({
      index: i + zoomLeft, 
      value: val 
  }));

  const handleReset = () => {
     setZoomLeft(0);
     setZoomRight(data.length);
  };

  const handleZoomIn = () => {
    const range = zoomRight - zoomLeft;
    const quarter = Math.floor(range / 4);
    if (range > 20) {
        setZoomLeft(zoomLeft + quarter);
        setZoomRight(zoomRight - quarter);
    }
  };
  
  const handleZoomOut = () => {
      const range = zoomRight - zoomLeft;
      const quarter = Math.floor(range / 2);
      setZoomLeft(Math.max(0, zoomLeft - quarter));
      setZoomRight(Math.min(data.length, zoomRight + quarter));
  };


  return (
    <div className="w-full bg-white p-4 rounded-xl shadow-sm border border-gray-100 relative group">
      <div className="flex items-center justify-between mb-2 px-2">
         <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            ECG Lead I (Filtered)
         </h3>
         <div className="flex items-center gap-1 bg-gray-50 rounded-lg p-1">
             <button onClick={handleZoomOut} className="p-1.5 hover:bg-white hover:shadow-sm rounded transition-all text-gray-500 hover:text-gray-900" title="Zoom Out">
                <ZoomOut className="w-4 h-4" />
             </button>
             <button onClick={handleReset} className="p-1.5 hover:bg-white hover:shadow-sm rounded transition-all text-gray-500 hover:text-gray-900" title="Reset View">
                <RotateCcw className="w-4 h-4" />
             </button>
             <button onClick={handleZoomIn} className="p-1.5 hover:bg-white hover:shadow-sm rounded transition-all text-gray-500 hover:text-gray-900" title="Zoom In">
                <ZoomIn className="w-4 h-4" />
             </button>
         </div>
      </div>

      <div className="h-72 w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={visibleData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis dataKey="index" hide />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
              itemStyle={{ color: '#ef4444', fontWeight: 600 }}
              labelFormatter={(label) => `Sample: ${label}`}
              formatter={(value) => [value.toFixed(3), 'Amplitude']}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#ef4444" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        
        {/* Custom Medical Grid Overlay (Visual Only) */}
        <div className="absolute inset-0 pointer-events-none border border-red-500/10 rounded-lg"></div>
      </div>
      
      <div className="flex justify-between items-center mt-2 px-2">
          <span className="text-xs text-gray-400 font-mono">25mm/s</span>
          <span className="text-xs text-gray-400 font-mono">10mm/mV</span>
      </div>
    </div>
  );
};

export default ECGGraph;
