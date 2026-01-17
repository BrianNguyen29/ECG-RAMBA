import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const FrequencyGraph = ({ data, color = "#8884d8" }) => {
    // Data format: { freqs: [], psd: [] } -> [{freq, power}, ...]
    
    if (!data || !data.freqs) return <div className="flex items-center justify-center h-full text-slate-400 text-xs">No Frequency Data</div>;

    const chartData = data.freqs.map((f, i) => ({
        freq: f.toFixed(1),
        power: data.psd[i]
    })).filter(d => parseFloat(d.freq) <= 60); // Focus on 0-60Hz for EEG/ECG

    return (
        <div className="w-full h-full min-h-[200px] bg-slate-900/50 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                    <defs>
                        <linearGradient id="colorPower" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
                            <stop offset="95%" stopColor={color} stopOpacity={0}/>
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis 
                        dataKey="freq" 
                        stroke="#94a3b8" 
                        tick={{fontSize: 10}} 
                        label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 10 }}
                    />
                    <YAxis hide />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                        itemStyle={{ color: color }}
                        labelStyle={{ color: '#94a3b8' }}
                    />
                    <Area type="monotone" dataKey="power" stroke={color} fillOpacity={1} fill="url(#colorPower)" />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default FrequencyGraph;
