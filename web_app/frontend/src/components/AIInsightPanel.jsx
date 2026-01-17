import React from 'react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import { Brain, Heart, Activity, Zap, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';

// Circular Progress for Confidence
const ConfidenceGauge = ({ value, size = 120, strokeWidth = 8, color = 'blue' }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (value / 100) * circumference;
    
    const colorMap = {
        green: { stroke: '#10b981', bg: '#d1fae5' },
        red: { stroke: '#ef4444', bg: '#fee2e2' },
        blue: { stroke: '#3b82f6', bg: '#dbeafe' },
        yellow: { stroke: '#f59e0b', bg: '#fef3c7' }
    };
    
    const colors = colorMap[color] || colorMap.blue;
    
    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="transform -rotate-90">
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={colors.bg}
                    strokeWidth={strokeWidth}
                />
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={colors.stroke}
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                    style={{ strokeDasharray: circumference }}
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold text-slate-800">{value.toFixed(0)}%</span>
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Confidence</span>
            </div>
        </div>
    );
};

// Disentanglement Bar
const DisentanglementBar = ({ label, value, color, icon: Icon }) => (
    <div className="space-y-1">
        <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5">
                <Icon className={`w-3.5 h-3.5 ${color}`} />
                <span className="font-semibold text-slate-700">{label}</span>
            </div>
            <span className={`font-bold ${color}`}>{(value * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <motion.div
                className={`h-full rounded-full ${
                    label === 'Morphology' ? 'bg-gradient-to-r from-purple-500 to-purple-400' :
                    'bg-gradient-to-r from-blue-500 to-blue-400'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${value * 100}%` }}
                transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
            />
        </div>
    </div>
);

const AIInsightPanel = React.memo(({ prediction, loading, darkMode = false }) => {
    if (loading) {
        return (
            <div className={clsx(
                "h-full flex flex-col items-center justify-center gap-4 p-6",
                darkMode ? "bg-slate-800" : "bg-slate-50"
            )}>
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
                    <Brain className="absolute inset-0 m-auto w-6 h-6 text-blue-500" />
                </div>
                <div className="text-center">
                    <p className={clsx("text-sm font-semibold", darkMode ? "text-slate-200" : "text-slate-700")}>
                        AI Analyzing ECG...
                    </p>
                    <p className={clsx("text-xs mt-1", darkMode ? "text-slate-400" : "text-slate-500")}>
                        Mamba Context Modeling
                    </p>
                </div>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className={clsx(
                "h-full flex flex-col items-center justify-center gap-4 p-6 text-center",
                darkMode ? "bg-slate-800" : "bg-slate-50"
            )}>
                <div className={clsx(
                    "w-20 h-20 rounded-full flex items-center justify-center",
                    darkMode ? "bg-slate-700" : "bg-slate-100"
                )}>
                    <Brain className={clsx("w-10 h-10", darkMode ? "text-slate-500" : "text-slate-400")} />
                </div>
                <div>
                    <p className={clsx("text-sm font-semibold", darkMode ? "text-slate-300" : "text-slate-600")}>
                        AI Insights Ready
                    </p>
                    <p className={clsx("text-xs mt-1", darkMode ? "text-slate-400" : "text-slate-500")}>
                        Upload ECG and click "Analyze" to see results
                    </p>
                </div>
            </div>
        );
    }

    const { 
        diagnosis, 
        top_diagnosis,
        confidence, 
        disentanglement, 
        inference_time_s,
        predictions = [],
        all_probabilities = {}
    } = prediction;

    const diagnosisName = top_diagnosis || diagnosis || "Unknown";
    const confidenceValue = typeof confidence === 'number' ? confidence * 100 : parseFloat(confidence) || 0;
    const isNormal = diagnosisName.toLowerCase().includes('normal') || diagnosisName.toLowerCase().includes('sinus');
    
    const morphScore = disentanglement?.morphology_score || 0;
    const rhythmScore = disentanglement?.rhythm_score || 0;

    return (
        <div className={clsx(
            "h-full flex flex-col gap-4 p-4 overflow-y-auto scrollbar-thin",
            darkMode ? "bg-slate-800" : "bg-slate-50"
        )}>
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className={clsx(
                    "text-xs font-bold uppercase tracking-widest flex items-center gap-2",
                    darkMode ? "text-slate-400" : "text-slate-500"
                )}>
                    <Brain className="w-3.5 h-3.5 text-blue-500" />
                    AI Diagnosis
                </h2>
                {inference_time_s && (
                    <Badge variant="outline" className={clsx(
                        "text-[9px] h-5",
                        darkMode ? "border-slate-600 text-slate-400" : "border-slate-300 text-slate-500"
                    )}>
                        <Zap className="w-2.5 h-2.5 mr-1 text-amber-500" />
                        {inference_time_s}s
                    </Badge>
                )}
            </div>

            {/* Main Diagnosis Card */}
            <Card className={clsx(
                "overflow-hidden border-l-4",
                isNormal ? "border-l-emerald-500" : "border-l-rose-500",
                darkMode ? "bg-slate-700 border-slate-600" : "bg-white"
            )}>
                <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                            {isNormal ? (
                                <CheckCircle className="w-5 h-5 text-emerald-500" />
                            ) : (
                                <AlertTriangle className="w-5 h-5 text-rose-500" />
                            )}
                            <Badge variant={isNormal ? "success" : "destructive"} className="text-[9px] uppercase">
                                {isNormal ? "Low Risk" : "Attention Required"}
                            </Badge>
                        </div>
                        <h3 className={clsx(
                            "text-xl font-bold tracking-tight",
                            darkMode ? "text-white" : "text-slate-900"
                        )}>
                            {diagnosisName}
                        </h3>
                        <p className={clsx(
                            "text-xs mt-1",
                            darkMode ? "text-slate-400" : "text-slate-500"
                        )}>
                            {isNormal 
                                ? "No significant abnormalities detected" 
                                : "Clinical correlation recommended"}
                        </p>
                    </div>
                    <ConfidenceGauge 
                        value={confidenceValue} 
                        size={90} 
                        strokeWidth={6}
                        color={isNormal ? 'green' : 'red'}
                    />
                </CardContent>
            </Card>

            {/* Disentanglement Chart */}
            <Card className={clsx(
                darkMode ? "bg-slate-700 border-slate-600" : "bg-white"
            )}>
                <CardContent className="p-4 space-y-4">
                    <div className="space-y-1">
                        <h4 className={clsx(
                            "text-xs font-bold uppercase tracking-wider",
                            darkMode ? "text-slate-400" : "text-slate-500"
                        )}>
                            Feature Disentanglement
                        </h4>
                        <span className={clsx(
                            "text-[9px] block",
                            darkMode ? "text-slate-500" : "text-slate-400"
                        )}>
                            "Why this prediction?"
                        </span>
                    </div>
                    
                    <DisentanglementBar 
                        label="Morphology" 
                        value={morphScore}
                        color="text-purple-600"
                        icon={Heart}
                    />
                    <DisentanglementBar 
                        label="Rhythm" 
                        value={rhythmScore}
                        color="text-blue-600"
                        icon={Activity}
                    />
                    
                    <p className={clsx(
                        "text-[10px] italic pt-2 border-t",
                        darkMode ? "text-slate-500 border-slate-600" : "text-slate-400 border-slate-100"
                    )}>
                        {rhythmScore > morphScore 
                            ? "Diagnosis based primarily on rhythm abnormalities (HRV + Mamba Context)"
                            : "Diagnosis based primarily on morphological features (QRS/ST patterns)"
                        }
                    </p>
                </CardContent>
            </Card>

            {/* All Probabilities */}
            {Object.keys(all_probabilities).length > 0 && (
                <Card className={clsx(
                    darkMode ? "bg-slate-700 border-slate-600" : "bg-white"
                )}>
                    <CardContent className="p-4">
                        <h4 className={clsx(
                            "text-xs font-bold uppercase tracking-wider mb-3",
                            darkMode ? "text-slate-400" : "text-slate-500"
                        )}>
                            Class Probabilities
                        </h4>
                        <div className="space-y-2">
                            {Object.entries(all_probabilities)
                                .sort((a, b) => b[1] - a[1])
                                .map(([className, prob]) => (
                                    <div key={className} className="flex items-center justify-between text-xs">
                                        <span className={clsx(
                                            darkMode ? "text-slate-300" : "text-slate-700"
                                        )}>{className}</span>
                                        <span className={clsx(
                                            "font-mono font-bold",
                                            prob > 0.5 ? "text-rose-500" : (darkMode ? "text-slate-400" : "text-slate-500")
                                        )}>
                                            {(prob * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))
                            }
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* HRV Metrics */}
            {disentanglement?.hrv_metrics && (
                <Card className={clsx(
                    darkMode ? "bg-slate-700 border-slate-600" : "bg-white"
                )}>
                    <CardContent className="p-4">
                        <h4 className={clsx(
                            "text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2",
                            darkMode ? "text-slate-400" : "text-slate-500"
                        )}>
                            <TrendingUp className="w-3 h-3" />
                            HRV Metrics
                        </h4>
                        <div className="flex flex-wrap gap-2">
                            {disentanglement.hrv_metrics.raw_vector?.slice(0, 5).map((val, i) => (
                                <div key={i} className={clsx(
                                    "text-center px-3 py-2 rounded-lg flex-1 min-w-[55px]",
                                    darkMode ? "bg-slate-600" : "bg-slate-50"
                                )}>
                                    <div className={clsx(
                                        "text-sm font-mono font-bold truncate",
                                        darkMode ? "text-slate-200" : "text-slate-700"
                                    )}>
                                        {val.toFixed(1)}
                                    </div>
                                    <div className={clsx(
                                        "text-[8px] uppercase",
                                        darkMode ? "text-slate-500" : "text-slate-400"
                                    )}>
                                        HRV{i + 1}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
});

export default AIInsightPanel;
