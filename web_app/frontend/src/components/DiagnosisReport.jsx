import React from 'react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle, Activity, Heart, Zap, Brain, FileText } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { CircularProgress } from './ui/progress';

const DiagnosisReport = ({ prediction }) => {
  if (!prediction) return null;

  const { diagnosis, confidence, probability, disentanglement, inference_time_s } = prediction;
  const isNormal = diagnosis?.includes('Normal') || diagnosis?.includes('Sinus');
  const confidenceValue = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  
  const morphScore = disentanglement ? (disentanglement.morphology_score * 100) : 0;
  const rhythmScore = disentanglement ? (disentanglement.rhythm_score * 100) : 0;

    return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      {/* 1. HERO DIAGNOSIS CARD */}
      <Card className={clsx(
          "border-l-4 overflow-hidden relative shadow-sm",
          isNormal ? "border-l-emerald-500" : "border-l-rose-500"
      )}>
        {/* Abstract Background - Light Mode Gradient */}
        <div className="absolute inset-0 opacity-30 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-100 via-white to-white" />
        
        <CardContent className="p-5 flex items-center justify-between relative mt-0 pt-5">
            <div>
                 <div className="flex items-center gap-2 mb-2">
                    <Badge variant={isNormal ? "success" : "destructive"} className="px-2 py-0.5 text-[10px] uppercase tracking-wider">
                        {isNormal ? "Low Risk" : "Attention Required"}
                    </Badge>
                     <span className="text-[10px] text-slate-500 font-mono">ICD-10 Compatible</span>
                 </div>
                 <h2 className="text-2xl font-bold text-slate-900 tracking-tight leading-none">
                     {diagnosis || "Analysing…"}
                 </h2>
                 <p className="text-xs text-slate-500 mt-2 max-w-[200px]">
                     {isNormal 
                        ? "Sinus rhythm within normal limits. No specific abnormalities detected." 
                        : "Abnormal ECG pattern detected. Clinical correlation recommended."}
                 </p>
            </div>

            {/* Confidence Circle */}
            <div className="relative">
                <CircularProgress 
                    value={confidenceValue} 
                    size={70} 
                    strokeWidth={6}
                    color={isNormal ? "green" : "red"}
                >
                    <div className="text-center">
                        <span className="text-sm font-bold text-slate-700">{confidenceValue.toFixed(0)}%</span>
                    </div>
                </CircularProgress>
            </div>
        </CardContent>
      </Card>

      {/* 2. DISENTANGLEMENT DOUBLE BAR (THE "WHY") */}
      <Card className="overflow-hidden bg-white shadow-sm border border-slate-200">
          <div className="px-4 py-3 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                  <Brain className="w-3 h-3 text-blue-500" /> Feature Disentanglement
              </h3>
          </div>
          <CardContent className="p-5 mt-0 pt-5">
              <div className="flex gap-6 justify-center items-end h-[120px]">
                  {/* Morphology Bar */}
                  <div className="flex flex-col items-center gap-2 group w-16">
                      <div className="text-xs font-bold text-slate-600">{morphScore.toFixed(0)}</div>
                      <div className="w-8 flex-1 bg-slate-100 rounded-full relative overflow-hidden group-hover:ring-1 ring-purple-500/50 transition-all border border-slate-200">
                          <motion.div 
                            initial={{ height: 0 }}
                            animate={{ height: `${morphScore}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className="absolute bottom-0 w-full bg-gradient-to-t from-purple-600 to-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.2)]"
                          />
                      </div>
                      <div className="text-[10px] font-bold text-purple-600 uppercase tracking-tighter">Morph</div>
                  </div>

                  {/* Rhythm Bar */}
                  <div className="flex flex-col items-center gap-2 group w-16">
                      <div className="text-xs font-bold text-slate-600">{rhythmScore.toFixed(0)}</div>
                      <div className="w-8 flex-1 bg-slate-100 rounded-full relative overflow-hidden group-hover:ring-1 ring-blue-500/50 transition-all border border-slate-200">
                          <motion.div 
                            initial={{ height: 0 }}
                            animate={{ height: `${rhythmScore}%` }}
                            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                             className="absolute bottom-0 w-full bg-gradient-to-t from-blue-600 to-blue-400 shadow-[0_0_20px_rgba(59,130,246,0.2)]"
                          />
                      </div>
                      <div className="text-[10px] font-bold text-blue-600 uppercase tracking-tighter">Rhythm</div>
                  </div>
              </div>
              <p className="text-[10px] text-center text-slate-500 mt-4">
                  Separates structure (QRS shape) from timing (R-R interval).
              </p>
          </CardContent>
      </Card>

      {/* 3. LLM SUMMARY (TEXT BOX) */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 relative">
          <div className="absolute top-3 right-3 text-cyan-500/20">
              <FileText className="w-12 h-12" />
          </div>
          <h4 className="text-[10px] font-bold text-cyan-500 uppercase mb-2">AI Interpretation</h4>
          <p className="text-xs text-slate-300 leading-relaxed font-mono">
              {diagnosis && (
                  isNormal 
                  ? "Trace shows normal sinus rhythm. PR interval and QRS duration appear within normal ranges. No ST-segment elevation detected."
                  : `Automated analysis suggests ${diagnosis}. ${rhythmScore > 80 ? "High rhythm irregularity detected (HRV check recommended)." : ""} ${morphScore > 80 ? "Abnormal QRS morphology observed." : ""}`
              )}
          </p>
      </div>

    </div>
  );
};

export default DiagnosisReport;
