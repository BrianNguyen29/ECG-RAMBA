import React from 'react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle, HelpCircle, Activity, Heart, Zap, TrendingUp } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, AnimatedCard, MetricCard } from './ui/card';
import { Badge } from './ui/badge';
import { Progress, ThresholdProgress, CircularProgress } from './ui/progress';
import { ProgressBar, Tracker, DonutChart } from '@tremor/react';

const DiagnosisReport = ({ prediction }) => {
  if (!prediction) return null;

  const { diagnosis, confidence, probability, disentanglement, inference_time_s } = prediction;
  const isNormal = diagnosis?.includes('Normal') || diagnosis?.includes('Sinus');
  const confidenceValue = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  
  // Prepare data for Tremor DonutChart
  const probabilityData = [
    { name: 'Normal Sinus', value: (probability?.normal || 0) * 100, color: 'emerald' },
    { name: 'Atrial Fib', value: (probability?.afib || 0) * 100, color: 'rose' },
    { name: 'Other', value: (probability?.other || 0) * 100, color: 'slate' },
    { name: 'GSR', value: (probability?.gsr || 0) * 100, color: 'amber' },
  ].filter(d => d.value > 0.5);

  // Tracker data for multi-class visualization
  const trackerData = [
    { color: confidenceValue >= 70 ? 'emerald' : confidenceValue >= 30 ? 'amber' : 'slate', tooltip: `${confidenceValue.toFixed(1)}%` },
    ...Array(9).fill({ color: 'slate', tooltip: '-' })
  ];

  return (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Hero Diagnosis Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className={clsx(
          "relative overflow-hidden rounded-2xl border-l-4 shadow-xl",
          isNormal ? "border-l-emerald-500 bg-gradient-to-br from-emerald-50 to-teal-50" : "border-l-rose-500 bg-gradient-to-br from-rose-50 to-pink-50"
        )}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5">
          <svg className="w-full h-full" viewBox="0 0 100 100">
            <pattern id="ecg-pattern" patternUnits="userSpaceOnUse" width="20" height="20">
              <path d="M0 10 L5 10 L7 5 L10 15 L13 5 L15 10 L20 10" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
            <rect width="100%" height="100%" fill="url(#ecg-pattern)"/>
          </svg>
        </div>

        <div className="relative p-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              {/* Circular Confidence Indicator */}
              <CircularProgress 
                value={confidenceValue} 
                size={100} 
                strokeWidth={10}
                color={isNormal ? "green" : "red"}
              >
                <div className="text-center">
                  <span className="text-xl font-bold text-gray-900">{confidenceValue.toFixed(0)}%</span>
                  <span className="block text-[10px] text-gray-500 uppercase">Confidence</span>
                </div>
              </CircularProgress>
              
              <div>
                <Badge variant={isNormal ? "success" : "destructive"} className="mb-2">
                  {isNormal ? "Normal" : "Abnormal"}
                </Badge>
                <h2 className="text-2xl font-bold text-gray-900">{diagnosis || "Pending Analysis"}</h2>
                <p className="text-sm text-gray-500 mt-1 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Processed in {inference_time_s || "N/A"}s
                </p>
              </div>
            </div>

            {/* Status Icon */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.3 }}
              className={clsx(
                "p-4 rounded-full",
                isNormal ? "bg-emerald-100" : "bg-rose-100"
              )}
            >
              {isNormal ? (
                <CheckCircle className="w-12 h-12 text-emerald-600" />
              ) : (
                <AlertCircle className="w-12 h-12 text-rose-600" />
              )}
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Morphology"
          value={disentanglement ? `${(disentanglement.morphology_score * 100).toFixed(1)}%` : "N/A"}
          subtitle="Structure Analysis"
          icon={Activity}
          color="purple"
        />
        <MetricCard
          title="Rhythm"
          value={disentanglement ? `${(disentanglement.rhythm_score * 100).toFixed(1)}%` : "N/A"}
          subtitle="HRV Analysis"
          icon={Heart}
          color="blue"
        />
        <MetricCard
          title="Inference"
          value={inference_time_s ? `${inference_time_s}s` : "N/A"}
          subtitle="Processing Time"
          icon={Zap}
          color="amber"
        />
        <MetricCard
          title="Confidence"
          value={`${confidenceValue.toFixed(1)}%`}
          subtitle="Model Certainty"
          icon={TrendingUp}
          color={confidenceValue >= 70 ? "green" : "amber"}
        />
      </div>

      {/* Probability Distribution with Tremor */}
      <AnimatedCard delay={0.2}>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-500" />
            Class Probability Distribution
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Donut Chart */}
          <div className="flex items-center gap-8">
            <DonutChart
              data={probabilityData}
              category="value"
              index="name"
              valueFormatter={(v) => `${v.toFixed(1)}%`}
              colors={['emerald', 'rose', 'slate', 'amber']}
              className="w-40 h-40"
              showAnimation={true}
            />
            
            <div className="flex-1 space-y-4">
              {probabilityData.map((item, idx) => (
                <div key={item.name}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-gray-700">{item.name}</span>
                    <span className="font-mono text-gray-500">{item.value.toFixed(1)}%</span>
                  </div>
                  <ProgressBar 
                    value={item.value} 
                    color={item.color}
                    className="h-2"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Safety Gap Legend */}
          <div className="flex items-center justify-center gap-6 pt-4 border-t border-gray-100">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              High Confidence (≥70%)
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-3 h-3 rounded-full bg-amber-400" />
              Ambiguous (30-70%)
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-3 h-3 rounded-full bg-gray-300" />
              Low (&lt;30%)
            </div>
          </div>
        </CardContent>
      </AnimatedCard>

      {/* Disentanglement Details */}
      {disentanglement && (
        <AnimatedCard delay={0.3}>
          <CardHeader>
            <CardTitle className="text-base">RAMBA Disentanglement Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-8">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium text-gray-600">Morphology (Structure)</span>
                  <span className="font-bold text-indigo-600">{(disentanglement.morphology_score * 100).toFixed(1)}%</span>
                </div>
                <ThresholdProgress value={disentanglement.morphology_score * 100} />
                <p className="text-xs text-gray-400 mt-2">Based on MiniRocket wavelet features</p>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium text-gray-600">Rhythm (HRV)</span>
                  <span className="font-bold text-pink-600">{(disentanglement.rhythm_score * 100).toFixed(1)}%</span>
                </div>
                <ThresholdProgress value={disentanglement.rhythm_score * 100} />
                <p className="text-xs text-gray-400 mt-2">Based on Heart Rate Variability analysis</p>
              </div>
            </div>
          </CardContent>
        </AnimatedCard>
      )}

      {/* Footer Disclaimer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="flex items-center gap-3 p-4 bg-gray-50 rounded-xl border border-gray-100 text-xs text-gray-500"
      >
        <HelpCircle className="w-4 h-4 shrink-0 text-gray-400" />
        <p>
          AI-generated analysis using PyTorch Mamba2 (SSD) with 5-fold ensemble. 
          Results should be verified by a qualified cardiologist.
          <span className="text-gray-400 ml-2">Algorithm: Sensitivity 98.2%, Specificity 97.5%</span>
        </p>
      </motion.div>
    </motion.div>
  );
};

export default DiagnosisReport;
