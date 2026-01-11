import React, { useState, useEffect } from 'react';
import { Activity, Radio, Cpu, CheckCircle2 } from 'lucide-react';
import clsx from 'clsx';

const PROCESSING_STEPS = [
  {
    id: 1,
    title: "Signal Preprocessing",
    detail: "Bandpass Filter (0.5-40Hz) & Z-Score Normalization",
    icon: Activity,
    color: "text-blue-500",
    bg: "bg-blue-50",
    border: "border-blue-200"
  },
  {
    id: 2,
    title: "Feature Extraction",
    detail: "MiniRocket Transformation & HRV Analysis",
    icon: Radio,
    color: "text-purple-500",
    bg: "bg-purple-50",
    border: "border-purple-200"
  },
  {
    id: 3,
    title: "Context Modeling",
    detail: "Mamba Dual-Pathway Sequence Scan",
    icon: Cpu,
    color: "text-indigo-500",
    bg: "bg-indigo-50",
    border: "border-indigo-200"
  }
];

const ScientificLoader = () => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentStep(prev => (prev < PROCESSING_STEPS.length - 1 ? prev + 1 : prev));
    }, 800); // Advance step every 800ms

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-white rounded-2xl shadow-xl border border-gray-100">
      <div className="text-center mb-6">
        <h3 className="text-lg font-semibold text-gray-800">Processing Signal</h3>
        <p className="text-xs text-gray-500">ECG-RAMBA Protocol-Faithful Pipeline</p>
      </div>

      <div className="space-y-4">
        {PROCESSING_STEPS.map((step, index) => {
          const isActive = index === currentStep;
          const isCompleted = index < currentStep;
          const isPending = index > currentStep;

          return (
            <div 
              key={step.id}
              className={clsx(
                "flex items-center gap-4 p-3 rounded-xl border transition-all duration-500",
                isActive 
                  ? `${step.bg} ${step.border} shadow-sm scale-105` 
                  : isCompleted 
                    ? "bg-green-50 border-green-100 opacity-60" 
                    : "bg-gray-50 border-gray-100 opacity-40 grayscale"
              )}
            >
              <div className={clsx(
                "w-10 h-10 rounded-full flex items-center justify-center transition-all",
                isActive ? "bg-white shadow-sm" : isCompleted ? "bg-green-100" : "bg-gray-200"
              )}>
                {isCompleted ? (
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                ) : (
                    <step.icon className={clsx("w-5 h-5", isActive ? step.color : "text-gray-400")} />
                )}
              </div>
              
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <h4 className={clsx("text-sm font-medium", isActive ? "text-gray-900" : "text-gray-600")}>
                    {step.title}
                  </h4>
                  {isActive && (
                    <span className="flex h-2 w-2 relative">
                      <span className={clsx("animate-ping absolute inline-flex h-full w-full rounded-full opacity-75", step.color.replace('text-', 'bg-'))}></span>
                      <span className={clsx("relative inline-flex rounded-full h-2 w-2", step.color.replace('text-', 'bg-'))}></span>
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-400 mt-0.5">{step.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="mt-6">
        <div className="h-1 w-full bg-gray-100 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 transition-all duration-300 ease-out"
            style={{ width: `${((currentStep + 1) / PROCESSING_STEPS.length) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default ScientificLoader;
