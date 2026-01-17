import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  BookOpen,
  HelpCircle,
  CheckCircle,
  XCircle,
  GraduationCap,
  Scale,
  AlertTriangle,
} from "lucide-react";
import DiagnosisReport from "./DiagnosisReport";
import TutorChat from "./TutorChat";
import clsx from "clsx";
import { motion, AnimatePresence } from "framer-motion";

// --- SOKOLOW-LYON CALCULATOR (Simplified) ---
// S in V1 + R in V5 or V6 > 3.5 mV (35mm)
const calculateSokolowLyon = (signal) => {
  if (!signal || signal.length < 12) return null;

  // Lead Indices: V1=6, V5=10, V6=11
  const v1 = signal[6];
  const v5 = signal[10];
  const v6 = signal[11];

  // Naively find Min (S-wave depth) and Max (R-wave height)
  // Note: Signal is Z-normalized in main app, but we expect Raw for typical calculation.
  // This is a naive estimation assuming signal scale ~ 1 unit = 1mV approx if normalized.
  // For educational purpose, we'll simulate the check logic.

  const s_v1 = Math.abs(Math.min(...v1));
  const r_v5 = Math.max(...v5);
  const r_v6 = Math.max(...v6);

  const index = s_v1 + Math.max(r_v5, r_v6);
  return {
    value: index.toFixed(2),
    isPositive: index > 3.5, // Threshold
    sv1: s_v1.toFixed(2),
    rv5_6: Math.max(r_v5, r_v6).toFixed(2),
  };
};

const EducationMode = ({ prediction, signalData }) => {
  const [revealed, setRevealed] = useState(false);
  const [userGuess, setUserGuess] = useState(null);
  const [chatMinimized, setChatMinimized] = useState(false); // DeepTutor State

  // Mock Guidelines Calculation
  const sokolow = signalData ? calculateSokolowLyon(signalData) : null;

  const OPTIONS = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Left Bundle Branch Block (LBBB)",
    "Right Bundle Branch Block (RBBB)",
    "Premature Ventricular Contraction",
  ];

  const handleGuess = (option) => {
    setUserGuess(option);
    setRevealed(true);
  };

  const isCorrect =
    prediction &&
    userGuess &&
    prediction.diagnosis.includes(userGuess.split(" ")[0]); // Loose match

  return (
    <div className="h-full p-4 overflow-y-auto scrollbar-thin max-w-5xl mx-auto space-y-8">
      {/* HERO HEADER */}
      <div className="text-center space-y-2">
        <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-indigo-100">
          <GraduationCap className="w-6 h-6 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900">
          Interactive Case Study
        </h2>
        <p className="text-slate-500 max-w-md mx-auto">
          Analyze the ECG trace provided in the viewer. Select your diagnosis
          below, then compare with Guidelines and AI.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* LEFT: QUIZ PANEL */}
        <Card className="border-indigo-100 bg-white shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-indigo-700">
              <HelpCircle className="w-5 h-5 text-indigo-500" />
              What is your diagnosis?
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!revealed ? (
              <div className="grid grid-cols-1 gap-2">
                {OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => handleGuess(opt)}
                    className="p-4 text-left rounded-xl bg-slate-50 hover:bg-indigo-50 hover:border-indigo-200 border border-slate-100 transition-all text-slate-700 font-medium hover:text-indigo-700 hover:shadow-sm"
                  >
                    {opt}
                  </button>
                ))}
              </div>
            ) : (
              <div className="space-y-6 animate-in zoom-in-95 duration-300">
                <div
                  className={clsx(
                    "p-6 rounded-2xl text-center border shadow-sm",
                    isCorrect
                      ? "bg-emerald-50 border-emerald-200"
                      : "bg-rose-50 border-rose-200"
                  )}
                >
                  {isCorrect ? (
                    <>
                      <CheckCircle className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
                      <h3 className="text-xl font-bold text-emerald-800">
                        Correct Diagnosis!
                      </h3>
                      <p className="text-emerald-600 mt-1">
                        Excellent analysis. The AI agrees with you.
                      </p>
                    </>
                  ) : (
                    <>
                      <XCircle className="w-12 h-12 text-rose-500 mx-auto mb-3" />
                      <h3 className="text-xl font-bold text-rose-800">
                        Review Required
                      </h3>
                      <p className="text-rose-600 mt-1">
                        You selected <b>{userGuess}</b>, but the AI suggests
                        looking closer at <b>{prediction?.diagnosis}</b>.
                      </p>
                    </>
                  )}
                </div>

                <Button
                  onClick={() => setRevealed(false)}
                  className="w-full bg-slate-900 text-white hover:bg-slate-800"
                >
                  Try Another Case
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* RIGHT: GUIDELINES & AI REVEAL */}
        <div className="space-y-4">
          {revealed && prediction && (
            <div className="animate-in slide-in-from-right-8 duration-500 space-y-4">
              {/* AI Result */}
              <div className="relative">
                <div className="absolute -top-3 left-4 bg-purple-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full z-10">
                  AI DRIVEN
                </div>
                <DiagnosisReport prediction={prediction} />
              </div>

              {/* Guidelines Comparison & Structural Analysis */}
              <Card className="border-blue-100 bg-white shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2 text-blue-600">
                    <Scale className="w-4 h-4" /> Standard Guidelines Check
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* SOKOLOW */}
                  <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <div>
                      <p className="text-xs font-bold text-slate-500">
                        Sokolow-Lyon Index
                      </p>
                      <p className="text-[10px] text-slate-400">
                        For LVH Detection (SV1 + RV5/6)
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-mono font-bold text-slate-700">
                        {sokolow?.value} mV
                      </p>
                      <Badge
                        variant={
                          sokolow?.isPositive ? "destructive" : "outline"
                        }
                        className="text-[10px]"
                      >
                        {sokolow?.isPositive
                          ? "POSITIVE (>3.5)"
                          : "NEGATIVE (<3.5)"}
                      </Badge>
                    </div>
                  </div>

                  {/* DEEP MORPHOLOGY (New) */}
                  {prediction?.clinical_features && (
                    <div className="space-y-2">
                      <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mt-4">
                        Morphology (Lead II)
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-slate-50 rounded border border-slate-200">
                          <span className="text-[10px] text-slate-400 block">
                            Heart Rate
                          </span>
                          <span className="text-sm font-mono font-bold text-emerald-600">
                            {Math.round(
                              prediction.clinical_features.heart_rate
                            )}{" "}
                            <span className="text-[10px] text-slate-500">
                              BPM
                            </span>
                          </span>
                        </div>
                        <div className="p-2 bg-slate-50 rounded border border-slate-200">
                          <span className="text-[10px] text-slate-400 block">
                            ST Segment
                          </span>
                          {prediction.clinical_features.st_findings?.some(
                            (f) => f.status !== "Normal"
                          ) ? (
                            <span className="text-xs font-bold text-rose-400 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" /> Abnormal
                            </span>
                          ) : (
                            <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                              <CheckCircle className="w-3 h-3" /> Normal
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Wave Intervals */}
                      <div className="p-3 bg-slate-50 rounded border border-slate-200 text-[10px] font-mono text-slate-500 space-y-1">
                        <div className="flex justify-between">
                          <span>R-R Interval (Avg)</span>
                          <span className="text-slate-700">
                            {prediction.clinical_features.r_peaks.length > 1
                              ? (
                                  (prediction.clinical_features.r_peaks[1] -
                                    prediction.clinical_features.r_peaks[0]) *
                                  2
                                ).toFixed(0)
                              : "--"}{" "}
                            ms
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="p-3 bg-indigo-900/20 border border-indigo-500/20 rounded-lg text-xs text-indigo-200">
                    <strong>Educational Note:</strong>
                    <br />
                    Deep Learning models (like Mamba) can detect subtle
                    non-linear features that standard manual measurements (like
                    Sokolow-Lyon) might miss, especially in early-stage
                    pathologies.
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {!revealed && (
            <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50 p-8 text-slate-400">
              <div className="text-center">
                <BookOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>
                  Make a diagnosis to reveal <br /> AI & Guideline analysis.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* DEEP TUTOR CHAT INTEGRATION */}
      <TutorChat
        context={prediction}
        minimized={chatMinimized}
        setMinimized={setChatMinimized}
        onAction={(action) => console.log("Tutor Action:", action)}
      />
    </div>
  );
};

export default EducationMode;
