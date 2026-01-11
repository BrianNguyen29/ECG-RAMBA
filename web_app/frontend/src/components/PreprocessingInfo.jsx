import React from 'react';
import { AlertTriangle, CheckCircle, Info, Zap, Activity } from 'lucide-react';
import clsx from 'clsx';

/**
 * PreprocessingInfo Component
 * Displays preprocessing steps and lead dropout warnings
 */
function PreprocessingInfo({ uploadResult }) {
  if (!uploadResult) return null;

  const { 
    preprocessing = [], 
    warnings = [], 
    accuracy_notes = [],
    mapped_leads = [],
    missing_leads = [],
    num_leads,
    samples,
    duration_s,
    sample_rate
  } = uploadResult;

  const hasMissingLeads = missing_leads && missing_leads.length > 0;
  const hasWarnings = warnings && warnings.length > 0;
  const hasAccuracyNotes = accuracy_notes && accuracy_notes.length > 0;

  return (
    <div className="space-y-4">
      {/* Signal Info Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
            <Activity className="w-4 h-4 text-blue-600" />
          </div>
          <h4 className="font-semibold text-gray-800">Thông tin tín hiệu</h4>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          <InfoChip label="Leads" value={`${num_leads || 12}/12`} />
          <InfoChip label="Samples" value={samples?.toLocaleString() || '5000'} />
          <InfoChip label="Thời gian" value={`${duration_s?.toFixed(1) || '10'}s`} />
          <InfoChip label="Sample rate" value={`${sample_rate || 500} Hz`} />
        </div>
      </div>

      {/* Lead Mapping */}
      {hasMissingLeads && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-yellow-800 mb-2">
                Thiếu {missing_leads.length}/12 chuyển đạo
              </h4>
              <div className="flex flex-wrap gap-1 mb-2">
                {['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].map(lead => (
                  <span
                    key={lead}
                    className={clsx(
                      "px-2 py-0.5 rounded text-xs font-medium",
                      mapped_leads?.includes(lead) 
                        ? "bg-green-100 text-green-700"
                        : "bg-red-100 text-red-600"
                    )}
                  >
                    {lead}
                  </span>
                ))}
              </div>
              <p className="text-xs text-yellow-700">
                Các leads thiếu được zero-padded theo chiến lược CPSC 2021 Zero-shot
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Accuracy Notes (Lead Dropout Warnings) */}
      {hasAccuracyNotes && (
        <div className="space-y-2">
          {accuracy_notes.map((note, idx) => (
            <AccuracyNote key={idx} note={note} />
          ))}
        </div>
      )}

      {/* Preprocessing Steps */}
      {preprocessing && preprocessing.length > 0 && (
        <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-purple-600" />
            <h4 className="font-semibold text-gray-700 text-sm">Preprocessing Pipeline</h4>
          </div>
          <ol className="space-y-1 text-xs text-gray-600">
            {preprocessing.map((step, idx) => (
              <li key={idx} className="flex items-center gap-2">
                <span className="w-5 h-5 bg-purple-100 text-purple-700 rounded-full flex items-center justify-center text-xs font-bold">
                  {idx + 1}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* General Warnings */}
      {hasWarnings && (
        <div className="bg-orange-50 border border-orange-200 rounded-xl p-3">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-orange-600 mt-0.5 flex-shrink-0" />
            <ul className="text-xs text-orange-700 space-y-1">
              {warnings.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function InfoChip({ label, value }) {
  return (
    <div className="bg-white/80 rounded-lg px-3 py-2">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="font-semibold text-gray-800">{value}</p>
    </div>
  );
}

function AccuracyNote({ note }) {
  const isWarning = note.includes('⚠️') || note.includes('GIẢM') || note.includes('giảm');
  const isOk = note.includes('✓') || note.includes('tốt');
  const isInfo = note.includes('ℹ️') || note.includes('Dữ liệu');

  return (
    <div className={clsx(
      "rounded-xl p-3 flex items-start gap-2 text-sm",
      isWarning ? "bg-red-50 border border-red-200" :
      isOk ? "bg-green-50 border border-green-200" :
      "bg-blue-50 border border-blue-200"
    )}>
      {isWarning ? (
        <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
      ) : isOk ? (
        <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
      ) : (
        <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
      )}
      <p className={clsx(
        isWarning ? "text-red-700" :
        isOk ? "text-green-700" :
        "text-blue-700"
      )}>
        {note.replace('⚠️', '').replace('✓', '').replace('ℹ️', '').trim()}
      </p>
    </div>
  );
}

export default PreprocessingInfo;
