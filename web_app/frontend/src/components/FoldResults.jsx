import React from 'react';
import { CheckCircle, XCircle, Activity, TrendingUp } from 'lucide-react';
import clsx from 'clsx';

/**
 * FoldResults Component
 * Displays individual 5-fold cross-validation results in a grid layout
 */
function FoldResults({ prediction }) {
  if (!prediction || !prediction.fold_results) return null;

  const { fold_results, num_folds, confidence_std } = prediction;

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-gray-100 bg-gradient-to-r from-indigo-50 to-purple-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-100 rounded-xl flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-800">
                5-Fold Cross-Validation Results
              </h3>
              <p className="text-xs text-gray-500">
                Ensemble of {num_folds || fold_results.length} models • 
                Std: ±{((confidence_std || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Agreement:</span>
            <div className="flex">
              {fold_results.map((fold, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    "w-3 h-3 rounded-full -ml-1 first:ml-0 border-2 border-white",
                    fold.status === 'success' ? "bg-green-500" : "bg-red-400"
                  )}
                  title={`${fold.fold}: ${fold.top_diagnosis || 'Error'}`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Fold Grid */}
      <div className="p-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {fold_results.map((fold, idx) => (
            <FoldCard key={idx} fold={fold} index={idx} />
          ))}
        </div>
      </div>

      {/* Summary Bar */}
      <div className="px-5 py-3 bg-gray-50 border-t border-gray-100 flex items-center justify-between text-xs text-gray-500">
        <span>Average probability computed from all successful folds</span>
        <span className="flex items-center gap-1">
          <Activity className="w-3 h-3" />
          ECG-RAMBA Ensemble Mode
        </span>
      </div>
    </div>
  );
}

function FoldCard({ fold, index }) {
  const isSuccess = fold.status === 'success';
  
  // Color based on confidence
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'text-green-600 bg-green-50';
    if (conf >= 0.6) return 'text-blue-600 bg-blue-50';
    if (conf >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  };

  return (
    <div className={clsx(
      "rounded-xl border p-4 transition-all hover:shadow-md",
      isSuccess 
        ? "border-gray-200 bg-white" 
        : "border-red-200 bg-red-50"
    )}>
      {/* Fold Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          {fold.fold || `Fold ${index + 1}`}
        </span>
        {isSuccess ? (
          <CheckCircle className="w-4 h-4 text-green-500" />
        ) : (
          <XCircle className="w-4 h-4 text-red-400" />
        )}
      </div>

      {isSuccess ? (
        <>
          {/* Diagnosis */}
          <div className="mb-2">
            <p className="text-sm font-bold text-gray-800 truncate" title={fold.top_diagnosis}>
              {fold.top_diagnosis}
            </p>
          </div>

          {/* Confidence */}
          <div className={clsx(
            "inline-flex items-center px-2 py-1 rounded-lg text-xs font-semibold",
            getConfidenceColor(fold.confidence)
          )}>
            {(fold.confidence * 100).toFixed(1)}%
          </div>

          {/* Mini probability bar */}
          <div className="mt-3 h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all"
              style={{ width: `${fold.confidence * 100}%` }}
            />
          </div>
        </>
      ) : (
        <p className="text-xs text-red-600">
          {fold.error || 'Inference failed'}
        </p>
      )}
    </div>
  );
}

export default FoldResults;
