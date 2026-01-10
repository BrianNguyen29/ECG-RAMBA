import React, { useState, useEffect } from 'react';
import { Trash2, User, Calendar, Activity, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

const History = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const data = JSON.parse(localStorage.getItem('ecg_history') || '[]');
    setHistory(data);
  }, []);

  const clearHistory = () => {
    if (window.confirm("Are you sure you want to delete all history?")) {
      localStorage.removeItem('ecg_history');
      setHistory([]);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
       <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-800 tracking-tight">Patient History</h1>
            <p className="text-gray-500 text-sm">Recent analysis records stored locally</p>
          </div>
          {history.length > 0 && (
            <button 
              onClick={clearHistory}
              className="flex items-center gap-2 px-4 py-2 text-sm text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors font-medium"
            >
               <Trash2 className="w-4 h-4" />
               Clear History
            </button>
          )}
       </div>

       {history.length === 0 ? (
         <div className="text-center py-20 bg-white rounded-2xl shadow-sm border border-gray-100">
            <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-gray-300" />
            </div>
            <h3 className="text-lg font-medium text-gray-900">No Records Found</h3>
            <p className="text-gray-500 text-sm mt-1">Run an analysis to see it appear here.</p>
         </div>
       ) : (
         <div className="space-y-4">
            <AnimatePresence>
              {history.map((record, index) => (
                 <motion.div 
                    key={record.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="group bg-white p-5 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-all flex flex-col md:flex-row md:items-center justify-between gap-4"
                 >
                    <div className="flex items-start gap-4">
                        <div className={clsx(
                           "w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 font-bold text-lg",
                           record.diagnosis.toLowerCase().includes("normal") ? "bg-emerald-100 text-emerald-600" : "bg-amber-100 text-amber-600"
                        )}>
                            {record.diagnosis.charAt(0)}
                        </div>
                        <div>
                           <h3 className="font-bold text-gray-800">{record.patient.name || "Unknown Patient"}</h3>
                           <div className="flex items-center gap-4 text-xs text-gray-500 mt-1">
                               <span className="flex items-center gap-1">
                                  <User className="w-3 h-3" /> {record.patient.id || "N/A"}
                               </span>
                               <span className="flex items-center gap-1">
                                  <Calendar className="w-3 h-3" /> {new Date(record.timestamp).toLocaleString()}
                               </span>
                           </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-8 pl-16 md:pl-0 border-t md:border-0 pt-4 md:pt-0">
                         <div>
                            <p className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-1">Diagnosis</p>
                            <p className={clsx(
                                "font-bold text-sm",
                                record.diagnosis.toLowerCase().includes("normal") ? "text-emerald-600" : "text-amber-600"
                            )}>{record.diagnosis}</p>
                         </div>
                         <div>
                            <p className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-1">Confidence</p>
                            <p className="font-bold text-sm text-gray-700">{(record.confidence * 100).toFixed(1)}%</p>
                         </div>
                         <div className="hidden lg:block">
                            <p className="text-xs uppercase tracking-wider text-gray-400 font-semibold mb-1">Model</p>
                            <p className="font-medium text-sm text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{record.model}</p>
                         </div>
                    </div>
                 </motion.div>
              ))}
            </AnimatePresence>
         </div>
       )}
    </div>
  );
};

export default History;
