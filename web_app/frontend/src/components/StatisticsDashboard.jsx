import React from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar
} from 'recharts';
import { Activity, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

/**
 * Statistics Dashboard Component
 * Displays comprehensive analysis results with healthcare-grade visualizations
 */
function StatisticsDashboard({ prediction }) {
  if (!prediction) return null;

  const { predictions = [], all_probabilities = {}, top_diagnosis, confidence } = prediction;

  // Prepare data for Pie Chart (top predictions)
  const pieData = predictions.slice(0, 5).map(([label, prob], idx) => ({
    name: label,
    value: Math.round(prob * 100),
    color: getPieColor(idx)
  }));

  // Prepare data for Bar Chart (all probabilities)
  const barData = Object.entries(all_probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 12)
    .map(([label, prob]) => ({
      class: label,
      probability: Math.round(prob * 100),
      fill: prob >= 0.5 ? '#10B981' : prob >= 0.3 ? '#F59E0B' : '#6B7280'
    }));

  // Risk assessment based on predictions
  const riskLevel = getRiskLevel(predictions);

  return (
    <div className="space-y-6">
      {/* Risk Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <RiskCard
          title="Primary Diagnosis"
          value={top_diagnosis}
          subtitle={`${(confidence * 100).toFixed(1)}% confidence`}
          icon={Activity}
          color="blue"
        />
        <RiskCard
          title="Detected Conditions"
          value={predictions.length}
          subtitle={predictions.length > 1 ? 'Multi-label detection' : 'Single condition'}
          icon={TrendingUp}
          color="purple"
        />
        <RiskCard
          title="Risk Assessment"
          value={riskLevel.level}
          subtitle={riskLevel.message}
          icon={riskLevel.level === 'Normal' ? CheckCircle : AlertTriangle}
          color={riskLevel.color}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart - Top Predictions */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center">
              <Activity className="w-4 h-4 text-blue-600" />
            </div>
            Prediction Distribution
          </h3>
          
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value) => `${value}%`}
                  contentStyle={{ 
                    borderRadius: '12px', 
                    border: 'none',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                  }}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={36}
                  formatter={(value) => <span className="text-sm text-gray-600">{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-400">
              No predictions available
            </div>
          )}
        </div>

        {/* Bar Chart - All Probabilities */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-purple-100 flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-purple-600" />
            </div>
            Class Probabilities
          </h3>
          
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={barData} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  type="number" 
                  domain={[0, 100]} 
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fill: '#6B7280', fontSize: 12 }}
                />
                <YAxis 
                  type="category" 
                  dataKey="class" 
                  tick={{ fill: '#374151', fontSize: 11 }}
                  width={55}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, 'Probability']}
                  contentStyle={{ 
                    borderRadius: '12px', 
                    border: 'none',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                  }}
                />
                <Bar 
                  dataKey="probability" 
                  radius={[0, 6, 6, 0]}
                  fill="#6366F1"
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-400">
              No probability data available
            </div>
          )}
        </div>
      </div>

      {/* Confidence Gauge */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-100">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Analysis Confidence</h3>
            <p className="text-sm text-gray-500 mt-1">
              Model certainty for the primary diagnosis
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative w-24 h-24">
              <svg className="w-24 h-24 transform -rotate-90">
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke="#E5E7EB"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke={confidence >= 0.7 ? '#10B981' : confidence >= 0.5 ? '#F59E0B' : '#EF4444'}
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={`${confidence * 251.2} 251.2`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-gray-800">
                  {Math.round(confidence * 100)}%
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-gray-800">{top_diagnosis}</p>
              <p className="text-sm text-gray-500">Primary Finding</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper Components
function RiskCard({ title, value, subtitle, icon: Icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    purple: 'bg-purple-50 border-purple-200 text-purple-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    yellow: 'bg-yellow-50 border-yellow-200 text-yellow-700',
    red: 'bg-red-50 border-red-200 text-red-700'
  };

  const iconBg = {
    blue: 'bg-blue-100 text-blue-600',
    purple: 'bg-purple-100 text-purple-600',
    green: 'bg-green-100 text-green-600',
    yellow: 'bg-yellow-100 text-yellow-600',
    red: 'bg-red-100 text-red-600'
  };

  return (
    <div className={`rounded-xl border-2 p-5 ${colorClasses[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium opacity-80">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          <p className="text-xs opacity-70 mt-1">{subtitle}</p>
        </div>
        <div className={`p-2 rounded-lg ${iconBg[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}

// Helper functions
function getPieColor(index) {
  const colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#6366F1'];
  return colors[index % colors.length];
}

function getRiskLevel(predictions) {
  const criticalConditions = ['MI', 'VT', 'VF', 'Myocardial'];
  const warningConditions = ['AF', 'AFL', 'LBBB', 'RBBB', 'Brady', 'Tachy'];
  const normalConditions = ['SNR', 'Normal', 'SR'];

  const labels = predictions.map(([l]) => l);

  if (labels.some(l => criticalConditions.some(c => l.includes(c)))) {
    return { level: 'Critical', color: 'red', message: 'Immediate medical attention recommended' };
  }
  if (labels.some(l => warningConditions.some(c => l.includes(c)))) {
    return { level: 'Attention', color: 'yellow', message: 'Follow-up with specialist advised' };
  }
  if (labels.some(l => normalConditions.some(c => l.includes(c)))) {
    return { level: 'Normal', color: 'green', message: 'No immediate concerns detected' };
  }
  return { level: 'Review', color: 'blue', message: 'Clinical correlation recommended' };
}

export default StatisticsDashboard;
