import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Story from './pages/Story';
import History from './pages/History';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="history" element={<History />} />
            <Route path="story" element={<Story />} />
            <Route path="settings" element={
              <div className="p-8 text-center">
                <h2 className="text-2xl font-bold text-gray-700 mb-4">Settings</h2>
                <p className="text-gray-500">Configuration options coming soon.</p>
              </div>
            } />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
