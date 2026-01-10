import React from 'react';
import { Heart, ShieldCheck, Zap, Users } from 'lucide-react';

const FeatureCard = ({ icon: Icon, title, desc }) => (
  <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all">
    <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mb-4 text-blue-600">
      <Icon className="w-6 h-6" />
    </div>
    <h3 className="text-lg font-bold text-gray-900 mb-2">{title}</h3>
    <p className="text-gray-600 leading-relaxed text-sm">{desc}</p>
  </div>
);

const Story = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-12 pb-12">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">
          Revolutionizing <span className="text-blue-600">Cardiac Care</span> with AI
        </h1>
        <p className="text-xl text-gray-500 max-w-2xl mx-auto">
          Our mission is to empower cardiologists with state-of-the-art Deep Learning tools for rapid, accurate ECG diagnosis.
        </p>
      </div>

      {/* The Challenge */}
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">The Challenge</h2>
        <p className="text-gray-600 leading-relaxed mb-6">
          Cardiovascular diseases remain the leading cause of death globally. Early detection through Electrocardiograms (ECGs) is critical, but manual interpretation is time-consuming and prone to human fatigue.
          Traditional automated systems often lack the nuance to detect complex arrhythmia patterns.
        </p>
        <div className="relative h-64 rounded-xl overflow-hidden bg-gray-100">
           {/* Placeholder for an image - using CSS gradient to simulate */}
           <div className="absolute inset-0 bg-gradient-to-r from-blue-900 to-indigo-900 opacity-90 flex items-center justify-center">
              <span className="text-white opacity-20 text-6xl font-black">ECG DATA</span>
           </div>
        </div>
      </div>

      {/* The Solution */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">Our Solution: CardioAI</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FeatureCard 
             icon={Zap} 
             title="Real-time Inference" 
             desc="Powered by advanced PyTorch models, our system processes signals in milliseconds, providing instant feedback during patient consultations."
          />
          <FeatureCard 
             icon={ShieldCheck} 
             title="Clinical Grade Accuracy" 
             desc="Trained on thousands of validated medical records, ensuring high sensitivity and specificity for critical conditions like Atrial Fibrillation."
          />
          <FeatureCard 
             icon={Heart} 
             title="Patient Centric" 
             desc="Designed to integrate seamlessly into existing hospital workflows, focusing on patient outcomes rather than complex technology."
          />
           <FeatureCard 
             icon={Users} 
             title="Assistive AI" 
             desc="We believe in AI as a partner, not a replacement. Our tools provide 'Second Opinions' to support doctor decision-making."
          />
        </div>
      </div>

      {/* Footer Quote */}
      <div className="text-center border-t border-gray-200 pt-12">
        <blockquote className="text-2xl font-medium text-gray-900 italic">
          "Technology is best when it brings people together."
        </blockquote>
        <div className="mt-4 text-gray-500">- Matt Mullenweg</div>
      </div>
    </div>
  );
};

export default Story;
