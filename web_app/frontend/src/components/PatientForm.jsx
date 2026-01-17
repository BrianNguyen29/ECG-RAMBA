import React, { useState } from 'react';
import { User, AlignLeft, Calendar } from 'lucide-react';

const PatientForm = ({ onUpdate }) => {
  const [data, setData] = useState({
    name: '',
    id: '',
    age: '',
    gender: 'Select'
  });

  const handleChange = (e) => {
    const newData = { ...data, [e.target.name]: e.target.value };
    setData(newData);
    onUpdate(newData); // Notify parent
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-800">
        <User className="w-5 h-5 text-blue-500" />
        Patient Information
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Name */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">Full Name</label>
          <div className="relative">
            <input 
              type="text"
              name="name"
              value={data.name}
              onChange={handleChange}
              placeholder="Ex: John Doe"
              className="w-full pl-3 pr-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus-visible:outline-none transition-all text-sm"
            />
          </div>
        </div>

        {/* Patient ID */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">Patient ID</label>
          <div className="relative">
            <AlignLeft className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
            <input 
              type="text"
              name="id"
              value={data.id}
              onChange={handleChange}
              placeholder="Ex: P-12345"
              className="w-full pl-9 pr-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus-visible:outline-none transition-all text-sm"
            />
          </div>
        </div>

         {/* Age */}
         <div>
          <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">Age</label>
          <div className="relative">
             <input 
              type="number"
              name="age"
              value={data.age}
              onChange={handleChange}
              placeholder="Ex: 45"
              className="w-full pl-3 pr-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus-visible:outline-none transition-all text-sm"
            />
          </div>
        </div>

        {/* Gender */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1 uppercase">Gender</label>
          <select
            name="gender"
            value={data.gender}
            onChange={handleChange}
            className="w-full pl-3 pr-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus-visible:outline-none transition-all text-sm"
          >
            <option>Select</option>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default PatientForm;
