import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, BookOpen, Settings, Activity, Sparkles, Menu } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

const SidebarItem = ({ to, icon: Icon, label }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <NavLink to={to} className="relative block group">
      {isActive && (
        <motion.div
           layoutId="active-pill"
           className="absolute inset-0 bg-cyan-500/10 border-l-2 border-cyan-500"
           initial={{ opacity: 0 }}
           animate={{ opacity: 1 }}
           exit={{ opacity: 0 }}
        />
      )}
      <div className={clsx(
        "relative flex items-center gap-3 px-4 py-3 transition-colors duration-200 z-10",
        isActive ? "text-cyan-400 font-medium" : "text-slate-400 group-hover:text-slate-200"
      )}>
        <Icon className={clsx("w-5 h-5", isActive ? "text-cyan-400" : "text-slate-500 group-hover:text-slate-300")} />
        <span className="text-sm tracking-wide">{label}</span>
      </div>
    </NavLink>
  );
};

const DashboardLayout = () => {
  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden font-sans selection:bg-blue-100 selection:text-blue-900">
      
      {/* Dynamic Background Grid */}
      <div className="fixed inset-0 z-0 pointer-events-none opacity-100 ecg-grid-light" />

      {/* Medical Sidebar */}
      <aside className="w-48 hidden md:flex flex-col z-10 border-r border-slate-200 bg-white/50 backdrop-blur-xl">
         <div className="p-4 flex items-center gap-2 border-b border-slate-100">
           <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-lg shadow-lg shadow-blue-500/20">
              <Activity className="w-5 h-5 text-white" />
           </div>
           <div>
             <h1 className="text-base font-extrabold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">ECG-RAMBA</h1>
             <p className="text-[9px] text-blue-500 font-bold uppercase tracking-widest">Clinical Suite</p>
           </div>
        </div>

        <nav className="flex-1 px-2 space-y-1 py-6 overflow-y-auto">
          <div className="px-4 mb-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Module</div>
          <SidebarItem to="/" icon={LayoutDashboard} label="Cockpit" />
          <SidebarItem to="/history" icon={BookOpen} label="Patient History" />
          <SidebarItem to="/story" icon={Sparkles} label="Architecture" />
          
          <div className="px-4 mb-2 mt-8 text-[10px] font-bold text-slate-500 uppercase tracking-widest">System</div>
          <SidebarItem to="/settings" icon={Settings} label="Calibration" />
        </nav>

        <div className="p-4 mt-auto border-t border-slate-800/50">
          <div className="flex items-center gap-3 px-2">
             <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-cyan-500 font-bold text-xs ring-1 ring-slate-700">
               DR
             </div>
             <div>
                <p className="text-xs font-bold text-slate-200">Dr. Cardio</p>
                <p className="text-[10px] text-slate-500">Cardiology Dept.</p>
             </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden relative z-10">
        {/* Mobile Header */}
        <header className="h-16 md:hidden flex items-center justify-between px-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur-lg">
           <div className="flex items-center gap-3">
              <Activity className="w-5 h-5 text-cyan-500" />
              <span className="font-bold text-slate-100">ECG-RAMBA</span>
           </div>
           <button className="p-2 text-slate-400">
              <Menu className="w-5 h-5" />
           </button>
        </header>

        <main className="flex-1 overflow-hidden relative">
           <AnimatePresence mode="wait">
             <motion.div 
               key={useLocation().pathname}
               initial={{ opacity: 0 }}
               animate={{ opacity: 1 }}
               exit={{ opacity: 0 }}
               transition={{ duration: 0.2 }}
               className="h-full w-full"
             >
                <Outlet />
             </motion.div>
           </AnimatePresence>
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
