import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, BookOpen, Settings, Heart, Menu, Activity, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

const SidebarItem = ({ to, icon: Icon, label }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <NavLink to={to} className="relative block">
      {isActive && (
        <motion.div
           layoutId="active-pill"
           className="absolute inset-0 bg-gradient-to-r from-blue-100/50 to-indigo-100/50 rounded-xl"
           initial={{ opacity: 0 }}
           animate={{ opacity: 1 }}
           exit={{ opacity: 0 }}
        />
      )}
      <div className={clsx(
        "relative flex items-center gap-3 px-4 py-3 rounded-xl transition-colors duration-200 z-10",
        isActive ? "text-blue-700 font-medium" : "text-gray-600 hover:bg-white/50 hover:text-gray-900"
      )}>
        <Icon className={clsx("w-5 h-5", isActive ? "text-blue-600" : "text-gray-400")} />
        <span>{label}</span>
      </div>
    </NavLink>
  );
};

const DashboardLayout = () => {
  return (
    <div className="flex h-screen bg-[#F3F4F6] overflow-hidden font-sans selection:bg-blue-200 selection:text-blue-900">
      
      {/* Dynamic Background Accents */}
      <div className="fixed inset-0 z-0 pointer-events-none">
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-400/20 rounded-full blur-[100px]" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-400/20 rounded-full blur-[100px]" />
      </div>

      {/* Glass Sidebar */}
      <aside className="w-72 hidden md:flex flex-col z-10 m-4 rounded-3xl bg-white/70 backdrop-blur-xl border border-white/50 shadow-xl shadow-blue-500/5">
        <div className="p-8 pb-4 flex items-center gap-3">
           <motion.div 
             initial={{ rotate: -10, scale: 0.9 }}
             animate={{ rotate: 0, scale: 1 }}
             transition={{ duration: 0.5, ease: "backOut" }}
             className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/30"
           >
              <Activity className="w-6 h-6 text-white" />
           </motion.div>
           <div>
             <h1 className="text-xl font-bold text-slate-800 tracking-tight">CardioAI</h1>
             <p className="text-xs text-slate-500 font-medium tracking-wide">CLINICAL SUITE</p>
           </div>
        </div>

        <nav className="flex-1 px-4 space-y-2 py-6 overflow-y-auto">
          <div className="px-4 mb-2 text-xs font-bold text-slate-400/80 uppercase tracking-widest">Platform</div>
          <SidebarItem to="/" icon={LayoutDashboard} label="Dashboard" />
          <SidebarItem to="/history" icon={Activity} label="Patient History" />
          <SidebarItem to="/story" icon={BookOpen} label="Project Story" />
          
          <div className="px-4 mb-2 mt-8 text-xs font-bold text-slate-400/80 uppercase tracking-widest">System</div>
          <SidebarItem to="/settings" icon={Settings} label="Configuration" />
        </nav>

        <div className="p-4 mt-auto">
          <div className="bg-gradient-to-br from-white/60 to-white/30 backdrop-blur-md rounded-2xl p-4 border border-white/60 shadow-sm relative overflow-hidden group">
             <div className="absolute top-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Sparkles className="w-4 h-4 text-amber-400" />
             </div>
             <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold shadow-inner">
                  DC
                </div>
                <div>
                   <p className="text-sm font-bold text-gray-800">Dr. Cardio</p>
                   <p className="text-xs text-blue-600 font-medium">Cardiology Head</p>
                </div>
             </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden relative z-10">
        <header className="h-20 flex items-center justify-between px-8 md:px-12">
           <div className="md:hidden flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg text-white">
                 <Activity className="w-5 h-5" />
              </div>
              <span className="font-bold text-gray-800 text-lg">CardioAI</span>
           </div>
           
           <div className="hidden md:flex items-center gap-2 text-gray-400 text-sm font-medium">
              <span>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</span>
           </div>

           <button className="md:hidden p-2 text-gray-600 bg-white rounded-lg shadow-sm">
              <Menu className="w-6 h-6" />
           </button>
        </header>

        <main className="flex-1 overflow-y-auto px-4 pb-4 md:px-8 md:pb-8">
           <AnimatePresence mode="wait">
             <motion.div 
               key={useLocation().pathname}
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               exit={{ opacity: 0, y: -10 }}
               transition={{ duration: 0.3 }}
               className="h-full"
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
