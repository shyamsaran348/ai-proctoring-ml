import React, { useEffect, useState } from 'react';
import { ShieldCheck, OctagonAlert, ShieldAlert } from 'lucide-react';

export default function SafeZoneHud({ riskScore, violationType, connected }) {
  const [hudState, setHudState] = useState({ class: 'risk-safe', text: 'Safe Zone Active', Icon: ShieldCheck });
  
  // A secondary piece of state that tracks live unacknowledged faculty warnings
  // In a full build, this would pop from SSE or a Django Poller
  const [facultyWarning, setFacultyWarning] = useState(null);

  useEffect(() => {
    if (!connected) {
       setHudState({ class: 'bg-gray-100 text-gray-500 border-gray-200', text: 'Camera Offline', Icon: ShieldAlert });
       return;
    }
    
    // Risk Score logic from legacy index.html
    // Safe (<0.3), Warning (0.3 - 0.7), Danger (>0.7)
    if (riskScore >= 0.7) {
      setHudState({ class: 'risk-danger', text: violationType || 'Critical Anomaly Detected', Icon: OctagonAlert });
    } else if (riskScore >= 0.3) {
      setHudState({ class: 'risk-warning', text: 'Warning: Boundary Shift', Icon: ShieldAlert });
    } else {
      setHudState({ class: 'risk-safe', text: 'Safe Zone Active', Icon: ShieldCheck });
    }
  }, [riskScore, violationType, connected]);

  const { class: hudClass, text, Icon } = hudState;

  return (
    <div className="flex flex-col gap-4 w-full">
       <div className={`flex items-center justify-between px-6 py-5 rounded-3xl border-2 ${hudClass} transition-all duration-500 shadow-2xl relative overflow-hidden backdrop-blur-sm`}>
          {/* Neural Background Pulse (Phase 19) */}
          <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-20 group-hover:opacity-40 transition-opacity"></div>
          
          <div className="flex items-center space-x-5 z-10">
             <div className="p-3 bg-white/20 rounded-2xl shadow-inner">
                <Icon size={32} className={riskScore >= 0.7 ? 'animate-bounce text-white' : ''} />
             </div>
             <div>
                <h3 className="font-black text-xl leading-tight uppercase tracking-tighter italic">{text}</h3>
                <div className="flex items-center gap-3 mt-1.5 min-w-[200px]">
                   <p className="text-[10px] font-black uppercase tracking-[0.2em] opacity-80 border-r border-current/20 pr-3">
                      {connected ? `Sentinel Fusion: ${(riskScore * 100).toFixed(1)}%` : 'Link Initializing...'}
                   </p>
                   <div className="flex items-center gap-1 opacity-70">
                      <div className={`w-1 h-1 rounded-full ${connected ? 'bg-current animate-pulse' : 'bg-slate-400'}`}></div>
                      <span className="text-[8px] font-black uppercase tracking-widest">7-Signal Array</span>
                   </div>
                </div>
             </div>
          </div>
          
          {/* Status Dots */}
          <div className="flex space-x-1.5 opacity-90">
             <div className={`w-2.5 h-2.5 rounded-full ${riskScore < 0.3 && connected ? 'bg-current' : 'bg-current opacity-30 shadow-sm'}`}></div>
             <div className={`w-2.5 h-2.5 rounded-full ${riskScore >= 0.3 && riskScore < 0.7 && connected ? 'bg-current animate-pulse' : 'bg-current opacity-30'}`}></div>
             <div className={`w-2.5 h-2.5 rounded-full ${riskScore >= 0.7 && connected ? 'bg-current animate-pulse shadow-[0_0_10px_currentColor]' : 'bg-current opacity-30'}`}></div>
          </div>
       </div>

       {/* Faculty Warning Flash Message - Mirroring legacy "intervention-alert" logic */}
       {facultyWarning && (
         <div className="bg-red-600 animate-pulse rounded-xl p-4 text-white font-bold flex flex-col shadow-2xl relative overflow-hidden">
             <div className="absolute top-0 right-0 p-2 opacity-50"><OctagonAlert size={48} /></div>
             <span className="uppercase text-red-200 text-xs tracking-widest mb-1 z-10">Live Faculty Intervention</span>
             <span className="text-lg z-10 mb-4">{facultyWarning}</span>
             <button 
                onClick={() => setFacultyWarning(null)}
                className="bg-white/20 hover:bg-white/30 text-white rounded-lg px-4 py-2 mt-auto w-fit transition-colors z-10 font-bold tracking-wide shadow-md"
             >
                I Understand - Resume Exam
             </button>
         </div>
       )}
    </div>
  );
}
