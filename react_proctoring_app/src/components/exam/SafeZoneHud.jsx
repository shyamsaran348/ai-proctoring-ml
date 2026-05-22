import React, { useEffect, useState, useRef } from 'react';
import { ShieldCheck, OctagonAlert, ShieldAlert, Zap } from 'lucide-react';

export default function SafeZoneHud({ riskScore, violationType, connected }) {
  const [hudState, setHudState] = useState({ class: 'risk-safe', text: 'Safe Zone Active', Icon: ShieldCheck });
  const [hint, setHint] = useState('');
  const prevRisk = useRef(riskScore);
  const [isRecovering, setIsRecovering] = useState(false);
  
  // A secondary piece of state that tracks live unacknowledged faculty warnings
  const [facultyWarning, setFacultyWarning] = useState(null);

  useEffect(() => {
    if (!connected) {
       setHudState({ class: 'bg-gray-100 text-gray-500 border-gray-200', text: 'Camera Offline', Icon: ShieldAlert });
       setHint('Sentinel Link Disconnected. Please check your camera.');
       return;
    }
    
    // Recovery Detection (Risk trending down)
    if (riskScore < prevRisk.current && riskScore > 0.1) {
       setIsRecovering(true);
       setTimeout(() => setIsRecovering(false), 1000);
    }
    prevRisk.current = riskScore;

    // Hint Logic based on detailed violation types
    let currentHint = '';
    if (riskScore >= 0.3) {
        switch(violationType) {
            case 'FACE_NOT_DETECTED': currentHint = 'Ensure your full face is visible to the camera.'; break;
            case 'IDENTITY_MISMATCH': currentHint = 'Face identity does not match reference. Remove masks/glasses.'; break;
            case 'LOOKING_AWAY': currentHint = 'Please look directly at the exam screen.'; break;
            case 'OFFSCREEN_GAZE': currentHint = 'Multiple eye-tracking deviations detected. Focus on the screen.'; break;
            case 'BEYOND_SCREEN_BOUNDARY': currentHint = 'You are moving outside the safe zone. Please re-center.'; break;
            case 'MULTIPLE_FACES_DETECTED': currentHint = 'Ensure you are the only person in the camera frame.'; break;
            case 'SOPHISTICATED_AUDIO_ANOMALY': currentHint = 'High background noise detected. Maintain silence.'; break;
            default: currentHint = 'System detecting unusual patterns. Please stay focused.';
        }
    }
    setHint(currentHint);

    if (riskScore >= 0.7) {
      setHudState({ class: 'risk-danger', text: violationType?.replace(/_/g, ' ') || 'Critical Anomaly', Icon: OctagonAlert });
    } else if (riskScore >= 0.3) {
      setHudState({ class: 'risk-warning', text: 'Warning: Policy Deviation', Icon: ShieldAlert });
    } else {
      setHudState({ class: 'risk-safe', text: 'Safe Zone Active', Icon: ShieldCheck });
    }
  }, [riskScore, violationType, connected]);

  const { class: hudClass, text, Icon } = hudState;

  return (
    <div className="flex flex-col gap-4 w-full">
        <div className={`flex items-center justify-between px-6 py-5 rounded-3xl border-2 ${hudClass} transition-all duration-500 shadow-2xl relative overflow-hidden backdrop-blur-sm group`}>
           {/* Neural Background Pulse (Phase 19) */}
           <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-20 group-hover:opacity-40 transition-opacity"></div>
           
           {/* Recovery Pulse (Phase 24) */}
           {isRecovering && (
              <div className="absolute inset-0 bg-green-400/10 animate-pulse pointer-events-none"></div>
           )}

           <div className="flex items-center space-x-5 z-10 w-full">
              <div className="relative">
                <div className={`p-3 bg-white/20 rounded-2xl shadow-inner transition-transform duration-500 ${riskScore >= 0.7 ? 'scale-110' : ''}`}>
                    <Icon size={32} className={riskScore >= 0.7 ? 'animate-bounce text-white' : ''} />
                </div>
                {isRecovering && (
                   <div className="absolute -top-2 -right-2 bg-green-500 text-white p-1 rounded-full animate-bounce shadow-lg">
                      <Zap size={12} fill="currentColor" />
                   </div>
                )}
              </div>

              <div className="flex-1">
                 <h3 className="font-black text-xl leading-tight uppercase tracking-tighter italic">{text}</h3>
                 
                 {hint && riskScore >= 0.3 ? (
                   <p className="text-xs font-bold mt-1 text-current opacity-90 animate-pulse flex items-center gap-2">
                     <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                     {hint}
                   </p>
                 ) : (
                   <div className="flex items-center gap-3 mt-1.5 min-w-[200px]">
                      <p className="text-[10px] font-black uppercase tracking-[0.2em] opacity-80 border-r border-current/20 pr-3">
                         {connected ? `Sentinel Fusion: ${(riskScore * 100).toFixed(1)}%` : 'Link Initializing...'}
                      </p>
                      <div className="flex items-center gap-1 opacity-70">
                         <div className={`w-1 h-1 rounded-full ${connected ? 'bg-current animate-pulse' : 'bg-slate-400'}`}></div>
                         <span className="text-[8px] font-black uppercase tracking-widest">7-Signal Array</span>
                      </div>
                   </div>
                 )}
              </div>

              {/* Status Dots */}
              <div className="flex space-x-1.5 opacity-90 ml-4 group-hover:scale-110 transition-transform">
                 <div className={`w-2 h-6 rounded-full transition-all duration-500 ${riskScore < 0.3 && connected ? 'bg-current h-8' : 'bg-current opacity-30 shadow-sm'}`}></div>
                 <div className={`w-2 h-6 rounded-full transition-all duration-500 ${riskScore >= 0.3 && riskScore < 0.7 && connected ? 'bg-current h-8 animate-pulse' : 'bg-current opacity-30'}`}></div>
                 <div className={`w-2 h-6 rounded-full transition-all duration-500 ${riskScore >= 0.7 && connected ? 'bg-current h-8 animate-pulse shadow-[0_0_10px_currentColor]' : 'bg-current opacity-30'}`}></div>
              </div>
           </div>
        </div>

        {/* Faculty Warning Flash Message */}
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
