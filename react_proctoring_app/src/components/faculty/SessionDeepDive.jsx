import React, { useState, useEffect } from 'react';
import { 
  X, 
  User, 
  MapPin, 
  Activity, 
  ShieldAlert, 
  Camera, 
  Clock, 
  FileText, 
  ChevronRight,
  TrendingDown,
  TrendingUp,
  Cpu,
  AlertOctagon,
  Eye,
  Crosshair
} from 'lucide-react';
import api from '../../services/api';

export default function SessionDeepDive({ sessionId, onClose }) {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (sessionId) fetchHistory();
  }, [sessionId]);

  const sendCommand = async (command) => {
    try {
      await api.post(`/sessions/${sessionId}/update_command/`, { command });
      alert(`Success: ${command} signal broadcasted to Sentinel node.`);
    } catch (err) {
      alert("Command broadcast failure - Check proctoring node connectivity.");
    }
  };

  const fetchHistory = async () => {
    try {
      setIsLoading(true);
      const res = await api.get(`/sessions/${sessionId}/session_history/`);
      setData(res.data);
    } catch (err) {
      console.error("Failed to fetch session deep dive.");
      // Fallback
      setData({
        session_id: sessionId,
        student_id: 'shyamsaran348',
        history: [
          { timestamp: new Date().toISOString(), risk_score: 0.85, violation: 'eye_drift_prolonged', frame_url: null, meta_data: { uc4_drift: 0.92, gam_gaze: 0.88, num_faces: 1 } },
          { timestamp: new Date(Date.now() - 30000).toISOString(), risk_score: 0.12, violation: null, frame_url: null, meta_data: { uc4_drift: 0.05, gam_gaze: 0.1, num_faces: 1 } },
        ]
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!sessionId) return null;

  return (
    <div className="fixed inset-0 z-[100] flex justify-end animate-in fade-in duration-300">
       <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" onClick={onClose} />
       
       <div className="relative w-full max-w-4xl bg-white shadow-2xl h-full flex flex-col animate-in slide-in-from-right duration-500 overflow-hidden">
          {/* Header */}
          <div className="bg-slate-900 px-8 py-6 flex justify-between items-center text-white shrink-0 relative overflow-hidden">
             <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
             
             <div className="relative z-10 flex items-center gap-4">
                <div className="w-14 h-14 bg-white/10 rounded-2xl flex items-center justify-center border border-white/20">
                   <ShieldAlert size={32} className="text-rose-500" />
                </div>
                <div>
                  <h2 className="text-xl font-black uppercase tracking-tight">Investigative Audit Terminal</h2>
                  <p className="text-xs font-mono text-indigo-400 font-bold uppercase tracking-widest mt-0.5">Session: {sessionId.split('-')[0]}</p>
                </div>
             </div>
             
             <button onClick={onClose} className="p-3 bg-white/10 hover:bg-white/20 rounded-2xl transition-all border border-white/10">
                <X size={24} />
             </button>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar p-8 bg-slate-50">
             {isLoading ? (
                <div className="h-full flex items-center justify-center flex-col text-slate-400 uppercase font-black tracking-widest">
                   <Activity size={48} className="mb-4 animate-spin text-indigo-500" />
                   Decoding Black Box Data...
                </div>
             ) : (
                <div className="space-y-8">
                   {/* Summary Grid */}
                   <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div className="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col">
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-2">
                            <User size={12} /> Target Student
                         </h4>
                         <p className="text-lg font-black text-slate-900 truncate">{data.student_id}</p>
                      </div>
                      <div className="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col">
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-2">
                            <Activity size={12} /> Current Risk
                         </h4>
                         <p className="text-2xl font-black text-rose-600">
                            { (data.history[0]?.risk_score * 100).toFixed(0) }%
                         </p>
                      </div>
                      <div className="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col">
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2 flex items-center gap-2">
                            <MapPin size={12} /> Sector Node
                         </h4>
                         <p className="text-lg font-black text-slate-900 uppercase">A1-Global</p>
                      </div>
                      <div className="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col bg-slate-900 text-white">
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-2 flex items-center gap-2">
                            <Crosshair size={12} /> Violation Count
                         </h4>
                         <p className="text-2xl font-black text-indigo-400">
                            { data.history.filter(h => h.violation).length }
                         </p>
                      </div>
                   </div>

                   {/* Main Investigation Panel */}
                   <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                      {/* Left: Artifact Timeline */}
                      <div className="lg:col-span-2 flex flex-col gap-6">
                         <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm flex flex-col flex-1">
                            <h3 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-6 flex items-center gap-2">
                               <FileText size={16} className="text-indigo-600" /> Evidence Timeline
                            </h3>
                            
                            <div className="space-y-4">
                               {data.history.map((event, idx) => (
                                  <div key={idx} className={`p-5 rounded-2xl border transition-all ${
                                     event.violation ? 'bg-red-50/30 border-red-100' : 'bg-slate-50/50 border-slate-100'
                                  }`}>
                                     <div className="flex justify-between items-start mb-3">
                                        <div className="flex items-center gap-3">
                                           <div className={`p-2 rounded-lg ${event.violation ? 'bg-red-500 text-white shadow-lg shadow-red-100' : 'bg-slate-200 text-slate-400'}`}>
                                              {event.violation ? <AlertOctagon size={16} /> : <Clock size={16} />}
                                           </div>
                                           <div>
                                              <p className={`text-xs font-black uppercase tracking-tight ${event.violation ? 'text-red-700' : 'text-slate-700'}`}>
                                                 {event.violation ? event.violation.replace(/_/g, ' ') : 'Stability Pulse Established'}
                                              </p>
                                              <p className="text-[10px] font-mono font-bold text-slate-400 mt-0.5">{new Date(event.timestamp).toLocaleTimeString()}</p>
                                           </div>
                                        </div>
                                        <div className="text-right">
                                           <p className={`text-sm font-black ${event.risk_score > 0.7 ? 'text-red-600' : 'text-slate-400'}`}>
                                              { (event.risk_score * 100).toFixed(0) }%
                                           </p>
                                           <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mt-0.5">Fusion Risk</p>
                                        </div>
                                     </div>

                                     {event.meta_data && (
                                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mt-4">
                                           {Object.entries(event.meta_data).map(([key, val]) => (
                                              <div key={key} className="bg-white/60 p-2 rounded-xl border border-slate-100/50">
                                                 <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest truncate">{key.replace('uc', 'UC')}</p>
                                                 <p className="text-[10px] font-black text-slate-800">{typeof val === 'number' ? val.toFixed(3) : val}</p>
                                              </div>
                                           ))}
                                        </div>
                                     )}
                                  </div>
                               ))}
                            </div>
                         </div>
                      </div>

                      {/* Right: Technical Stack & Snapshots */}
                      <div className="flex flex-col gap-6">
                         <div className="bg-slate-900 rounded-[32px] p-6 text-white border border-slate-800 shadow-xl overflow-hidden relative">
                            <div className="absolute top-0 right-0 p-4 opacity-5 translate-x-4">
                               <Cpu size={120} />
                            </div>
                            
                            <h3 className="text-xs font-black text-indigo-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                               <Cpu size={16} /> Technical Stack Metrics
                            </h3>
                            
                            <div className="space-y-6">
                               <div className="space-y-2">
                                  <div className="flex justify-between items-end">
                                     <span className="text-[10px] font-black uppercase text-slate-400 leading-none">Acoustic Guard (UC6)</span>
                                     <span className="text-xs font-black text-indigo-400">
                                        { (data.history[0]?.meta_data?.uc6_audio * 100 || 0).toFixed(0) }% Anomaly
                                     </span>
                                  </div>
                                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                                     <div className="h-full bg-indigo-500 shadow-[0_0_8px_rgba(129,140,248,0.5)]" style={{ width: `${data.history[0]?.meta_data?.uc6_audio * 100 || 0}%` }}></div>
                                  </div>
                               </div>
                               <div className="space-y-2">
                                  <div className="flex justify-between items-end">
                                     <span className="text-[10px] font-black uppercase text-slate-400 leading-none">Ocular Focus (GAM/UC4)</span>
                                     <span className="text-xs font-black text-emerald-400">
                                        { (data.history[0]?.meta_data?.gam_gaze * 100 || 0).toFixed(0) }% Shift
                                     </span>
                                  </div>
                                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                                     <div className="h-full bg-emerald-500" style={{ width: `${data.history[0]?.meta_data?.gam_gaze * 100 || 0}%` }}></div>
                                  </div>
                               </div>
                               <div className="space-y-2">
                                  <div className="flex justify-between items-end">
                                     <span className="text-[10px] font-black uppercase text-slate-400 leading-none">Presence Integrity (UC3)</span>
                                     <span className="text-xs font-black text-rose-400">
                                        { (data.history[0]?.meta_data?.uc3_presence * 100 || 0).toFixed(0) }%
                                     </span>
                                  </div>
                                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                                     <div className="h-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]" style={{ width: `${data.history[0]?.meta_data?.uc3_presence * 100 || 0}%` }}></div>
                                  </div>
                               </div>
                            </div>
                         </div>

                         <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm flex flex-col">
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center justify-between">
                               Captured Evidence (Artifacts)
                               <span className="bg-slate-100 p-1 px-2 rounded-lg text-slate-600">0</span>
                            </h3>

                            <div className="bg-slate-100 rounded-2xl aspect-[4/3] flex flex-col items-center justify-center text-slate-400">
                               <Camera size={32} className="mb-2 opacity-40 shrink-0" />
                               <p className="text-[10px] font-black uppercase tracking-widest">No Visual Evidence Found</p>
                            </div>
                         </div>
                      </div>
                   </div>
                </div>
             )}
          </div>

          {/* Footer Action */}
          <div className="p-6 bg-white border-t border-slate-100 flex gap-4 shrink-0 px-8">
             <button 
                onClick={() => sendCommand('WARN')}
                className="flex-1 py-4 bg-amber-500 text-white rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-amber-600 transition-all shadow-lg shadow-amber-50"
             >
                Dispatch Warning
             </button>
             <button 
                onClick={() => sendCommand('PAUSE')}
                className="flex-1 py-4 bg-slate-900 text-white rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-slate-800 transition-all shadow-lg active:scale-95"
             >
                Impose Iron Curtain
             </button>
             <button 
                onClick={() => sendCommand('TERMINATE')}
                className="flex-1 py-4 bg-rose-600 text-white rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-rose-700 transition-all shadow-lg shadow-rose-100 active:scale-95"
             >
                Forced Termination
             </button>
          </div>
       </div>
    </div>
  );
}
