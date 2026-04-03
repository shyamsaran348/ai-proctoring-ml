import React, { useEffect, useState } from 'react';
import { AlertOctagon, Activity, Eye, ShieldAlert, Cpu } from 'lucide-react';
import api from '../../services/api';

export default function LiveAlertStream({ onDeepDive }) {
  const [alerts, setAlerts] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // connecting, live, disconnected

  useEffect(() => {
    // We attach an EventSource to the Django SSE endpoint
    // IMPORTANT: withCredentials ensures the Django Session Auth cookie is passed from localhost:5173 to localhost:8000
    const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/';
    const sseURL = baseURL.endsWith('/') ? `${baseURL}proctoring_pulse` : `${baseURL}/proctoring_pulse`;
    
    let pulse;
    try {
        pulse = new EventSource(sseURL, { withCredentials: true });
        
        pulse.onopen = () => setConnectionStatus('live');
        
        pulse.onerror = (err) => {
            console.error("SSE Error:", err);
            setConnectionStatus('disconnected');
            pulse.close();
        };

        pulse.addEventListener('risk_alert', (e) => {
            try {
                const dataArray = JSON.parse(e.data);
                if (dataArray && dataArray.length > 0) {
                    setAlerts(prev => {
                        // Merge and keep the latest 100 alerts
                        const newAlerts = [...dataArray, ...prev].slice(0, 100);
                        return newAlerts;
                    });
                }
            } catch (err) { }
        });

    } catch (e) {
        setConnectionStatus('disconnected');
    }

    return () => {
      if (pulse) pulse.close();
    };
  }, []);

  return (
    <div className="bg-white rounded-[32px] shadow-sm border border-gray-200 overflow-hidden flex flex-col h-full max-h-screen">
       
       {/* Global Status Banner */}
       <div className={`px-8 py-6 border-b flex items-center justify-between ${
          connectionStatus === 'live' ? 'bg-slate-900 text-white' : 
          connectionStatus === 'connecting' ? 'bg-amber-100 border-amber-200 text-amber-800' : 
          'bg-red-100 border-red-200 text-red-800'
       }`}>
           <div className="flex items-center space-x-3">
              {connectionStatus === 'live' && <Activity size={24} className="animate-pulse text-indigo-400" />}
              {connectionStatus === 'connecting' && <Cpu size={24} className="animate-spin text-amber-600" />}
              {connectionStatus === 'disconnected' && <ShieldAlert size={24} className="text-red-600" />}
              <h2 className="text-lg font-black uppercase tracking-widest leading-none">
                 {connectionStatus === 'live' ? 'Global Fusion Core: Live' : 
                  connectionStatus === 'connecting' ? 'Synchronizing...' : 
                  'Connection Lost'}
              </h2>
           </div>
           
           <div className="flex items-center space-x-2 text-[10px] font-black uppercase tracking-widest opacity-60">
              <span>Security Stream</span>
           </div>
       </div>

       {/* Alert Stream List */}
       <div className="flex-1 overflow-y-auto bg-slate-50 p-6 space-y-4 custom-scrollbar">
          {alerts.length === 0 ? (
             <div className="h-full flex flex-col items-center justify-center text-slate-300 opacity-60">
                 <ShieldAlert size={64} className="mb-4" />
                 <p className="font-black uppercase tracking-widest">No Sector Anomalies</p>
                 <p className="text-xs italic mt-2">All examination nodes are stabilized.</p>
             </div>
          ) : (
             alerts.map((alert, idx) => (
                <div 
                   key={`${alert.session_id}-${idx}`}
                   className="bg-white border-l-4 border-rose-500 rounded-2xl p-6 shadow-sm hover:shadow-xl hover:shadow-rose-100 transition-all relative overflow-hidden group animate-in slide-in-from-right-4"
                >
                   {/* Danger BG glow */}
                   <div className="absolute top-0 right-0 w-32 h-32 bg-rose-100 rounded-full blur-3xl -mr-16 -mt-16 opacity-0 group-hover:opacity-60 transition-opacity"></div>
                   
                   <div className="relative z-10 flex justify-between items-center">
                      <div className="flex-1">
                         <div className="flex items-center space-x-2 mb-2">
                            <AlertOctagon size={16} className="text-rose-500 flex-shrink-0" />
                            <span className="font-black text-rose-700 text-[10px] uppercase tracking-[0.2em]">
                               Critical Fusion Spike
                            </span>
                         </div>
                         <h3 className="text-xl font-black text-slate-900 flex items-center mb-1 uppercase tracking-tight">
                            {alert.primary_violation || 'Suspicious Behavior'}
                         </h3>
                         <p className="text-xs text-slate-400 font-black uppercase tracking-widest">
                            Target Identity: <span className="text-slate-900">{alert.student}</span>
                         </p>
                         <p className="text-[10px] text-slate-300 font-mono mt-1">
                            LOG_ID: {alert.session_id.split('-')[0].toUpperCase()}
                         </p>
                      </div>

                      <div className="text-right flex flex-col items-end shrink-0 pl-6">
                         <div className="bg-rose-50 text-rose-600 font-black text-2xl px-4 py-2 rounded-2xl border border-rose-100 shadow-sm">
                            {(alert.risk * 100).toFixed(0)}%
                         </div>
                         <span className="text-[9px] text-slate-400 uppercase font-black tracking-widest mt-2 block">Risk Prob</span>
                         
                         {/* Sentinel 7-Signal Mini-Hud */}
                         {alert.metrics && (
                            <div className="grid grid-cols-3 gap-1 mt-4 w-full">
                               {[
                                 { label: 'ID', val: alert.metrics.sim, color: alert.metrics.sim < 0.5 ? 'bg-rose-500' : 'bg-emerald-500' },
                                 { label: 'PR', val: alert.metrics.presence, color: alert.metrics.presence > 0.6 ? 'bg-rose-500' : 'bg-emerald-500' },
                                 { label: 'AU', val: alert.metrics.audio, color: alert.metrics.audio > 0.6 ? 'bg-rose-500' : 'bg-emerald-500' }
                               ].map(m => (
                                  <div key={m.label} className="flex flex-col items-center">
                                     <div className={`w-full h-1 rounded-full ${m.color} opacity-40 mb-1`}>
                                        <div className={`h-full ${m.color} rounded-full`} style={{ width: `${m.val * 100}%` }}></div>
                                     </div>
                                     <span className="text-[7px] font-black text-slate-400">{m.label}</span>
                                  </div>
                               ))}
                            </div>
                         )}

                         <button 
                             onClick={() => onDeepDive(alert.session_id)}
                             className="mt-6 flex items-center text-[10px] font-black uppercase tracking-widest bg-slate-900 hover:bg-indigo-600 text-white px-4 py-2.5 rounded-xl transition-all shadow-lg active:scale-95 w-full justify-center"
                         >
                            <Eye size={14} className="mr-2" /> Deep Dive
                         </button>
                      </div>
                   </div>
                </div>
             ))
          )}
       </div>
    </div>
  );
}
