import React from 'react';
import { Shield, Eye, User, Cpu, AlertCircle } from 'lucide-react';

export default function ExamAIBriefing({ riskData, connected }) {
  const { risk, violation, components } = riskData;

  const getStatusColor = (val) => {
    if (val > 0.7) return 'text-red-500 bg-red-50 border-red-100';
    if (val > 0.4) return 'text-amber-500 bg-amber-50 border-amber-100';
    return 'text-emerald-500 bg-emerald-50 border-emerald-100';
  };

  return (
    <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm mt-4">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
          <Cpu size={18} className="text-indigo-600" /> 
          Neural Intelligence Briefing
        </h3>
        <div className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-tighter border ${connected ? 'text-emerald-600 bg-emerald-50 border-emerald-100' : 'text-red-600 bg-red-50 border-red-100'}`}>
          {connected ? 'Syncing Live' : 'Link Severed'}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Face Detection */}
        <div className={`p-4 rounded-xl border transition-all ${getStatusColor(components?.face_score || 0)}`}>
          <div className="flex items-center gap-3 mb-2">
            <User size={16} />
            <span className="text-[10px] font-black uppercase opacity-70">Identity Lock</span>
          </div>
          <div className="text-xl font-black">{(components?.face_score || 0) > 0.5 ? 'DETECTED' : 'UNSTABLE'}</div>
        </div>

        {/* Gaze Analysis */}
        <div className={`p-4 rounded-xl border transition-all ${getStatusColor(components?.gaze_score || 0)}`}>
          <div className="flex items-center gap-3 mb-2">
            <Eye size={16} />
            <span className="text-[10px] font-black uppercase opacity-70">Ocular Focus</span>
          </div>
          <div className="text-xl font-black">{(components?.gaze_score || 0) < 0.3 ? 'CENTERED' : 'DIVERTED'}</div>
        </div>
      </div>

      {violation && (
        <div className="mt-6 p-4 bg-red-600 text-white rounded-xl shadow-lg shadow-red-100 animate-bounce">
          <div className="flex items-center gap-3">
            <AlertCircle size={20} />
            <div>
              <div className="text-[10px] font-black uppercase opacity-80 leading-none mb-1">Violation Protocol Triggered</div>
              <div className="font-bold text-sm leading-tight uppercase">{violation.replace(/_/g, ' ')}</div>
            </div>
          </div>
        </div>
      )}

      {!violation && risk > 0.1 && (
        <div className="mt-6 flex items-center gap-3 p-4 bg-slate-50 rounded-xl border border-slate-100">
          <div className="w-2 h-2 rounded-full bg-amber-500 animate-ping"></div>
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
            Heuristic Deviance: <span className="text-slate-800">{(risk * 100).toFixed(0)}% Risk Potential</span>
          </p>
        </div>
      )}
    </div>
  );
}
