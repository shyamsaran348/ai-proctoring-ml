import React, { useEffect, useState } from 'react';
import { FileWarning, Shield, Calendar, Clock, DownloadCloud, Eye, AlertCircle, Activity } from 'lucide-react';
import api from '../../services/api';

export default function ViolationGallery({ onDeepDive }) {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRecords = async () => {
      try {
         const res = await api.get('/proctoring_records/');
         setRecords(res.data);
      } catch (err) {
         console.error("Failed fetching violations.");
      } finally {
         setLoading(false);
      }
    };
    fetchRecords();
  }, []);

  if (loading) {
     return (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-20 animate-pulse">
           <Activity size={48} className="mb-4 animate-spin text-indigo-500" />
           <p className="font-black uppercase tracking-widest">Scanning Artifact Database...</p>
        </div>
     );
  }

  const BASE_URL = import.meta.env.VITE_API_URL ? import.meta.env.VITE_API_URL.replace('/api/', '') : 'http://localhost:8000';

  return (
    <div className="bg-white flex flex-col h-full overflow-hidden">
       <div className="bg-white border-b border-slate-100 px-8 py-6 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-3">
             <div className="p-2 bg-slate-900 rounded-xl text-white">
                <FileWarning size={20} />
             </div>
             <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest leading-none">Violation Artifact Vault</h2>
          </div>
          <div className="flex items-center gap-2">
             <span className="bg-slate-100 text-slate-500 text-[10px] font-black px-3 py-1 rounded-full uppercase tracking-widest border border-slate-200">
                {records.length} Logs Archived
             </span>
          </div>
       </div>
       
       <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-slate-50/50">
          {records.length === 0 ? (
             <div className="h-full flex flex-col items-center justify-center text-slate-300 opacity-60">
                 <Shield className="mb-4" size={64} />
                 <p className="font-black uppercase tracking-widest">Archive Depleted</p>
                 <p className="text-xs italic mt-2">All security logs remain within stability parameters.</p>
             </div>
          ) : (
             <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {records.map(record => (
                   <div key={record.id} className="group bg-white rounded-[32px] p-6 border border-slate-100 hover:border-indigo-100 hover:shadow-2xl hover:shadow-slate-200 transition-all duration-500 flex flex-col">
                       
                       <div className="flex justify-between items-start mb-6">
                          <div className="flex-1 min-w-0 pr-4">
                             <div className="flex items-center gap-2 mb-1">
                                <AlertCircle size={14} className="text-rose-500" />
                                <h4 className="font-black text-slate-900 uppercase tracking-tight line-clamp-1 group-hover:text-indigo-600 transition-colors">
                                   {record.violation_type?.replace(/_/g, ' ') || 'Unknown Anomaly'}
                                </h4>
                             </div>
                             <p className="text-[10px] text-slate-400 font-black uppercase tracking-widest">
                                Target: <span className="text-slate-800">{record.student_name}</span>
                             </p>
                          </div>
                          <div className={`px-3 py-1 rounded-xl text-[10px] font-black uppercase tracking-widest border-2 shadow-sm ${
                             record.risk_score >= 0.7 ? 'bg-rose-50 text-rose-600 border-rose-100' : 
                             'bg-amber-50 text-amber-600 border-amber-100'
                          }`}>
                            {(record.risk_score * 100).toFixed(0)}% Risk
                          </div>
                       </div>
                       
                       <div className="flex space-x-6 mb-6 text-[10px] text-slate-400 font-black uppercase tracking-[0.15em]">
                          <span className="flex items-center gap-2"><Calendar size={14} className="text-indigo-500 opacity-60"/> {new Date(record.created_at).toLocaleDateString()}</span>
                          <span className="flex items-center gap-2"><Clock size={14} className="text-indigo-500 opacity-60"/> {new Date(record.created_at).toLocaleTimeString()}</span>
                       </div>

                       {record.snapshot_url && (
                          <div className="mt-auto relative rounded-[24px] overflow-hidden border border-slate-100 bg-slate-900 group/image aspect-video">
                             <img 
                                src={record.snapshot_url.startsWith('http') ? record.snapshot_url : `${BASE_URL}${record.snapshot_url}`} 
                                alt="Violation Artifact" 
                                className="w-full h-full object-cover opacity-80 group-hover/image:opacity-100 group-hover/image:scale-105 transition-all duration-700"
                                onError={(e) => { e.target.style.display = 'none'; }}
                             />
                             <div className="absolute inset-0 bg-slate-900/60 flex flex-col items-center justify-center opacity-0 group-hover/image:opacity-100 transition-all duration-500 backdrop-blur-[2px]">
                                <div className="flex gap-2">
                                   <button 
                                      onClick={() => onDeepDive && onDeepDive(record.session_id)}
                                      className="text-white bg-indigo-600 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center shadow-lg hover:bg-indigo-700 transition-all active:scale-95"
                                   >
                                      <Eye size={14} className="mr-2" /> Inspect Session
                                   </button>
                                   <a 
                                     href={record.snapshot_url.startsWith('http') ? record.snapshot_url : `${BASE_URL}${record.snapshot_url}`} 
                                     target="_blank" rel="noreferrer"
                                     className="text-white bg-slate-800/80 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center shadow-lg hover:bg-slate-700 transition-all active:scale-95"
                                   >
                                     <DownloadCloud size={14} className="mr-2" /> Capture Frame
                                   </a>
                                </div>
                             </div>
                          </div>
                       )}

                   </div>
                ))}
             </div>
          )}
       </div>
    </div>
  );
}
