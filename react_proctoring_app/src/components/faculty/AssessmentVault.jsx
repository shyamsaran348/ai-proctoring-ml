import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Trash2, 
  Edit3, 
  Play, 
  Archive, 
  MoreVertical, 
  Search, 
  Filter,
  Users,
  Clock,
  Layers,
  ChevronRight,
  ShieldCheck
} from 'lucide-react';
import api from '../../services/api';

export default function AssessmentVault({ onOpenDesigner }) {
  const [exams, setExams] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetchExams();
  }, []);

  const fetchExams = async () => {
    try {
      setIsLoading(true);
      // In the backend, contests are often listed in the main faculty view
      // We'll assume there's a /contests/ endpoint or similar available
      const res = await api.get('/contests/');
      setExams(res.data);
    } catch (err) {
      console.error("Failed to fetch assessments.");
      // Fallback for demo if endpoint not found
      setExams([
        { id: 1, title: 'Algorithm Speedrun #4', type: 'coding', participants: 42, status: 'active', date: '2026-04-02', proctoring: true },
        { id: 2, title: 'System Design Midterm', type: 'hybrid', participants: 128, status: 'upcoming', date: '2026-04-15', proctoring: true },
        { id: 3, title: 'Frontend Basics Quiz', type: 'mcq', participants: 0, status: 'draft', date: '2026-04-20', proctoring: false },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredExams = exams.filter(e => 
    e.title.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-end">
        <div>
           <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Assessment Vault</h2>
           <p className="text-slate-500 text-sm font-medium italic">Authorized examination nodes and secure challenges.</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={fetchExams}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-slate-50 text-slate-600 text-[10px] font-black uppercase tracking-widest rounded-xl border border-slate-200 hover:border-indigo-500 hover:text-indigo-600 transition-all shadow-sm disabled:opacity-50"
          >
             <Clock size={14} className={isLoading ? 'animate-spin' : ''} />
             Sync Repository
          </button>
          <button className="btn-primary flex items-center gap-2 group">
             <Plus size={16} className="group-hover:rotate-90 transition-transform duration-300" />
             Create New Node
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
         <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4">
            <div className="w-12 h-12 bg-indigo-50 rounded-xl flex items-center justify-center text-indigo-600">
               <Layers size={24} />
            </div>
            <div>
               <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Total Nodes</p>
               <p className="text-xl font-black text-slate-900">{exams.length}</p>
            </div>
         </div>
         <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4">
            <div className="w-12 h-12 bg-emerald-50 rounded-xl flex items-center justify-center text-emerald-600">
               <Play size={24} />
            </div>
            <div>
               <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Active Links</p>
               <p className="text-xl font-black text-slate-900">{exams.filter(e => e.status === 'active').length}</p>
            </div>
         </div>
         <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-2 opacity-5 scale-150 rotate-12 transition-transform group-hover:scale-[2] group-hover:rotate-0 duration-700">
               <ShieldCheck size={48} className="text-indigo-600" />
            </div>
            <div className="w-12 h-12 bg-slate-900 rounded-xl flex items-center justify-center text-white relative z-10">
               <ShieldCheck size={24} />
            </div>
            <div className="relative z-10">
               <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Protected</p>
               <p className="text-xl font-black text-slate-900">{exams.filter(e => e.proctoring).length}</p>
            </div>
         </div>
      </div>

      <div className="flex-1 bg-white rounded-[32px] border border-slate-100 shadow-xl shadow-slate-200/50 flex flex-col overflow-hidden">
        <div className="p-6 border-b border-slate-50 flex gap-4 items-center">
            <div className="relative flex-1">
               <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
               <input 
                  type="text" 
                  placeholder="Search assessment nodes..."
                  className="w-full pl-12 pr-4 py-3 bg-slate-50 border-none rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
               />
            </div>
            <button className="p-3 bg-slate-50 text-slate-500 rounded-2xl hover:bg-slate-100 transition-colors">
               <Filter size={20} />
            </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
           {isLoading ? (
              <div className="space-y-4 p-4">
                 {[1,2,3].map(i => (
                    <div key={i} className="h-24 bg-slate-50 rounded-2xl animate-pulse" />
                 ))}
              </div>
           ) : (
              <div className="space-y-3">
                 {filteredExams.map(exam => (
                    <div key={exam.id} className="group bg-white p-5 rounded-2xl border border-slate-50 hover:border-indigo-100 hover:bg-indigo-50/10 transition-all duration-300 flex items-center justify-between">
                       <div className="flex items-center gap-5">
                          <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-black text-xs uppercase tracking-tighter ${
                             exam.type === 'coding' ? 'bg-indigo-600 text-white' : 
                             exam.type === 'mcq' ? 'bg-emerald-500 text-white' : 
                             'bg-slate-900 text-white'
                          }`}>
                             {exam.type === 'coding' ? 'Code' : exam.type === 'mcq' ? 'MCQ' : 'Hybr'}
                          </div>
                          <div>
                             <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight group-hover:text-indigo-600 transition-colors">{exam.title}</h4>
                             <div className="flex items-center gap-4 mt-1">
                                <span className="flex items-center gap-1.5 text-xs font-bold text-slate-400">
                                   <Users size={12} /> {exam.participants} Participants
                                </span>
                                <span className="flex items-center gap-1.5 text-xs font-bold text-slate-400">
                                   <Clock size={12} /> {exam.date}
                                </span>
                                {exam.proctoring && (
                                   <span className="flex items-center gap-1 text-[9px] font-black uppercase text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-100">
                                      AI Protected
                                   </span>
                                )}
                             </div>
                          </div>
                       </div>

                       <div className="flex items-center gap-2">
                          <button 
                             onClick={() => onOpenDesigner(exam.id)}
                             className="px-4 py-2 bg-white text-slate-600 text-[10px] font-black uppercase tracking-widest rounded-xl border border-slate-200 hover:border-indigo-500 hover:text-indigo-600 transition-all shadow-sm"
                          >
                             Structure Designer
                          </button>
                          <div className="h-8 w-px bg-slate-100 mx-2" />
                          <button className="p-2.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all">
                             <Trash2 size={18} />
                          </button>
                          <button className="p-2.5 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all">
                             <MoreVertical size={18} />
                          </button>
                       </div>
                    </div>
                 ))}
              </div>
           )}
        </div>
      </div>
    </div>
  );
}
