import React, { useState, useEffect } from 'react';
import { 
  X, 
  Search, 
  Plus, 
  Trash2, 
  ArrowRight, 
  GripVertical,
  Clock,
  Layers,
  Zap,
  Save,
  CheckCircle,
  Database
} from 'lucide-react';
import api from '../../services/api';

export default function ExamDesigner({ contestId, onClose }) {
  const [available, setAvailable] = useState({ problems: [], mcqs: [] });
  const [selected, setSelected] = useState({ problems: [], mcqs: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (contestId) fetchStructure();
  }, [contestId]);

  const fetchStructure = async () => {
    try {
      setIsLoading(true);
      const res = await api.get(`/contests/${contestId}/exam_designer/`);
      
      // Available (not in contest)
      const unselectedProbs = res.data.all_problems.filter(p => 
        !res.data.current_problems.some(cp => cp.problem_id === p.id)
      );
      const unselectedMcqs = res.data.all_mcqs.filter(m => 
        !res.data.current_mcqs.some(cm => cm.id === m.id)
      );
      
      setAvailable({ problems: unselectedProbs, mcqs: unselectedMcqs });
      
      // Selected (current state)
      const currProbs = res.data.current_problems.map(cp => {
         const base = res.data.all_problems.find(p => p.id === cp.problem_id);
         return { ...base, limit: cp.time_limit_override || 60 };
      });
      const currMcqs = res.data.current_mcqs.map(cm => {
         return res.data.all_mcqs.find(m => m.id === cm.id);
      });
      
      setSelected({ problems: currProbs, mcqs: currMcqs });
    } catch (err) {
      console.error("Failed to fetch designer structure.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddProblem = (prob) => {
    setAvailable(prev => ({ ...prev, problems: prev.problems.filter(p => p.id !== prob.id) }));
    setSelected(prev => ({ ...prev, problems: [...prev.problems, { ...prob, limit: 60 }] }));
  };

  const handleRemoveProblem = (prob) => {
    setSelected(prev => ({ ...prev, problems: prev.problems.filter(p => p.id !== prob.id) }));
    setAvailable(prev => ({ ...prev, problems: [...prev.problems, prob] }));
  };

  const handleAddMcq = (mcq) => {
    setAvailable(prev => ({ ...prev, mcqs: prev.mcqs.filter(m => m.id !== mcq.id) }));
    setSelected(prev => ({ ...prev, mcqs: [...prev.mcqs, mcq] }));
  };

  const handleRemoveMcq = (mcq) => {
    setSelected(prev => ({ ...prev, mcqs: prev.mcqs.filter(m => m.id !== mcq.id) }));
    setAvailable(prev => ({ ...prev, mcqs: [...prev.mcqs, mcq] }));
  };

  const handleSave = async () => {
    try {
      setIsSaving(true);
      const payload = {
        contest_id: contestId,
        problems: selected.problems.map((p, i) => ({ id: p.id, order: i, limit: p.limit })),
        mcqs: selected.mcqs.map((m, i) => ({ id: m.id, order: i }))
      };
      await api.post(`/contests/${contestId}/update_structure/`, payload);
      alert('Examination Node Restructured Successfully.');
      onClose();
    } catch (err) {
      alert('Restructuring command failed.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-8 animate-in fade-in duration-300">
       <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={onClose} />
       
       <div className="relative w-full max-w-7xl bg-white shadow-2xl rounded-[40px] flex flex-col h-full animate-in zoom-in-95 duration-500 overflow-hidden">
          {/* Header */}
          <div className="bg-white px-10 py-8 flex justify-between items-center shrink-0 border-b border-slate-100">
             <div className="flex items-center gap-5">
                <div className="w-16 h-16 bg-slate-900 rounded-[22px] flex items-center justify-center text-white shadow-xl shadow-slate-200">
                   <Layers size={32} />
                </div>
                <div>
                  <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Node Structure Designer</h2>
                  <p className="text-sm font-medium text-slate-400 italic">Configure the sequence and constraints of the examination flow.</p>
                </div>
             </div>
             
             <div className="flex gap-4">
                <button onClick={onClose} className="px-6 py-3 bg-slate-50 text-slate-500 rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-slate-100 transition-all">
                   Abort Design
                </button>
                <button 
                   onClick={handleSave}
                   disabled={isSaving}
                   className="px-8 py-3 bg-indigo-600 text-white rounded-2xl font-black text-[10px] uppercase tracking-widest hover:bg-slate-900 transition-all shadow-lg shadow-indigo-100 flex items-center gap-3 active:scale-95"
                >
                   {isSaving ? 'Synching...' : (
                      <> <Save size={16} /> Deploy Structure </>
                   )}
                </button>
             </div>
          </div>

          <div className="flex-1 overflow-hidden flex bg-slate-50">
             {/* Left: Available Bank */}
             <div className="w-1/3 flex flex-col border-r border-slate-100 h-full p-8 space-y-6">
                <div className="flex items-center justify-between">
                   <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                      <Database size={14} /> Global Resource Bank
                   </h3>
                   <span className="text-[10px] font-black text-indigo-600 bg-indigo-50 px-2 py-1 rounded-lg">
                      {available.problems.length + available.mcqs.length} Available
                   </span>
                </div>

                <div className="relative">
                   <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                   <input 
                      type="text" 
                      placeholder="Search question bank..."
                      className="w-full pl-10 pr-4 py-3 bg-white border border-slate-200 rounded-2xl text-[11px] font-bold uppercase tracking-widest focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none"
                   />
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 pr-2">
                   <p className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] mb-4">Coding Protocols</p>
                   {available.problems.map(p => (
                      <div key={p.id} className="group bg-white p-4 rounded-2xl border border-slate-200 hover:border-indigo-400 hover:shadow-md transition-all flex items-center justify-between">
                         <div>
                            <p className="text-xs font-black text-slate-800 uppercase tracking-tight">{p.title}</p>
                            <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">{p.difficulty}</span>
                         </div>
                         <button onClick={() => handleAddProblem(p)} className="p-2 bg-indigo-50 text-indigo-600 rounded-xl hover:bg-slate-900 hover:text-white transition-all">
                            <Plus size={16} />
                         </button>
                      </div>
                   ))}

                   <p className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] mt-8 mb-4">MCQ Artifacts</p>
                   {available.mcqs.map(m => (
                      <div key={m.id} className="group bg-white p-4 rounded-2xl border border-slate-200 hover:border-emerald-400 hover:shadow-md transition-all flex items-center justify-between">
                         <div className="flex-1 pr-4">
                            <p className="text-xs font-black text-slate-800 uppercase tracking-tight line-clamp-2">{m.question_text}</p>
                         </div>
                         <button onClick={() => handleAddMcq(m)} className="p-2 bg-emerald-50 text-emerald-600 rounded-xl hover:bg-slate-900 hover:text-white transition-all shrink-0">
                            <Plus size={16} />
                         </button>
                      </div>
                   ))}
                </div>
             </div>

             {/* Right: Active Exam Structure */}
             <div className="flex-1 flex flex-col p-10 space-y-8 overflow-y-auto custom-scrollbar">
                <div className="flex items-center gap-10">
                   <div className="flex-1 p-6 bg-indigo-600 rounded-[32px] text-white flex items-center justify-between relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10">
                         <Zap size={64} />
                      </div>
                      <div>
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-indigo-200 mb-1">Active Coding Pulse</h4>
                         <p className="text-4xl font-black">{selected.problems.length}</p>
                      </div>
                      <ArrowRight size={32} className="text-indigo-400" />
                   </div>
                   <div className="flex-1 p-6 bg-slate-900 rounded-[32px] text-white flex items-center justify-between relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10">
                         <LayoutGrid size={64} />
                      </div>
                      <div>
                         <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">MCQ Batch Size</h4>
                         <p className="text-4xl font-black">{selected.mcqs.length}</p>
                      </div>
                      <ArrowRight size={32} className="text-slate-700" />
                   </div>
                </div>

                <div className="space-y-4">
                   <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                       <GripVertical size={16} className="text-slate-400" /> Sequencing Order
                   </h3>
                   
                   {selected.problems.length === 0 && selected.mcqs.length === 0 ? (
                      <div className="py-20 border-2 border-dashed border-slate-200 rounded-[40px] flex flex-col items-center justify-center text-slate-300">
                         <Layers size={48} className="mb-4 opacity-20" />
                         <p className="text-sm font-black uppercase tracking-widest leading-none">Structure Empty</p>
                         <p className="text-xs mt-2 italic">Add resources from the bank to begin constructing the node.</p>
                      </div>
                   ) : (
                      <div className="space-y-3">
                         {selected.problems.map((p, i) => (
                            <div key={p.id} className="bg-white p-5 rounded-3xl border border-indigo-100 shadow-sm flex items-center gap-5 group animate-in slide-in-from-left-4">
                               <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center text-[10px] font-black">
                                  {i + 1}
                               </div>
                               <GripVertical size={20} className="text-slate-300 cursor-move" />
                               <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                     <span className="text-[10px] font-black uppercase text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-lg">Code</span>
                                     <h5 className="text-sm font-black text-slate-900 uppercase tracking-tight">{p.title}</h5>
                                  </div>
                                  <div className="flex gap-4">
                                     <span className="flex items-center gap-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                                        <Clock size={12} /> Limit: {p.limit}m
                                     </span>
                                  </div>
                               </div>
                               <button onClick={() => handleRemoveProblem(p)} className="p-3 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-2xl transition-all">
                                  <Trash2 size={20} />
                               </button>
                            </div>
                         ))}

                         {selected.mcqs.map((m, i) => (
                            <div key={m.id} className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm flex items-center gap-5 group animate-in slide-in-from-left-4">
                               <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center text-[10px] font-black">
                                  {selected.problems.length + i + 1}
                               </div>
                               <GripVertical size={20} className="text-slate-300 cursor-move" />
                               <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                     <span className="text-[10px] font-black uppercase text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-lg">MCQ</span>
                                     <h5 className="text-sm font-black text-slate-900 uppercase tracking-tight line-clamp-1">{m.question_text}</h5>
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                                     Score Weight: {m.marks}pts
                                  </span>
                               </div>
                               <button onClick={() => handleRemoveMcq(m)} className="p-3 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-2xl transition-all">
                                  <Trash2 size={20} />
                               </button>
                            </div>
                         ))}
                      </div>
                   )}
                </div>
             </div>
          </div>
       </div>
    </div>
  );
}
