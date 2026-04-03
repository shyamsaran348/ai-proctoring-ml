import React, { useState, useEffect } from 'react';
import { 
  X, 
  Save, 
  Code, 
  ListOrdered, 
  Trash2, 
  Plus, 
  PlusCircle, 
  CheckCircle2, 
  AlertCircle 
} from 'lucide-react';
import api from '../../services/api';

export default function QuestionEditor({ type, question, onClose, onSave }) {
  const isNew = !question;
  const [formData, setFormData] = useState(
    question || (type === 'coding' ? {
      title: '',
      difficulty: 'easy',
      points: 10,
      problem_statement: '',
      input_format: '',
      output_format: '',
      constraints: '',
      sample_input: '',
      sample_output: '',
    } : {
      question_text: '',
      options: ['', '', '', ''],
      correct_option: 0,
      marks: 1,
    })
  );

  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleOptionChange = (idx, val) => {
    const newOptions = [...formData.options];
    newOptions[idx] = val;
    setFormData(prev => ({ ...prev, options: newOptions }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      const endpoint = type === 'coding' ? '/problems/' : '/mcqs/';
      if (isNew) {
        await api.post(endpoint, formData);
      } else {
        await api.put(`${endpoint}${question.id}/`, formData);
      }
      onSave();
      onClose();
    } catch (err) {
      console.error("Save failed:", err);
      alert("Verification Error: Encryption keys mismatch. Repository rejected the payload.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center p-8 animate-in fade-in duration-300">
       <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={onClose} />
       
       <div className="relative w-full max-w-5xl bg-white shadow-2xl rounded-[40px] flex flex-col max-h-full animate-in zoom-in-95 duration-500 overflow-hidden">
          <form onSubmit={handleSubmit} className="flex flex-col h-full overflow-hidden">
             
             {/* Header */}
             <div className="px-10 py-8 border-b border-slate-100 flex justify-between items-center bg-white shrink-0">
                <div className="flex items-center gap-5">
                   <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-white shadow-xl shadow-slate-200 ${type === 'coding' ? 'bg-indigo-600' : 'bg-emerald-500'}`}>
                      {type === 'coding' ? <Code size={28} /> : <ListOrdered size={28} />}
                   </div>
                   <div>
                      <h2 className="text-xl font-black text-slate-900 uppercase tracking-tight">
                         {isNew ? 'Initialize New Neural Node' : 'Recalibrate Existing Node'}
                      </h2>
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none mt-1">
                         Type: {type === 'coding' ? 'Coding Logic Stream' : 'MCQ Artifact'}
                      </p>
                   </div>
                </div>
                <button type="button" onClick={onClose} className="p-3 bg-slate-50 hover:bg-slate-100 text-slate-500 rounded-2xl transition-all">
                   <X size={24} />
                </button>
             </div>

             <div className="flex-1 overflow-y-auto custom-scrollbar p-10 space-y-8">
                {type === 'coding' ? (
                   <>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                         <div className="md:col-span-2 space-y-1.5">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Node Title</label>
                            <input 
                               name="title" value={formData.title} onChange={handleChange} required
                               className="w-full bg-slate-50 border-none rounded-2xl p-4 text-xs font-black uppercase tracking-widest focus:ring-2 focus:ring-indigo-500/20 transition-all font-mono"
                               placeholder="e.g. TWO_SUM_OPTIMIZER"
                            />
                         </div>
                         <div className="space-y-1.5">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Complexity Rank</label>
                            <select 
                               name="difficulty" value={formData.difficulty} onChange={handleChange}
                               className="w-full bg-slate-50 border-none rounded-2xl p-4 text-xs font-black uppercase tracking-widest focus:ring-2 focus:ring-indigo-500/20 transition-all cursor-pointer"
                            >
                               <option value="easy">Rank: Easy</option>
                               <option value="medium">Rank: Medium</option>
                               <option value="hard">Rank: Hard</option>
                            </select>
                         </div>
                      </div>

                      <div className="space-y-1.5">
                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Objective Definition</label>
                         <textarea 
                            name="problem_statement" value={formData.problem_statement} onChange={handleChange} required
                            className="w-full bg-slate-50 border-none rounded-3xl p-6 text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all min-h-[160px] leading-relaxed"
                            placeholder="Define the logical objective of this node..."
                         />
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                         <div className="space-y-1.5">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Input Logic stream</label>
                            <textarea 
                               name="input_format" value={formData.input_format} onChange={handleChange}
                               className="w-full bg-slate-50 border-none rounded-2xl p-4 text-xs font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all min-h-[100px]"
                            />
                         </div>
                         <div className="space-y-1.5">
                            <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Output Target Format</label>
                            <textarea 
                               name="output_format" value={formData.output_format} onChange={handleChange}
                               className="w-full bg-slate-50 border-none rounded-2xl p-4 text-xs font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all min-h-[100px]"
                            />
                         </div>
                      </div>
                   </>
                ) : (
                   <>
                      <div className="space-y-1.5">
                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Probing Question</label>
                         <textarea 
                            name="question_text" value={formData.question_text} onChange={handleChange} required
                            className="w-full bg-slate-50 border-none rounded-3xl p-6 text-sm font-medium focus:ring-2 focus:ring-emerald-500/20 transition-all min-h-[120px] leading-relaxed"
                            placeholder="Input the inquiry for this node..."
                         />
                      </div>

                      <div className="space-y-4">
                         <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">Artifact Option Branches</label>
                         <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {formData.options.map((opt, i) => (
                               <div key={i} className={`flex items-center gap-4 p-4 rounded-2xl border-2 transition-all ${
                                  formData.correct_option === i ? 'border-emerald-500 bg-emerald-50/30' : 'border-slate-100 bg-white'
                               }`}>
                                  <button 
                                     type="button"
                                     onClick={() => setFormData(prev => ({ ...prev, correct_option: i }))}
                                     className={`w-6 h-6 rounded-full flex items-center justify-center transition-all ${
                                        formData.correct_option === i ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-200' : 'bg-slate-200 text-transparent'
                                     }`}
                                  >
                                     <CheckCircle2 size={14} />
                                  </button>
                                  <input 
                                     value={opt} 
                                     onChange={(e) => handleOptionChange(i, e.target.value)} 
                                     className="flex-1 bg-transparent border-none p-0 text-xs font-bold focus:ring-0"
                                     placeholder={`Option ${i + 1}`}
                                  />
                               </div>
                            ))}
                         </div>
                      </div>
                   </>
                )}
             </div>

             {/* Footer */}
             <div className="px-10 py-8 border-t border-slate-100 flex justify-between items-center bg-white shrink-0">
                <div className="flex items-center gap-2 text-rose-500 bg-rose-50 px-4 py-2 rounded-xl border border-rose-100">
                   <AlertCircle size={16} />
                   <span className="text-[10px] font-black uppercase tracking-widest">Unauthorized Access Prohibited</span>
                </div>
                <div className="flex gap-4">
                   <button type="button" onClick={onClose} className="px-6 py-3 text-slate-400 text-[10px] font-black uppercase tracking-widest hover:text-slate-600 transition-all">
                      Cancel Shift
                   </button>
                   <button 
                      type="submit" 
                      disabled={isLoading}
                      className={`px-10 py-3 rounded-2xl font-black text-[10px] uppercase tracking-widest flex items-center gap-3 transition-all active:scale-95 shadow-xl ${
                         type === 'coding' ? 'bg-indigo-600 text-white shadow-indigo-100' : 'bg-emerald-500 text-white shadow-emerald-100'
                      }`}
                   >
                      {isLoading ? 'Injecting Data...' : (
                         <> <Save size={16} /> Synchronize to Bank </>
                      )}
                   </button>
                </div>
             </div>
          </form>
       </div>
    </div>
  );
}
