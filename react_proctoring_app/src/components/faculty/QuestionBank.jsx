import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Trash2, 
  Edit3, 
  Search, 
  Filter, 
  Code, 
  ListOrdered, 
  ChevronRight, 
  Cpu, 
  ShieldCheck,
  Zap,
  LayoutGrid,
  MoreVertical
} from 'lucide-react';
import api from '../../services/api';
import QuestionEditor from './QuestionEditor';

export default function QuestionBank() {
  const [questions, setQuestions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [tab, setTab] = useState('coding'); // coding, mcq
  const [search, setSearch] = useState('');
  
  // Editor state
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [selectedQuestion, setSelectedQuestion] = useState(null);

  useEffect(() => {
    fetchResources();
  }, [tab]);

  const fetchResources = async () => {
    try {
      setIsLoading(true);
      const endpoint = tab === 'coding' ? '/problems/' : '/mcqs/';
      const res = await api.get(endpoint);
      setQuestions(res.data);
    } catch (err) {
      console.error("Failed to fetch questions.");
      // Fallback for demo
      if (tab === 'coding') {
         setQuestions([
            { id: 1, title: 'Two Sum Strategy', difficulty: 'easy', points: 10, category: 'Algorithms', problem_statement: 'Given an array of integers...', input_format: 'int[] nums', output_format: 'int[]', constraints: 'O(n) time', sample_input: '[1,2,3]', sample_output: '[0,1]' },
            { id: 2, title: 'B-Tree Balance Delta', difficulty: 'hard', points: 50, category: 'Data Structures' },
            { id: 3, title: 'LRU Cache Fusion', difficulty: 'medium', points: 30, category: 'Systems' },
         ]);
      } else {
         setQuestions([
            { id: 1, question_text: 'What is the time complexity of a hash table insertion?', marks: 2, options: ['O(1)', 'O(n)', 'O(log n)', 'O(n^2)'], correct_option: 0 },
            { id: 2, question_text: 'Which protocol is used for real-time video stream transmission?', marks: 5, options: ['HTTP', 'FTP', 'RTSP', 'SMTP'], correct_option: 2 },
         ]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleEdit = (q) => {
    setSelectedQuestion(q);
    setIsEditorOpen(true);
  };

  const handleNew = () => {
    setSelectedQuestion(null);
    setIsEditorOpen(true);
  };

  const filteredQuestions = questions.filter(q => 
    (q.title || q.question_text || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-end">
        <div>
           <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Question Command Repository</h2>
           <p className="text-slate-500 text-sm font-medium italic">Shared neural bank for distributed examination challenges.</p>
        </div>
        <div className="flex gap-2 bg-slate-100 p-1 rounded-2xl border border-slate-200 shadow-inner">
           <button 
              onClick={() => setTab('coding')}
              className={`flex items-center gap-2 px-4 py-2 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all ${
                 tab === 'coding' ? 'bg-white text-indigo-600 shadow-sm shadow-slate-200' : 'text-slate-500 hover:text-slate-700'
              }`}
           >
              <Code size={14} /> Coding Problems
           </button>
           <button 
              onClick={() => setTab('mcq')}
              className={`flex items-center gap-2 px-4 py-2 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all ${
                 tab === 'mcq' ? 'bg-white text-emerald-600 shadow-sm shadow-slate-200' : 'text-slate-500 hover:text-slate-700'
              }`}
           >
              <ListOrdered size={14} /> MCQ Bank
           </button>
        </div>
      </div>

      <div className="flex-1 bg-white rounded-[32px] border border-slate-100 shadow-xl shadow-slate-200/50 flex flex-col overflow-hidden">
        <div className="p-6 border-b border-slate-50 flex gap-4 items-center justify-between">
            <div className="relative flex-1 max-w-md">
               <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
               <input 
                  type="text" 
                  placeholder="Search repository..."
                  className="w-full pl-12 pr-4 py-3 bg-slate-50 border-none rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
               />
            </div>
            <div className="flex gap-2">
               <button onClick={handleNew} className="btn-primary flex items-center gap-2 text-[10px] px-6 py-3">
                  <Plus size={16} /> New {tab === 'coding' ? 'Problem' : 'MCQ'}
               </button>
               <button className="p-3 bg-slate-50 text-slate-500 rounded-2xl hover:bg-slate-100 transition-colors">
                  <Filter size={20} />
               </button>
            </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
           {isLoading ? (
              <div className="p-10 text-center text-slate-400 font-black uppercase tracking-widest animate-pulse">Syncing Repository...</div>
           ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                 {filteredQuestions.map(q => (
                    <div key={q.id} className="group bg-white p-6 rounded-3xl border border-slate-50 shadow-sm hover:border-indigo-100 hover:shadow-xl hover:shadow-indigo-100 transition-all duration-500 flex flex-col">
                       <div className="flex justify-between items-start mb-4">
                          <div className={`p-2 rounded-xl border ${
                             tab === 'coding' ? 'bg-indigo-50 border-indigo-100 text-indigo-600' : 'bg-emerald-50 border-emerald-100 text-emerald-600'
                          }`}>
                             {tab === 'coding' ? <Cpu size={18} /> : <Zap size={18} />}
                          </div>
                          <div className="flex items-center gap-2">
                             <button 
                               onClick={() => handleEdit(q)}
                               className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all"
                             >
                                <Edit3 size={16} />
                             </button>
                             <button className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all">
                                <Trash2 size={16} />
                             </button>
                             <button className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-xl transition-all">
                                <MoreVertical size={16} />
                             </button>
                          </div>
                       </div>

                       <h4 className="text-lg font-black text-slate-900 uppercase tracking-tight line-clamp-2 min-h-[3.5rem] group-hover:text-indigo-600 transition-colors">
                          {q.title || q.question_text}
                       </h4>

                       <div className="mt-6 pt-6 border-t border-slate-50 flex items-center justify-between">
                          <div className="flex items-center gap-3">
                             {tab === 'coding' && (
                                <span className={`text-[9px] font-black uppercase tracking-[0.2em] px-2 py-1 rounded-lg border ${
                                   q.difficulty === 'hard' ? 'bg-red-50 text-red-600 border-red-100' : 
                                   q.difficulty === 'medium' ? 'bg-amber-50 text-amber-600 border-amber-100' : 
                                   'bg-emerald-50 text-emerald-600 border-emerald-100'
                                }`}>
                                   {q.difficulty}
                                </span>
                             )}
                             <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
                                <LayoutGrid size={12} /> {q.points || q.marks} Precision Points
                             </span>
                          </div>
                          <div className="text-[10px] font-black text-slate-300 uppercase tracking-widest">
                             ID: {q.id}
                          </div>
                       </div>
                    </div>
                 ))}
              </div>
           )}
        </div>
      </div>

      {isEditorOpen && (
         <QuestionEditor 
            type={tab} 
            question={selectedQuestion} 
            onClose={() => setIsEditorOpen(false)}
            onSave={fetchResources}
         />
      )}
    </div>
  );
}
