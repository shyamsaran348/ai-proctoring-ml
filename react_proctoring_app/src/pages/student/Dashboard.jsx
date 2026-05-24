import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import { 
  LogOut, 
  BookOpen, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  Shield, 
  Trophy, 
  User as UserIcon,
  Activity,
  History,
  ChevronRight
} from 'lucide-react';

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [exams, setExams] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    integrity: "98%"
  });

  useEffect(() => {
    fetchExams();
  }, [user]);

  const fetchExams = async () => {
    try {
       setIsLoading(true);
       setError(null);
       const res = await api.get('/contests/', {
         params: { student_id: user?.username || user?.email }
       });
       setExams(res.data);
       
       // Calculate basic stats
       const comp = res.data.filter(e => e.status === 'completed').length;
       setStats(prev => ({
         ...prev,
         total: res.data.length,
         completed: comp
       }));
    } catch (err) {
       setError("Failed to load assessments from server.");
    } finally {
       setIsLoading(false);
    }
  };

  const checkAndStartExam = async (examId) => {
    try {
      const res = await api.post(`/contests/${examId}/start_exam/`, {
        student_id: user?.email || user?.username,
      });
      const sessionData = res.data;
      navigate(`/arena/${sessionData.session_id}`);
    } catch (err) {
      alert(err.response?.data?.error || "Failed to start session.");
    }
  };


  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navbar */}
      <nav className="bg-slate-900 border-b border-slate-800 px-8 py-5 flex justify-between items-center sticky top-0 z-50">
        <div className="flex items-center space-x-4">
          <div className="bg-indigo-600 p-2.5 rounded-xl text-white shadow-lg shadow-indigo-900/50">
             <Shield size={24} />
          </div>
          <div>
            <h1 className="text-xl font-black text-white tracking-widest uppercase italic">DevProctor</h1>
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Global Node: Active</p>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-6">
          <div className="flex items-center gap-4 bg-slate-800/50 px-4 py-2 rounded-2xl border border-slate-700">
             <div className="w-8 h-8 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400 border border-indigo-500/30">
                <UserIcon size={16} />
             </div>
             <p className="text-sm font-black text-slate-200 tracking-tight">{user?.username || user?.email}</p>
          </div>
          <button 
            onClick={logout}
            className="p-2.5 text-slate-400 hover:text-red-400 hover:bg-red-400/10 rounded-xl transition-all active:scale-90"
            title="Log Out"
          >
            <LogOut size={20} />
          </button>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto px-8 py-10">
        {/* Profile Performance Header */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
           <div className="lg:col-span-1 bg-white rounded-3xl p-8 border border-slate-100 shadow-xl shadow-slate-200/50">
              <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center text-white mb-6 shadow-lg shadow-indigo-100">
                 <Trophy size={32} />
              </div>
              <h2 className="text-sm font-black text-slate-400 uppercase tracking-[0.2em] mb-1">Academic Rank</h2>
              <p className="text-4xl font-black text-slate-900 mb-1">Senior</p>
              <p className="text-xs font-bold text-indigo-600 uppercase">Pre-Validation Tier</p>
           </div>
           
           <div className="lg:col-span-3 bg-slate-900 rounded-3xl p-8 text-white grid grid-cols-1 md:grid-cols-3 gap-8 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-600/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
              
              <div className="relative">
                 <h3 className="text-[10px] font-black uppercase text-indigo-400 tracking-widest mb-4 flex items-center gap-2">
                    <Activity size={14} /> System Integrity
                 </h3>
                 <p className="text-5xl font-black mb-2">{stats.integrity}</p>
                 <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-emerald-500 h-full w-[98%] shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                 </div>
              </div>
              
              <div className="relative border-l border-slate-800 pl-8">
                 <h3 className="text-[10px] font-black uppercase text-slate-400 tracking-widest mb-4 flex items-center gap-2">
                    <BookOpen size={14} /> Assessments
                 </h3>
                 <p className="text-5xl font-black mb-1">{stats.total}</p>
                 <p className="text-xs font-bold text-slate-500 uppercase">Total Published</p>
              </div>

              <div className="relative border-l border-slate-800 pl-8">
                 <h3 className="text-[10px] font-black uppercase text-emerald-400 tracking-widest mb-4 flex items-center gap-2">
                    <CheckCircle size={14} /> Completed
                 </h3>
                 <p className="text-5xl font-black mb-1 text-emerald-400">{stats.completed}</p>
                 <p className="text-xs font-bold text-slate-500 uppercase">Successfully Verified</p>
              </div>
           </div>
        </div>

        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-black text-slate-900 tracking-tight uppercase">Live Assessments</h2>
            <p className="text-slate-500 font-medium text-sm mt-1 italic">Authorized challenges requiring biometric validation.</p>
          </div>
          <div className="flex gap-2">
             <div onClick={fetchExams} className="bg-white p-2 rounded-xl border border-slate-100 shadow-sm text-slate-400 hover:text-indigo-600 transition-colors cursor-pointer">
                <History size={20} />
              </div>
          </div>
        </div>

        {error && (
          <div className="mb-8 bg-red-50 p-4 rounded-xl border border-red-100 flex items-center">
             <AlertTriangle className="text-red-500 mr-3" />
             <p className="text-red-800 font-medium">{error}</p>
          </div>
        )}

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[1, 2, 3].map(i => (
              <div key={i} className="bg-white rounded-3xl p-8 shadow-sm border border-slate-100 animate-pulse h-64">
                <div className="h-4 bg-slate-100 rounded-full w-1/4 mb-6"></div>
                <div className="h-8 bg-slate-50 rounded-xl w-3/4 mb-4"></div>
                <div className="h-4 bg-slate-50 rounded-lg w-full mb-2"></div>
                <div className="h-4 bg-slate-50 rounded-lg w-5/6 mb-12"></div>
                <div className="h-14 bg-slate-100 rounded-2xl w-full mt-auto"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {exams.length === 0 ? (
               <div className="col-span-full py-24 text-center text-slate-400 bg-white rounded-[40px] border border-dashed border-slate-200">
                 <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6">
                    <BookOpen size={40} className="text-slate-200" />
                 </div>
                 <p className="text-xl font-black uppercase tracking-widest leading-none">Vortex Status: Empty</p>
                 <p className="text-sm font-medium mt-2">No active assessments found in your sectors.</p>
               </div>
            ) : (
              exams.map(exam => (
                <div 
                  key={exam.id} 
                  className={`bg-white rounded-[32px] p-8 border transition-all duration-500 flex flex-col group relative overflow-hidden ${
                    exam.status === 'completed' 
                    ? 'border-slate-100 opacity-70 bg-slate-50 grayscale' 
                    : 'border-slate-100 hover:border-indigo-500 hover:shadow-2xl hover:shadow-indigo-200/50 hover:-translate-y-2'
                  }`}
                >
                  {exam.status !== 'completed' && (
                    <div className="absolute top-0 right-0 px-4 py-1.5 bg-indigo-600 text-white text-[10px] font-black uppercase tracking-widest rounded-bl-2xl opacity-0 group-hover:opacity-100 transition-opacity">
                      Secure Session
                    </div>
                  )}

                  <div className="flex justify-between items-start mb-6">
                     <span className="text-[10px] font-black text-indigo-600 uppercase tracking-[0.3em] bg-indigo-50 px-3 py-1 rounded-full">
                        {exam.status === 'completed' ? 'Redacted' : 'Active Channel'}
                     </span>
                     {exam.status === 'completed' && <CheckCircle className="text-emerald-500" size={24} />}
                  </div>
                  
                  <h3 className="text-2xl font-black text-slate-900 mb-3 group-hover:text-indigo-600 transition-colors uppercase leading-none">{exam.title}</h3>
                  <p className="text-slate-500 text-sm font-medium mb-10 line-clamp-3 leading-relaxed">
                    {exam.description || 'Access protocols not defined for this specific assessment tier.'}
                  </p>
                  
                  <div className="mt-auto space-y-4">
                    <button
                      onClick={() => checkAndStartExam(exam.id)}
                      disabled={exam.status === 'completed'}
                      className={`w-full py-4 px-6 rounded-2xl font-black text-xs uppercase tracking-widest transition-all shadow-md group-hover:shadow-indigo-200 ${
                        exam.status === 'completed'
                          ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                          : 'bg-indigo-600 text-white hover:bg-slate-900 flex justify-center items-center gap-3'
                      }`}
                    >
                      {exam.status === 'completed' ? 'Validation Complete' : (
                        <>
                          Establish Uplink <ChevronRight size={16} className="group-hover:translate-x-1 transition-transform" />
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

      </main>
    </div>
  );
}
