import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import api from '../../services/api';
import { 
  ShieldAlert, 
  LayoutDashboard, 
  BookOpen, 
  Database, 
  Settings, 
  LogOut,
  Activity,
  UserCheck,
  FileWarning,
  Plus,
  Radar
} from 'lucide-react';
import LiveAlertStream from '../../components/faculty/LiveAlertStream';
import ViolationGallery from '../../components/faculty/ViolationGallery';
import AssessmentVault from '../../components/faculty/AssessmentVault';
import QuestionBank from '../../components/faculty/QuestionBank';
import SessionDeepDive from '../../components/faculty/SessionDeepDive';
import ExamDesigner from '../../components/faculty/ExamDesigner';

export default function CommandCenter() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('overwatch'); // overwatch, assessments, repository
  const [selectedSession, setSelectedSession] = useState(null);
  const [editingContestId, setEditingContestId] = useState(null);
  const [isStatsSyncing, setIsStatsSyncing] = useState(false);
  const [liveStats, setLiveStats] = useState({
    active_sessions: 0,
    verified_students: 0,
    recent_anomalies: 0,
    total_problems: 0,
    total_mcqs: 0
  });

  useEffect(() => {
    const fetchStats = async () => {
      setIsStatsSyncing(true);
      try {
        const res = await api.get('/dashboard_stats/');
        setLiveStats(res.data);
      } catch (err) {
        console.error("Fusion stats sync failed.");
      } finally {
        setTimeout(() => setIsStatsSyncing(false), 800);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 30000); // Sync every 30s
    return () => clearInterval(interval);
  }, []);

  const stats = [
    { label: 'Active Sessions', value: liveStats.active_sessions, icon: Activity, color: 'text-indigo-600', bg: 'bg-indigo-50' },
    { label: 'Verified Students', value: liveStats.verified_students, icon: UserCheck, color: 'text-emerald-600', bg: 'bg-emerald-50' },
    { label: 'Anomalies (24h)', value: liveStats.recent_anomalies, icon: FileWarning, color: 'text-rose-600', bg: 'bg-rose-50' },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'overwatch':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full animate-in fade-in duration-500">
            <div className="lg:col-span-2 h-full flex flex-col gap-8 text-slate-800">
               <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                     <Activity size={14} className={isStatsSyncing ? 'animate-pulse text-indigo-600' : ''} />
                     Live Telemetry {isStatsSyncing && <span className="text-[10px] lowercase font-medium text-slate-300 italic opacity-50">Syncing...</span>}
                  </h3>
               </div>
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {stats.map((stat, i) => (
                    <div key={i} className="bg-white p-6 rounded-[32px] border border-slate-100 shadow-sm flex items-center gap-5 group hover:shadow-xl hover:shadow-slate-200 transition-all duration-500">
                       <div className={`w-14 h-14 ${stat.bg} ${stat.color} rounded-2xl flex items-center justify-center transition-transform group-hover:scale-110`}>
                          <stat.icon size={24} />
                       </div>
                       <div>
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none mb-1">{stat.label}</p>
                          <p className="text-2xl font-black text-slate-900 leading-none">{stat.value}</p>
                       </div>
                    </div>
                  ))}
               </div>
               <div className="flex-1 min-h-0 bg-white rounded-[32px] border border-slate-100 shadow-sm overflow-hidden flex flex-col">
                  <ViolationGallery />
               </div>
            </div>
            <div className="h-full min-h-0">
              <LiveAlertStream onDeepDive={setSelectedSession} />
            </div>
          </div>
        );
      case 'assessments':
        return <AssessmentVault onOpenDesigner={setEditingContestId} />;
      case 'repository':
        return <QuestionBank />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex overflow-hidden font-sans selection:bg-indigo-100 selection:text-indigo-900">
      
      {/* Sidebar Navigation */}
      <aside className="w-80 bg-white border-r border-slate-200 flex flex-col p-8 z-20 shadow-2xl shadow-slate-200/50">
         <div className="flex items-center gap-4 mb-12">
            <div className="w-12 h-12 bg-slate-900 rounded-2xl flex items-center justify-center text-white shadow-xl shadow-slate-200 rotate-3 transition-transform hover:rotate-0 duration-500">
               <Radar size={28} />
            </div>
            <div>
               <h1 className="text-xl font-black tracking-tighter text-slate-900 uppercase italic">Sentinel</h1>
               <p className="text-[8px] font-black text-indigo-600 uppercase tracking-widest text-center bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-100">Faculty Core</p>
            </div>
         </div>

         <nav className="flex-1 space-y-2">
            <NavItem 
               icon={LayoutDashboard} 
               label="Live Overwatch" 
               active={activeTab === 'overwatch'} 
               onClick={() => setActiveTab('overwatch')} 
            />
            <NavItem 
               icon={BookOpen} 
               label="Assessment Vault" 
               active={activeTab === 'assessments'} 
               onClick={() => setActiveTab('assessments')} 
            />
            <NavItem 
               icon={Database} 
               label="Question Bank" 
               active={activeTab === 'repository'} 
               onClick={() => setActiveTab('repository')} 
            />
            <div className="pt-8 pb-4">
               <p className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] px-4">System Settings</p>
            </div>
            <NavItem icon={Settings} label="Core Configuration" />
         </nav>

         <div className="mt-auto pt-8 border-t border-slate-100">
            <div className="bg-slate-50 rounded-3xl p-4 flex items-center gap-4 mb-6 group cursor-pointer hover:bg-slate-100 transition-colors">
               <div className="w-10 h-10 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                  <img src={`https://ui-avatars.com/api/?name=${user?.username || 'Professor'}&background=0f172a&color=fff`} alt="Avatar" />
               </div>
               <div className="flex-1 min-w-0">
                  <p className="text-xs font-black text-slate-900 uppercase truncate">{user?.username || 'Admin Faculty'}</p>
                  <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Level 1 Proctor</p>
               </div>
               <button onClick={logout} className="p-1 hover:text-rose-500 transition-colors">
                  <LogOut size={16} />
               </button>
            </div>
            <p className="text-[8px] font-black text-slate-300 uppercase tracking-widest text-center opacity-50 font-mono">Build v2.4.0-Stable // Fusion-8</p>
         </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-10 h-screen overflow-hidden flex flex-col relative bg-slate-50">
         {renderContent()}

         {/* Interstitial Search/Action Bar */}
         <div className="fixed bottom-10 right-10 flex gap-4 animate-in slide-in-from-bottom-8 duration-1000">
            <button className="w-14 h-14 bg-slate-900 text-white rounded-2xl shadow-2xl flex items-center justify-center hover:scale-110 active:scale-95 transition-all group relative">
               <Plus size={28} />
               <div className="absolute right-full mr-4 px-3 py-1.5 bg-slate-900 text-white text-[10px] font-black uppercase tracking-widest rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none shadow-xl border border-white/10 backdrop-blur-sm">
                  Rapid Resource Injection
               </div>
            </button>
         </div>
      </main>

      {/* Modals & Overlays */}
      {selectedSession && (
        <SessionDeepDive 
           sessionId={selectedSession} 
           onClose={() => setSelectedSession(null)} 
        />
      )}

      {editingContestId && (
        <ExamDesigner 
           contestId={editingContestId} 
           onClose={() => setEditingContestId(null)} 
        />
      )}

    </div>
  );
}

function NavItem({ icon: Icon, label, active, onClick }) {
  return (
    <button 
      onClick={onClick}
      className={`w-full flex items-center gap-4 px-6 py-4 rounded-2xl transition-all duration-300 group ${
        active 
        ? 'bg-slate-900 text-white shadow-xl shadow-slate-300' 
        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
      }`}
    >
      <Icon size={20} className={`transition-transform group-hover:scale-110 ${active ? 'text-indigo-400' : ''}`} />
      <span className={`text-xs font-black uppercase tracking-widest ${active ? 'opacity-100' : 'opacity-80'}`}>{label}</span>
      {active && (
         <div className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-400 shadow-[0_0_8px_rgba(129,140,248,0.8)]"></div>
      )}
    </button>
  );
}
