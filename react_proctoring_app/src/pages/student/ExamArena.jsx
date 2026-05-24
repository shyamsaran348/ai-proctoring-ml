import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../../services/api';
import PreFlightCheck from '../../components/exam/PreFlightCheck';
import WebcamWidget from '../../components/exam/WebcamWidget';
import ExamAIBriefing from '../../components/exam/ExamAIBriefing';
import SafeZoneHud from '../../components/exam/SafeZoneHud';
import CodeEditor from '../../components/exam/CodeEditor';
import { Shield, BrainCircuit, Activity, Clock, AlertTriangle, Lock, EyeOff, BookOpen } from 'lucide-react';

export default function ExamArena() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  const [sessionData, setSessionData] = useState(null);
  const [problemsList, setProblemsList] = useState([]);
  const [activeProblemId, setActiveProblemId] = useState(null);
  const [problem, setProblem] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isVerified, setIsVerified] = useState(false);
  
  // High-level Risk Management State (bubbles up from Webcam Widget)
  const [riskData, setRiskData] = useState({ risk: 0, violation: null, components: {} });
  const [cameraConnected, setCameraConnected] = useState(true);
  const [timeLeft, setTimeLeft] = useState(3600); // Default 60 mins
  const [securityLogs, setSecurityLogs] = useState([]);
  
  // Sentinel Enforcement State
  const [isLocked, setIsLocked] = useState(false);
  const [lockReason, setLockReason] = useState(null);
  const [isPaused, setIsPaused] = useState(false);

  // ─── Proctored Action Handlers ───
  const formatTime = useCallback((seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }, []);

  const triggerIronCurtain = useCallback((reason) => {
    setIsLocked(true);
    setLockReason(reason);
  }, []);

  const reportViolation = useCallback(async (type) => {
    if (!sessionId) return;
    console.log(`Security Protocol Violation: ${type}`);
    setSecurityLogs(prev => [{
      id: Date.now(),
      type: type,
      timestamp: new Date().toLocaleTimeString(),
      severity: 'high'
    }, ...prev].slice(0, 5));

    try {
      await api.post(`/sessions/${sessionId}/frame/`, {
        primary_violation: type,
        risk_score: 1.0, 
        risk_components: { [type.toLowerCase()]: 1 }
      });
    } catch (err) {
      console.error("Failed to report high-priority security event.", err);
    }
  }, [sessionId]);

  const processCommand = useCallback((cmd) => {
    console.log(`FACULTY COMMAND RECEIVED: ${cmd}`);
    if (cmd === 'WARN') {
      alert("⚠️ SENTINEL OVERWATCH: Behavioral Warning Issued by Faculty.");
    } else if (cmd === 'PAUSE') {
      setIsPaused(true);
      triggerIronCurtain('FACULTY_PAUSED_SESSION');
    } else if (cmd === 'RESUME') {
      setIsPaused(false);
      setIsLocked(false);
      setLockReason(null);
    } else if (cmd === 'TERMINATE') {
      navigate('/dashboard', { state: { error: 'Your session was terminated by a proctor.' } });
    }
  }, [triggerIronCurtain, navigate]);

  const handleRiskUpdate = useCallback((data) => {
    setRiskData(data);
    setCameraConnected(true);
    if (data.last_command && data.last_command !== 'NONE') {
      processCommand(data.last_command);
    }
  }, [processCommand]);

  const handleCameraError = useCallback((errMsg) => {
    setCameraConnected(false);
    console.error(errMsg);
  }, []);

  const handleAutoSubmit = useCallback(() => {
    console.log("Time expired. Triggering auto-submit...");
    handleFinalizeAttempt(true);
  }, []);

  const handleBiometricComplete = useCallback(async () => {
    setIsVerified(true);
  }, []);

  // ─── Effects & Lifecycle ───
  useEffect(() => {
    const fetchSessionInfo = async () => {
      try {
        const res = await api.get(`/sessions/${sessionId}/`);
        setSessionData(res.data);
        setTimeLeft(res.data.time_remaining || 3600);
        
        // Load the problems list
        if (res.data.problems && res.data.problems.length > 0) {
          setProblemsList(res.data.problems);
          setActiveProblemId(res.data.problems[0].id);
        } else if (res.data.problem) {
          const singleProb = { id: res.data.problem.id, title: res.data.problem.title };
          setProblemsList([singleProb]);
          setActiveProblemId(singleProb.id);
        }
      } catch (err) {
        alert("Session invalid or expired.");
        navigate('/dashboard');
      } finally {
        setIsLoading(false);
      }
    };
    fetchSessionInfo();
  }, [sessionId, navigate]);

  // Load active problem details
  useEffect(() => {
    if (!activeProblemId) return;
    const fetchProblemDetails = async () => {
      try {
        const probRes = await api.get(`/problems/${activeProblemId}/`);
        setProblem(probRes.data);
      } catch (err) {
        console.error("Failed to load problem details.");
      }
    };
    fetchProblemDetails();
  }, [activeProblemId]);

  useEffect(() => {
    if (!isVerified || timeLeft <= 0) return;
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          clearInterval(timer);
          handleAutoSubmit();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [isVerified, timeLeft, handleAutoSubmit]);

  useEffect(() => {
    if (!isVerified) return;
    const handleVisibilityChange = () => {
      if (document.hidden) {
        reportViolation('TAB_SWITCH');
      }
    };
    const handleBlur = () => {
      reportViolation('WINDOW_BLUR');
    };
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) {
        reportViolation('FULLSCREEN_EXIT');
      }
    };
    window.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('blur', handleBlur);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    if (document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
    return () => {
      window.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [isVerified, reportViolation]);

  const handleFinalizeAttempt = async (isAuto = false) => {
    if (!isAuto) {
      const confirm = window.confirm("Are you ready to submit your exam attempt? This will end all problem streams and secure AI proctoring immediately.");
      if (!confirm) return;
    }

    try {
      await api.post(`/sessions/${sessionId}/submit/`, {
        finalize: true
      });
      alert("🎉 Secure Assessment Node Finalized. Your attempts have been saved.");
      navigate('/dashboard');
    } catch (err) {
      alert("Failed to finalize session: " + (err.response?.data?.error || "Unknown error"));
    }
  };

  if (isLoading) {
    return <div className="min-h-screen flex items-center justify-center bg-gray-50"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div></div>;
  }

  if (!isVerified) {
    return <PreFlightCheck sessionId={sessionId} onVerified={handleBiometricComplete} />;
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col font-sans relative">
       {/* Sentinel Boundary Feedback - Global Overlay */}
       <div className={`fixed inset-0 pointer-events-none transition-all duration-1000 z-50 ${riskData.risk >= 0.7 ? 'bg-red-500/10' : 'bg-transparent'}`}>
          {riskData.risk >= 0.9 && (
             <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-600 text-white px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-[0.3em] flex flex-col items-center gap-1 shadow-2xl animate-pulse">
                <div className="flex items-center gap-2">
                   <AlertTriangle size={14} /> Critical Risk - Monitor Intervening
                </div>
                <div className="text-[8px] opacity-80 tracking-widest bg-black/20 px-2 py-0.5 rounded-full">
                   Reason: {riskData.violation?.replace(/_/g, ' ') || 'Sustained Anomaly'}
                </div>
             </div>
          )}
       </div>

       {/* Iron Curtain: Focus Lock Modal */}
       {isLocked && (
          <div className="fixed inset-0 z-[100] bg-slate-900/90 backdrop-blur-md flex items-center justify-center p-6 animate-in fade-in duration-300">
             <div className="bg-white max-w-md w-full rounded-3xl p-10 shadow-3xl text-center border border-slate-200">
                <div className="w-20 h-20 bg-red-100 text-red-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-sm">
                   {isPaused ? <Lock size={40} /> : <EyeOff size={40} />}
                </div>
                <h2 className="text-2xl font-black text-slate-900 mb-2 uppercase tracking-tight">
                   {isPaused ? 'Exam Paused' : 'Sentinel Focus Lock'}
                </h2>
                <p className="text-slate-500 font-medium mb-8 leading-relaxed">
                   {isPaused 
                     ? 'A proctor has manually paused your session. Please wait for the "RESUME" signal.'
                     : 'Focus integrity was lost. Switching tabs or windows is prohibited during the Secure Sentinel session.'}
                </p>
                
                <div className="bg-red-50 border border-red-100 rounded-xl px-4 py-3 mb-8">
                   <p className="text-[10px] font-black text-red-600 uppercase tracking-widest mb-1">Incident Token</p>
                   <code className="text-xs font-mono font-bold text-red-700">{lockReason}</code>
                </div>

                {!isPaused && (
                   <button 
                      onClick={() => { setIsLocked(false); setLockReason(null); }}
                      className="w-full bg-slate-900 text-white py-4 rounded-2xl font-black text-sm uppercase tracking-widest hover:bg-slate-800 transition-all active:scale-95 shadow-xl shadow-slate-200"
                   >
                      Return to Arena
                   </button>
                )}
             </div>
          </div>
       )}

       {/* Arena Navbar */}
       <header className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center shadow-md grow-0 z-20">
          <div className="flex items-center space-x-3">
             <div className="bg-indigo-600 p-2.5 rounded-xl text-white shadow-lg shadow-indigo-100">
                <BrainCircuit size={24} />
             </div>
             <div>
               <h1 className="text-xl font-black text-slate-800 tracking-tight leading-tight uppercase">
                 {sessionData?.contest?.title || 'Secure Assessment Node'}
               </h1>
               <div className="flex items-center gap-2 mt-0.5">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                  <p className="text-[10px] text-emerald-600 font-black uppercase tracking-[0.2em]">Enforcement Active</p>
               </div>
             </div>
          </div>
          
          {/* Central Timer HUD */}
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center bg-slate-900 text-white px-6 py-2 rounded-2xl shadow-xl border border-slate-700">
            <Clock size={16} className="mr-3 text-indigo-400" />
            <span className="font-mono text-xl font-black tracking-widest">{formatTime(timeLeft)}</span>
          </div>

          <div className="flex items-center space-x-4">
             <button 
               className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-xl font-black text-xs uppercase tracking-widest shadow-lg shadow-indigo-100 transition-all active:scale-95"
               onClick={() => handleFinalizeAttempt(false)}
             >
               Finalize Attempt
             </button>
             <div className="flex items-center bg-slate-50 text-slate-500 font-mono text-[10px] px-3 py-1.5 rounded-lg border border-slate-200">
               ID: {sessionId.split('-')[0]}
             </div>
          </div>
       </header>

       {/* Problems Switcher Tabs (For Multi-problem contests) */}
       {problemsList.length > 1 && (
         <div className="bg-white border-b border-slate-200 px-6 py-3 flex gap-3 z-10 shadow-sm shrink-0">
           {problemsList.map((p, idx) => {
             const isActive = activeProblemId === p.id;
             return (
               <button
                 key={p.id}
                 onClick={() => setActiveProblemId(p.id)}
                 className={`px-5 py-2.5 rounded-xl text-xs font-black uppercase tracking-wider transition-all duration-300 ${
                   isActive
                     ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-100'
                     : 'bg-slate-50 border border-slate-200 text-slate-600 hover:bg-slate-100'
                 }`}
               >
                 Problem {idx + 1}: {p.title}
               </button>
             );
           })}
         </div>
       )}

       {/* Workspace Matrix */}
       <main className="flex-1 flex overflow-hidden">
          
          {/* Left Panel: Problem Descriptor & AI Security Stack */}
          <div className="w-[450px] bg-white border-r border-gray-200 flex flex-col shadow-lg shrink-0 overflow-y-auto">
             
             {/* Problem Content */}
             <div className="p-8 border-b border-gray-100 flex-1">
                <h2 className="text-2xl font-extrabold text-gray-900 mb-6 flex items-center">
                   <Shield className="mr-3 text-emerald-500" size={28}/> Task Descriptor
                </h2>
                {problem ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-black text-slate-900 uppercase tracking-tight">{problem.title}</h3>
                      <p className="text-[10px] font-black text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-full inline-block mt-1 uppercase tracking-wider">{problem.difficulty} • {problem.points} points</p>
                    </div>
                    <div className="prose prose-indigo prose-sm text-gray-700 leading-relaxed font-medium whitespace-pre-wrap">
                       {problem.problem_statement || problem.description}
                    </div>
                    {problem.input_format && (
                      <div>
                        <h4 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-1">Input Format</h4>
                        <p className="text-xs font-medium text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-100">{problem.input_format}</p>
                      </div>
                    )}
                    {problem.output_format && (
                      <div>
                        <h4 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-1">Output Format</h4>
                        <p className="text-xs font-medium text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-100">{problem.output_format}</p>
                      </div>
                    )}
                    {problem.sample_input && (
                      <div>
                        <h4 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-1">Sample Input</h4>
                        <pre className="text-xs font-mono bg-slate-900 text-slate-100 p-3 rounded-xl overflow-x-auto">{problem.sample_input}</pre>
                      </div>
                    )}
                    {problem.sample_output && (
                      <div>
                        <h4 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-1">Sample Output</h4>
                        <pre className="text-xs font-mono bg-slate-900 text-slate-100 p-3 rounded-xl overflow-x-auto">{problem.sample_output}</pre>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-20 text-center text-slate-400 animate-pulse">Loading problem descriptor...</div>
                )}
             </div>

             {/* AI Proctoring Stack Widget */}
             <div className="bg-gray-50 p-6 flex flex-col space-y-6 shrink-0 relative">
                <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent"></div>
                
                <h3 className="font-black text-slate-400 text-[10px] tracking-widest uppercase mb-1 flex items-center gap-2">
                   <Activity size={14} className="text-indigo-500" /> Live Dynamics Stack
                </h3>
                
                <WebcamWidget 
                   sessionId={sessionId} 
                   onRiskUpdate={handleRiskUpdate} 
                   onError={handleCameraError} 
                />

                <ExamAIBriefing 
                   riskData={riskData} 
                   connected={cameraConnected} 
                />

                <SafeZoneHud 
                   riskScore={riskData.risk} 
                   violationType={riskData.violation} 
                   connected={cameraConnected} 
                />

                {/* Security activity log */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                   <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 flex items-center justify-between">
                      Security History
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                   </h4>
                   <div className="space-y-2 max-h-[160px] overflow-y-auto pr-1">
                      {securityLogs.length === 0 ? (
                         <div className="text-[10px] text-slate-400 italic font-medium px-2 py-4 bg-white rounded-xl border border-dashed border-slate-200 text-center">
                            No incidents recorded in current duration.
                         </div>
                      ) : (
                         securityLogs.map(log => (
                            <div key={log.id} className="flex items-center justify-between bg-white px-3 py-2.5 rounded-xl border border-slate-100 shadow-sm animate-in slide-in-from-left-2">
                               <div className="flex items-center gap-2">
                                  <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></div>
                                  <span className="text-[10px] font-black text-slate-700 uppercase tracking-tight">{log.type}</span>
                                </div>
                               <span className="text-[9px] text-slate-400 font-mono font-bold">{log.timestamp}</span>
                            </div>
                         ))
                      )}
                   </div>
                </div>
             </div>
          </div>

          {/* Right Panel: The Code Arena */}
          <div className="flex-1 p-6 flex flex-col">
             <CodeEditor sessionId={sessionId} problemId={activeProblemId} />
          </div>

       </main>
    </div>
  );
}
