import React, { useState, useEffect } from 'react';
import { X, Save, AlertCircle, Check, Users, Layers, Clock, Shield } from 'lucide-react';
import api from '../../services/api';

export default function CreateExamModal({ onClose, onCreated }) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [duration, setDuration] = useState(60);
  const [visibility, setVisibility] = useState('public');
  const [contestType, setContestType] = useState('coding');
  
  // Problems selection (1 to 3)
  const [problems, setProblems] = useState([]);
  const [selectedProblemIds, setSelectedProblemIds] = useState([]);
  
  // Student enrollment list
  const [students, setStudents] = useState([]);
  const [selectedStudentIds, setSelectedStudentIds] = useState([]);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchProblemsAndStudents();
  }, []);

  const fetchProblemsAndStudents = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Fetch problems
      const probRes = await api.get('/problems/');
      setProblems(probRes.data);
      
      // Fetch students (in a real app we might have /api/students/ or /api/users/)
      // If endpoint fails, we fall back to manual input or standard users
      try {
        const userRes = await api.get('/users/');
        // Filter out staff if possible, or just include all non-staff
        setStudents(userRes.data.filter(u => !u.is_staff));
      } catch (err) {
        // Fallback demo users if no users endpoint
        setStudents([
          { id: 1, username: 'student_shyam', email: 'shyam@example.com' },
          { id: 2, username: 'student_test', email: 'test@example.com' },
          { id: 3, username: 'candidate_alpha', email: 'alpha@example.com' },
        ]);
      }
    } catch (err) {
      setError("Failed to load global resources.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleProblem = (id) => {
    setSelectedProblemIds(prev => {
      if (prev.includes(id)) {
        return prev.filter(x => x !== id);
      } else {
        if (prev.length >= 3) {
          alert("⚠️ Sentinel Rule: An examination node can assign a maximum of 3 problems.");
          return prev;
        }
        return [...prev, id];
      }
    });
  };

  const handleToggleStudent = (username) => {
    setSelectedStudentIds(prev => 
      prev.includes(username) ? prev.filter(u => u !== username) : [...prev, username]
    );
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!title) return setError("Title is required.");
    if (selectedProblemIds.length < 1) return setError("Select at least 1 coding question (Maximum 3).");
    if (selectedProblemIds.length > 3) return setError("Select at maximum 3 coding questions.");

    setIsSaving(true);
    setError(null);

    try {
      // 1. Create Contest
      const contestPayload = {
        title,
        description,
        duration_minutes: parseInt(duration),
        visibility,
        contest_type: contestType,
        status: 'active',
        start_time: new Date().toISOString(),
        end_time: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days active
      };
      
      const contestRes = await api.post('/contests/', contestPayload);
      const contestId = contestRes.data.id;

      // 2. Update structure with selected problems
      const structurePayload = {
        problems: selectedProblemIds.map((pid, idx) => ({ id: pid, order: idx, limit: duration })),
        mcqs: []
      };
      await api.post(`/contests/${contestId}/update_structure/`, structurePayload);

      // 3. Enroll students if invite-only or private
      if (selectedStudentIds.length > 0) {
        await api.post(`/contests/${contestId}/enroll_students/`, {
          student_ids: selectedStudentIds
        });
      }

      alert("Examination Node Deployed successfully!");
      onCreated();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || "Deployment failed.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-8 animate-in fade-in duration-300">
      <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-md" onClick={onClose} />
      
      <div className="relative w-full max-w-4xl bg-white shadow-2xl rounded-[40px] flex flex-col max-h-[85vh] animate-in zoom-in-95 duration-500 overflow-hidden">
        {/* Header */}
        <div className="bg-white px-10 py-6 flex justify-between items-center border-b border-slate-100 shrink-0">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-indigo-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-indigo-100">
              <Layers size={24} />
            </div>
            <div>
              <h2 className="text-xl font-black text-slate-900 uppercase tracking-tight">Deploy Examination Node</h2>
              <p className="text-xs font-semibold text-slate-400 italic">Provision secure assessments, select problems, and register candidates.</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2.5 hover:bg-slate-50 text-slate-400 hover:text-slate-600 rounded-xl transition-all">
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-10 space-y-8 custom-scrollbar">
          {error && (
            <div className="bg-red-50 border border-red-100 rounded-2xl p-4 flex items-center text-red-700 gap-3">
              <AlertCircle size={20} className="shrink-0" />
              <p className="text-sm font-semibold">{error}</p>
            </div>
          )}

          {isLoading ? (
            <div className="py-20 flex justify-center items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
            </div>
          ) : (
            <form onSubmit={handleSave} className="space-y-8">
              {/* Core Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Exam Node Title</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Data Structures Midterm #1"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Duration (Minutes)</label>
                  <input
                    type="number"
                    min="10"
                    max="300"
                    required
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none font-bold"
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Rules & Instructions</label>
                  <textarea
                    placeholder="Describe secure proctor requirements or guidelines..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none h-20 resize-none"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Visibility Tier</label>
                  <select
                    value={visibility}
                    onChange={(e) => setVisibility(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none"
                  >
                    <option value="public">Public (All sector candidates)</option>
                    <option value="private">Private (Invite & Enroll required)</option>
                    <option value="invite_only">Invite Only (Secure access codes)</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Node Type</label>
                  <select
                    value={contestType}
                    onChange={(e) => setContestType(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-medium focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none font-bold text-indigo-600"
                  >
                    <option value="coding">Coding Protocol Only</option>
                  </select>
                </div>
              </div>

              {/* 1 to 3 Problems Checklist */}
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">
                    Select Coding Problems (Min 1, Max 3)
                  </label>
                  <span className="text-[10px] font-black text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-full">
                    {selectedProblemIds.length} of 3 Selected
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
                  {problems.map(prob => {
                    const isSelected = selectedProblemIds.includes(prob.id);
                    return (
                      <div
                        key={prob.id}
                        onClick={() => handleToggleProblem(prob.id)}
                        className={`p-4 rounded-2xl border transition-all duration-300 flex items-center justify-between cursor-pointer ${
                          isSelected
                            ? 'border-indigo-500 bg-indigo-50/20 shadow-md shadow-indigo-100/50'
                            : 'border-slate-200 bg-white hover:border-indigo-200'
                        }`}
                      >
                        <div>
                          <p className="text-xs font-black text-slate-800 uppercase tracking-tight">{prob.title}</p>
                          <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest mt-0.5">{prob.difficulty} • {prob.points || 100} pts</p>
                        </div>
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center border transition-all ${
                          isSelected ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-slate-300'
                        }`}>
                          {isSelected && <Check size={14} />}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Candidates (Students) Selection */}
              <div className="space-y-4">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">
                  Select Candidates to Enroll ({selectedStudentIds.length} Chosen)
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-h-[160px] overflow-y-auto pr-2 custom-scrollbar">
                  {students.map(student => {
                    const isSelected = selectedStudentIds.includes(student.username);
                    return (
                      <div
                        key={student.id}
                        onClick={() => handleToggleStudent(student.username)}
                        className={`p-3 rounded-xl border transition-all duration-300 flex items-center gap-3 cursor-pointer ${
                          isSelected
                            ? 'border-indigo-500 bg-indigo-50/20'
                            : 'border-slate-200 bg-white hover:border-indigo-200'
                        }`}
                      >
                        <div className={`w-4 h-4 rounded-md flex items-center justify-center border transition-all ${
                          isSelected ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-slate-300'
                        }`}>
                          {isSelected && <Check size={10} />}
                        </div>
                        <div className="min-w-0">
                          <p className="text-xs font-black text-slate-800 tracking-tight truncate">{student.username}</p>
                          <p className="text-[9px] font-semibold text-slate-400 truncate">{student.email}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Save/Deploy */}
              <div className="flex justify-end pt-4 border-t border-slate-100 shrink-0">
                <button
                  type="submit"
                  disabled={isSaving}
                  className="px-8 py-3.5 bg-indigo-600 text-white rounded-2xl font-black text-xs uppercase tracking-widest hover:bg-slate-900 transition-all shadow-lg shadow-indigo-100 flex items-center gap-3 active:scale-95 disabled:opacity-50"
                >
                  <Save size={16} />
                  {isSaving ? 'Deploying Node...' : 'Deploy Node'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
