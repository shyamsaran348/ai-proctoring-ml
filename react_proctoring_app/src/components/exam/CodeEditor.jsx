import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { dracula } from '@uiw/codemirror-theme-dracula';
import { python } from '@codemirror/lang-python';
import { javascript } from '@codemirror/lang-javascript';
import { Play, CheckCircle, XCircle, Code2, AlertTriangle, Send } from 'lucide-react';
import api from '../../services/api';
import { useNavigate } from 'react-router-dom';

export default function CodeEditor({ sessionId, problemId }) {
  const [code, setCode] = useState('def solve():\n    # Write your solution here\n    pass');
  const [language, setLanguage] = useState('python');
  const [isExecuting, setIsExecuting] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [stdout, setStdout] = useState(null);
  const [testResults, setTestResults] = useState(null);
  
  const navigate = useNavigate();

  const handleRunCode = async () => {
    setIsExecuting(true);
    setStdout(null);
    setTestResults(null);

    try {
      const res = await api.post(`/execute/`, {
        code,
        language
      });
      
      const { results } = res.data;
      if (results) {
        setTestResults(results);
      } else {
        setStdout("Warning: Backend failed to return formatted test results.");
      }
    } catch (err) {
      setStdout(err.response?.data?.error || "Execution failed. Check your logic and server connection.");
    } finally {
      setIsExecuting(false);
    }
  };

  const handleFinalSubmit = async () => {
    const confirm = window.confirm("Are you ready to submit your exam? You cannot undo this.");
    if (!confirm) return;

    setIsSubmitting(true);
    try {
      // Typically, an assessment suite has a separate mark-complete endpoint
      // We will mimic submitting the code one last time, then ending the session
      await api.post(`/sessions/${sessionId}/submit/`, { code, language });
      // Redirect out
      navigate('/dashboard');
    } catch (err) {
      alert("Failed to finalize exam: " + (err.response?.data?.error || "Unknown error"));
    } finally {
      setIsSubmitting(false);
    }
  };

  const extensions = [language === 'python' ? python() : javascript({ jsx: true })];

  return (
    <div className="flex flex-col h-full bg-[#282a36] rounded-2xl overflow-hidden shadow-2xl border border-gray-800">
       {/* Editor Toolbar */}
       <div className="bg-[#1e1f29] border-b border-gray-800 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
             <div className="flex items-center text-gray-400 bg-[#282a36] px-3 py-1.5 rounded-lg border border-gray-800 font-mono text-sm">
                <Code2 size={16} className="mr-2" />
                <select 
                   value={language}
                   onChange={(e) => setLanguage(e.target.value)}
                   className="bg-transparent outline-none cursor-pointer focus:ring-0 appearance-none"
                >
                   <option value="python">Python 3</option>
                   <option value="javascript">JavaScript</option>
                </select>
             </div>
          </div>
          <div className="flex items-center space-x-3">
             <button 
                onClick={handleRunCode}
                disabled={isExecuting || isSubmitting}
                className="flex items-center px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-800 text-white rounded-lg font-bold text-sm transition-colors shadow-sm"
             >
                <Play size={16} className="mr-2 fill-current" />
                {isExecuting ? 'Running...' : 'Run Tests'}
             </button>
             <button 
                onClick={handleFinalSubmit}
                disabled={isExecuting || isSubmitting}
                className="flex items-center px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800 text-white rounded-lg font-bold text-sm transition-colors shadow-sm"
             >
                <Send size={16} className="mr-2" />
                Submit Assessment
             </button>
          </div>
       </div>

       {/* Editor Window */}
       <div className="flex-1 overflow-auto bg-[#282a36]">
          <CodeMirror
             value={code}
             height="100%"
             theme={dracula}
             extensions={extensions}
             onChange={(val) => setCode(val)}
             className="text-base font-mono h-full"
             style={{ minHeight: '300px' }}
          />
       </div>

       {/* Output Console Bottom Panel */}
       <div className="h-[25vh] min-h-[200px] border-t border-gray-800 bg-[#1e1f29] overflow-y-auto">
          <div className="sticky top-0 bg-[#1e1f29] border-b border-gray-800 px-4 py-2 flex items-center justify-between z-10">
             <h3 className="text-gray-400 font-bold text-xs uppercase tracking-widest flex items-center">
                Terminal Output
             </h3>
          </div>
          
          <div className="p-4 font-mono text-sm">
             {isExecuting ? (
                <div className="text-gray-500 animate-pulse">Running test cases on remote server container...</div>
             ) : (
                <>
                  {stdout && <div className="text-amber-400 whitespace-pre-wrap flex items-start"><AlertTriangle className="mr-2 shrink-0 mt-0.5" size={16}/>{stdout}</div>}
                  
                  {testResults && testResults.length > 0 && (
                     <div className="space-y-4">
                        {testResults.map((tr, idx) => (
                           <div key={idx} className="bg-[#282a36] border border-gray-800 rounded-lg p-4">
                              <div className="flex justify-between items-center mb-3">
                                 <h4 className="text-gray-300 font-bold">{tr.name}</h4>
                                 {tr.passed ? (
                                    <span className="flex items-center text-emerald-400 font-bold"><CheckCircle size={16} className="mr-1"/> Passed</span>
                                 ) : (
                                    <span className="flex items-center text-red-400 font-bold"><XCircle size={16} className="mr-1"/> Failed</span>
                                 )}
                              </div>
                              <div className="grid grid-cols-2 gap-4 text-xs">
                                 <div>
                                    <span className="text-gray-500 block mb-1">Expected Output:</span>
                                    <div className="bg-[#1e1f29] text-gray-300 p-2 rounded">{JSON.stringify(tr.expected)}</div>
                                 </div>
                                 <div>
                                    <span className="text-gray-500 block mb-1">Your Final Output:</span>
                                    <div className="bg-[#1e1f29] text-gray-300 p-2 rounded overflow-hidden">
                                       {tr.error ? <span className="text-red-400">{tr.error}</span> : JSON.stringify(tr.actual)}
                                    </div>
                                 </div>
                              </div>
                           </div>
                        ))}
                     </div>
                  )}
                  
                  {!stdout && !testResults && (
                     <div className="text-gray-600">Waiting for code execution...</div>
                  )}
                </>
             )}
          </div>
       </div>
    </div>
  );
}
