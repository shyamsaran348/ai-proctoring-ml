import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldCheck, 
  Wifi, 
  Camera, 
  Mic, 
  AlertCircle, 
  CheckCircle, 
  ChevronRight,
  Info
} from 'lucide-react';
import BiometricOnboarding from './BiometricOnboarding';

export default function PreFlightCheck({ sessionId, onVerified }) {
  const [currentStep, setCurrentStep] = useState(0); // 0: Connectivity, 1: Hardware, 2: Rules, 3: Biometric
  const [checks, setChecks] = useState({
    network: { status: 'idle', latency: null },
    camera: { status: 'idle' },
    mic: { status: 'idle' },
    fullscreen: { status: 'idle' }
  });
  const [error, setError] = useState(null);

  // Step 0: Network Check
  useEffect(() => {
    if (currentStep === 0) {
      runNetworkCheck();
    } else if (currentStep === 1) {
      runHardwareCheck();
    }
  }, [currentStep]);

  const runNetworkCheck = async () => {
    setChecks(prev => ({ ...prev, network: { ...prev.network, status: 'loading' } }));
    const startTime = Date.now();
    try {
      // Small ping to check latency
      await fetch('/api/health-check/', { method: 'HEAD' });
      const latency = Date.now() - startTime;
      setChecks(prev => ({ 
        ...prev, 
        network: { status: 'success', latency } 
      }));
    } catch (err) {
      setChecks(prev => ({ ...prev, network: { ...prev.network, status: 'error' } }));
    }
  };

  // Step 1: Hardware Check
  const runHardwareCheck = async () => {
    setChecks(prev => ({ 
      ...prev, 
      camera: { status: 'loading' },
      mic: { status: 'loading' }
    }));
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      stream.getTracks().forEach(track => track.stop());
      setChecks(prev => ({ 
        ...prev, 
        camera: { status: 'success' },
        mic: { status: 'success' }
      }));
    } catch (err) {
      setChecks(prev => ({ 
        ...prev, 
        camera: { status: 'error' },
        mic: { status: 'error' }
      }));
      setError("Hardware access denied. Please enable camera and microphone permissions.");
    }
  };

  const nextStep = () => {
    if (currentStep === 0 && checks.network.status === 'success') setCurrentStep(1);
    if (currentStep === 1) runHardwareCheck().then(() => {
        if (checks.camera.status === 'success') setCurrentStep(2);
    });
    if (currentStep === 2) setCurrentStep(3);
  };

  if (currentStep === 3) {
    return <BiometricOnboarding sessionId={sessionId} onVerified={onVerified} />;
  }

  return (
    <div className="fixed inset-0 bg-slate-900 flex items-center justify-center p-4 z-50 overflow-y-auto">
      <div className="bg-white rounded-3xl shadow-2xl max-w-2xl w-full overflow-hidden flex flex-col md:flex-row min-h-[500px]">
        {/* Progress Sidebar */}
        <div className="bg-indigo-600 p-8 md:w-1/3 text-white flex flex-col">
          <div className="mb-auto">
            <ShieldCheck size={40} className="mb-6 opacity-90" />
            <h2 className="text-2xl font-bold mb-8">System Readiness</h2>
            
            <div className="space-y-6">
              {[
                { label: 'Connectivity', icon: <Wifi size={18} />, step: 0 },
                { label: 'Hardware', icon: <Camera size={18} />, step: 1 },
                { label: 'Environment', icon: <Info size={18} />, step: 2 },
                { label: 'Identity', icon: <Camera size={18} />, step: 3 },
              ].map((item, idx) => (
                <div 
                  key={idx} 
                  className={`flex items-center gap-3 transition-opacity ${currentStep >= item.step ? 'opacity-100' : 'opacity-40'}`}
                >
                  <div className={`p-2 rounded-lg ${currentStep === item.step ? 'bg-white text-indigo-600' : 'bg-indigo-500/30'}`}>
                    {item.icon}
                  </div>
                  <span className="text-sm font-semibold tracking-wide uppercase">{item.label}</span>
                  {currentStep > item.step && <CheckCircle size={16} className="ml-auto text-emerald-400" />}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="p-10 flex-1 flex flex-col">
          {currentStep === 0 && (
            <div className="flex-1 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <h3 className="text-2xl font-black text-slate-900 mb-4">Checking Connection</h3>
              <p className="text-slate-500 mb-8 font-medium italic">Ensuring a stable uplink to the proctoring servers...</p>
              
              <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 mb-8">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-xl ${checks.network.status === 'success' ? 'bg-emerald-100 text-emerald-700' : 'bg-indigo-100 text-indigo-700'}`}>
                      <Wifi size={24} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800">Server Latency</h4>
                      <p className="text-sm text-slate-500">{checks.network.status === 'loading' ? 'Measuring...' : checks.network.latency ? `${checks.network.latency}ms` : 'Ready'}</p>
                    </div>
                  </div>
                  {checks.network.status === 'success' && <CheckCircle className="text-emerald-500" />}
                </div>
              </div>

              <button 
                onClick={() => setCurrentStep(1)}
                disabled={checks.network.status !== 'success'}
                className="w-full py-4 bg-indigo-600 text-white rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-indigo-700 transition-all disabled:opacity-50 mt-auto shadow-lg shadow-indigo-100"
              >
                Proceed to Hardware <ChevronRight size={20} />
              </button>
            </div>
          )}

          {currentStep === 1 && (
            <div className="flex-1 animate-in fade-in slide-in-from-right-4 duration-500">
              <h3 className="text-2xl font-black text-slate-900 mb-4">Hardware Validation</h3>
              <p className="text-slate-500 mb-8 font-medium">Verify your audio and visual peripherals are active.</p>
              
              <div className="space-y-4 mb-8">
                <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="flex items-center gap-3">
                    <Camera size={20} className="text-slate-600" />
                    <span className="font-bold text-slate-800">Primary Camera</span>
                  </div>
                  {checks.camera.status === 'success' ? <CheckCircle className="text-emerald-500" /> : <div className="w-5 h-5 border-2 border-slate-300 border-t-indigo-600 rounded-full animate-spin"></div>}
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <div className="flex items-center gap-3">
                    <Mic size={20} className="text-slate-600" />
                    <span className="font-bold text-slate-800">Microphone Array</span>
                  </div>
                  {checks.mic.status === 'success' ? <CheckCircle className="text-emerald-500" /> : <div className="w-5 h-5 border-2 border-slate-300 border-t-indigo-600 rounded-full animate-spin"></div>}
                </div>
              </div>

              {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 border border-red-100 rounded-xl flex items-center gap-3 text-sm font-bold">
                  <AlertCircle size={18} /> {error}
                </div>
              )}

              <div className="flex gap-4 mt-auto">
                <button 
                  onClick={runHardwareCheck}
                  className="flex-1 py-4 bg-slate-100 text-slate-700 rounded-2xl font-bold hover:bg-slate-200 transition-all"
                >
                  Retry Access
                </button>
                <button 
                  onClick={() => setCurrentStep(2)}
                  disabled={checks.camera.status !== 'success'}
                  className="flex-[2] py-4 bg-indigo-600 text-white rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-indigo-700 transition-all shadow-lg"
                >
                  Initialize Rules <ChevronRight size={20} />
                </button>
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="flex-1 animate-in fade-in zoom-in-95 duration-500">
              <h3 className="text-2xl font-black text-slate-900 mb-6">Examination Protocol</h3>
              
              <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-100 mb-8">
                <ul className="space-y-4">
                  {[
                    "Remain within the camera's field of view at all times.",
                    "No unauthorized personnel or devices allowed.",
                    "Keep your eyes focused on the primary display.",
                    "The assessment environment is monitored by AI."
                  ].map((rule, i) => (
                    <li key={i} className="flex gap-3 text-sm font-medium text-slate-700">
                      <div className="w-5 h-5 bg-indigo-200 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] text-indigo-700 font-black">{i+1}</div>
                      {rule}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="flex items-start gap-3 mb-8 bg-slate-50 p-4 rounded-xl">
                <input type="checkbox" id="agree" className="mt-1 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer" />
                <label htmlFor="agree" className="text-xs font-bold text-slate-600 leading-tight cursor-pointer">
                  I understand that my biometric hash will be processed to maintain academic integrity and I agree to the protocol.
                </label>
              </div>

              <button 
                onClick={() => setCurrentStep(3)}
                className="w-full py-4 bg-indigo-600 text-white rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-indigo-700 transition-all shadow-lg mt-auto"
              >
                Begin Biometric Match <ChevronRight size={20} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
