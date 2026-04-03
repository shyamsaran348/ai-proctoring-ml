import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera, CheckCircle, ShieldCheck, AlertTriangle } from 'lucide-react';
import { apiForm } from '../../services/api';

export default function BiometricOnboarding({ sessionId, onVerified }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [streamActive, setStreamActive] = useState(false);
  const [step, setStep] = useState(1); // 1 = Face, 2 = ID Card, 3 = Complete
  const [statusMsg, setStatusMsg] = useState('Position your face in the center of the frame.');
  const [errorMsg, setErrorMsg] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Initialize Camera
  useEffect(() => {
    let activeStream = null;
    const startCamera = async () => {
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
          setStreamActive(true);
        }
      } catch (err) {
        setErrorMsg('Camera access denied or unavailable. Please check permissions.');
      }
    };
    startCamera();

    return () => {
      if (activeStream) {
         activeStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to strict Base64 exactly like the Python backend expects
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    return dataUrl.split(',')[1];
  }, []);

  const handleCaptureFace = async () => {
    setErrorMsg(null);
    setIsProcessing(true);
    setStatusMsg('Verifying biometric identity...');
    
    const base64Image = captureFrame();
    if (!base64Image) {
       setErrorMsg("Failed to capture image. Is the camera active?");
       setIsProcessing(false);
       return;
    }

    try {
      const fd = new FormData();
      fd.append('image', base64Image); // Matched with backend 'image' key
      
      const res = await apiForm.post(`/sessions/${sessionId}/verify_face_snapshot/`, fd);
      
      if (res.data.verified) {
         setStep(3);
         setStatusMsg('Biometric Identity Validation Complete.');
         setTimeout(() => {
            onVerified();
         }, 1500);
      } else {
         setErrorMsg(res.data.error || 'Identity verification failed. Please try again.');
         setStatusMsg('Position your face clearly.');
      }
    } catch (err) {
      setErrorMsg(err.response?.data?.error || 'Server error during verification.');
      setStatusMsg('Position your face clearly.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCaptureID = async () => {
    setErrorMsg(null);
    setIsProcessing(true);
    setStatusMsg('Scanning ID Card artifacts...');
    
    const base64Image = captureFrame();
    if (!base64Image) return setIsProcessing(false);

    try {
      const fd = new FormData();
      fd.append('id_card_data', base64Image);
      
      const res = await apiForm.post(`/sessions/${sessionId}/capture_id_card/`, fd);
      
      if (res.data.verified) {
         setStep(3);
         setStatusMsg('Multi-Factor Identity Validation Complete.');
         setTimeout(() => {
            onVerified();
         }, 1500);
      } else {
         setErrorMsg(res.data.error || 'Failed to validate ID Card.');
         setStatusMsg('Hold your ID Card clearly to the camera.');
      }
    } catch (err) {
      setErrorMsg(err.response?.data?.error || 'Server error during ID validation.');
      setStatusMsg('Hold your ID Card clearly to the camera.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-900/95 backdrop-blur-md flex justify-center items-center z-50 px-4">
       <div className="bg-white rounded-3xl overflow-hidden shadow-2xl max-w-4xl w-full flex flex-col md:flex-row border border-gray-700">
          
          {/* Left specific info panel */}
          <div className="bg-gradient-to-br from-indigo-900 to-indigo-700 text-white p-10 md:w-1/3 flex flex-col">
             <div className="mb-auto">
                <ShieldCheck size={48} className="mb-6 text-indigo-300" />
                <h2 className="text-3xl font-extrabold mb-4">Biometric Enrollment</h2>
                <p className="text-indigo-100 font-medium leading-relaxed mb-6">
                   To ensure exam integrity, DevProctor requires a live multi-factor confirmation.
                </p>
                <ul className="space-y-4 text-sm mt-8">
                   <li className="flex items-center text-emerald-400 font-bold">
                      <CheckCircle size={18} className="mr-3" /> Identity Verification
                   </li>
                </ul>
             </div>
          </div>

          {/* Right dynamic camera panel */}
          <div className="p-8 md:w-2/3 bg-gray-50 flex flex-col items-center">
             
             <div className="relative rounded-2xl overflow-hidden bg-black shadow-inner aspect-video w-full max-w-lg mb-6 border-4 border-gray-200">
                <video 
                   ref={videoRef} 
                   autoPlay 
                   playsInline 
                   muted 
                   className={`w-full h-full object-cover ${streamActive ? 'opacity-100' : 'opacity-0'}`} 
                />
                {!streamActive && (
                   <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                      Loading Camera...
                   </div>
                )}
                
                {/* Visual Guides overlay based on step */}
                {streamActive && step === 1 && (
                   <div className="absolute inset-x-0 inset-y-8 border-2 border-indigo-400 border-dashed rounded-[100px] pointer-events-none opacity-60 mx-16"></div>
                )}
                {streamActive && step === 2 && (
                   <div className="absolute inset-0 border-2 border-emerald-400 border-dashed m-12 pointer-events-none opacity-60 flex items-center justify-center">
                      <span className="text-emerald-400/80 font-bold tracking-widest uppercase">Align ID Here</span>
                   </div>
                )}
             </div>

             <canvas ref={canvasRef} className="hidden" />

             {errorMsg && (
                <div className="w-full max-w-lg mb-4 bg-red-100 border border-red-200 text-red-700 px-4 py-3 rounded-xl flex items-start text-sm font-medium">
                   <AlertTriangle className="shrink-0 mr-3 mt-0.5" size={18} />
                   {errorMsg}
                </div>
             )}

             <p className="text-gray-700 font-semibold mb-6 text-center">{statusMsg}</p>

             {step === 1 && (
                <button 
                  onClick={handleCaptureFace} 
                  disabled={!streamActive || isProcessing}
                  className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white px-8 py-3 rounded-xl font-bold flex items-center shadow-lg transition-all"
                >
                   <Camera size={20} className="mr-3" />
                   {isProcessing ? 'Analyzing Core Metrics...' : 'Capture & Verify Face'}
                </button>
             )}

             {step === 2 && (
                <button 
                  onClick={handleCaptureID} 
                  disabled={!streamActive || isProcessing}
                  className="bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-8 py-3 rounded-xl font-bold flex items-center shadow-lg transition-all"
                >
                   <ShieldCheck size={20} className="mr-3" />
                   {isProcessing ? 'Processing ID Entropy...' : 'Scan Official ID'}
                </button>
             )}

             {step === 3 && (
                <div className="text-emerald-600 font-extrabold flex items-center animate-pulse text-lg">
                   <CheckCircle className="mr-2" size={24} /> Redirecting to Arena...
                </div>
             )}
          </div>
       </div>
    </div>
  );
}
