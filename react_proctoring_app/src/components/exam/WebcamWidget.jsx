import React, { useRef, useEffect, useState, useCallback } from 'react';
import { CameraOff, Mic, MicOff } from 'lucide-react';
import api, { apiForm } from '../../services/api';

export default function WebcamWidget({ sessionId, onRiskUpdate, onError }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const pollingInterval = useRef(null);
  const activeStreamRef = useRef(null);
  const consecutiveFailures = useRef(0);
  const [networkWarning, setNetworkWarning] = useState(false);
  
  // Audio Telemetry Ref
  const audioContext = useRef(null);
  const analyser = useRef(null);
  const dataArray = useRef(null);
  const [volume, setVolume] = useState(0);

  useEffect(() => {
    let activeStream = null;

    const initMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" },
            audio: true 
        });
        
        activeStreamRef.current = stream;
        
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
            setStreamActive(true);
            setMicActive(true);
            
            // Initialize Audio Analyzer
            setupAudioAnalysis(stream);
            
            // Start the infinite background proctoring loop
            startProctoringLoop();
        }
      } catch (err) {
        setStreamActive(false);
        setMicActive(false);
        onError('Media access denied. Sentinel requires both Camera and Microphone for enforcement.');
      }
    };

    const setupAudioAnalysis = (stream) => {
      try {
        audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.current.createMediaStreamSource(stream);
        analyser.current = audioContext.current.createAnalyser();
        analyser.current.fftSize = 256;
        source.connect(analyser.current);
        dataArray.current = new Uint8Array(analyser.current.frequencyBinCount);
      } catch (e) {
        console.warn("Audio analysis failed to initialize:", e);
      }
    };

    initMedia();

    return () => {
      if (pollingInterval.current) clearInterval(pollingInterval.current);
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContext.current) audioContext.current.close();
    };
  }, [sessionId, onError]);

  const getAverageVolume = () => {
    if (!analyser.current || !dataArray.current) return 0;
    analyser.current.getByteTimeDomainData(dataArray.current);
    
    let sum = 0;
    for (let i = 0; i < dataArray.current.length; i++) {
      const v = (dataArray.current[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / dataArray.current.length);
    return Math.min(rms * 5, 1); // Normalize and scale for sensitivity
  };

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    return dataUrl.split(',')[1];
  }, []);

  const startProctoringLoop = useCallback(() => {
    pollingInterval.current = setInterval(async () => {
       const frameData = captureFrame();
       const currentVolume = getAverageVolume();
       setVolume(currentVolume);

       if (!frameData) return;

       try {
         // Create standard object for JSON post (matching views.py update)
         const payload = {
            frame: frameData,
            audio_volume: currentVolume
         };
         
         const res = await api.post(`/sessions/${sessionId}/frame/`, payload);
         
         // Successful response clears consecutive failure triggers
         consecutiveFailures.current = 0;
         setNetworkWarning(false);
         
         // Extract response metrics (including faculty commands)
         const data = res.data;
         
         if (onRiskUpdate) {
             onRiskUpdate({
                 risk: data.risk_score || 0,
                 violation: data.violation_type || null,
                 components: data, // Pass full telemetry for drill-down
                 last_command: data.last_command || 'NONE'
             });
         }
       } catch (err) {
         console.warn("Sentinel heartbeat skipped:", err.message);
         consecutiveFailures.current += 1;
         if (consecutiveFailures.current >= 3) {
            setNetworkWarning(true);
         }
       }
    }, 2000);
  }, [sessionId, captureFrame, onRiskUpdate]);

  return (
    <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-black shadow-lg border-2 border-slate-800">
       {!streamActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 bg-gray-900 z-10 text-center px-4">
             <CameraOff size={48} className="mb-4 text-gray-600 animate-pulse" />
             <span className="text-xs font-black uppercase tracking-widest leading-loose">
                Initializing Secure Sentinel <br/> Biometric & Acoustic Link
             </span>
          </div>
       )}

       {networkWarning && (
          <div className="absolute top-16 left-4 right-4 bg-amber-600/90 backdrop-blur-md px-4 py-2.5 rounded-xl text-[10px] font-black text-white shadow-2xl z-20 flex items-center gap-2 border border-amber-500/20 animate-pulse text-left">
             <span className="text-sm">⚠️</span>
             <div className="flex-1 tracking-wider uppercase leading-tight">
                Proctor Heartbeat failing. <br/> Check network uplink stability immediately.
             </div>
          </div>
       )}

       <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={`w-full h-full object-cover transition-opacity duration-1000 ${streamActive ? 'opacity-100' : 'opacity-0'}`} 
       />
       
       <canvas ref={canvasRef} className="hidden" />

       {/* Biometric Pulse HUD */}
       <div className="absolute top-4 right-4 flex items-center bg-black/60 backdrop-blur-xl px-4 py-2 rounded-2xl text-[10px] font-black text-white shadow-2xl z-20 gap-3 border border-white/10">
           <div className="flex items-center gap-1.5 pr-2 border-r border-white/20">
              <div className={`w-2 h-2 rounded-full ${streamActive ? 'bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.8)]' : 'bg-slate-600'}`}></div>
              <span className="tracking-widest uppercase">Video</span>
           </div>
           
           <div className="flex items-center gap-1.5">
              {micActive ? <Mic size={12} className="text-indigo-400" /> : <MicOff size={12} className="text-slate-500" />}
              <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-500 transition-all duration-200" 
                  style={{ width: `${volume * 100}%` }}
                ></div>
              </div>
           </div>
       </div>

       {/* Sentinel Lens Overlay */}
       <div className="absolute inset-0 pointer-events-none border-[20px] border-transparent group-hover:border-indigo-500/10 transition-all duration-700">
          <div className="absolute top-8 left-8 w-4 h-4 border-t-2 border-l-2 border-white/30"></div>
          <div className="absolute top-8 right-8 w-4 h-4 border-t-2 border-r-2 border-white/30"></div>
          <div className="absolute bottom-8 left-8 w-4 h-4 border-b-2 border-l-2 border-white/30"></div>
          <div className="absolute bottom-8 right-8 w-4 h-4 border-b-2 border-r-2 border-white/30"></div>
       </div>
    </div>
  );
}
