/**
 * Dream Catcher : Drowsiness Detection System
 * Built with React 19, Vite, and MediaPipe Vision.
 */
import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { 
  Camera, 
  AlertTriangle, 
  Activity, 
  Settings, 
  Volume2, 
  VolumeX,
  Eye,
  EyeOff,
  ShieldAlert,
  Zap,
  BarChart3,
  LayoutDashboard,
  Download,
  Trash2,
  Clock,
  CheckCircle2,
  FileSpreadsheet,
  Settings2,
  RefreshCw,
  Wifi,
  History,
  X,
  Maximize,
  Minimize,
  ZoomIn,
  ZoomOut,
  Pause,
  Play,
  Coffee,
  Battery,
  BatteryLow,
  Info,
  BookOpen,
  Sun,
  Plus,
  Minus
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  FaceLandmarker, 
  FilesetResolver,
  FaceLandmarkerResult
} from '@mediapipe/tasks-vision';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts';

// --- Global Log Suppression for MediaPipe/XNNPACK ---
// These informational logs can be confusing as they look like errors to users.
const suppressLogs = () => {
  const originalInfo = console.info;
  const originalLog = console.log;
  const originalWarn = console.warn;
  const originalError = console.error;

  const isDelegateLog = (args: any[]) => {
    const msg = args[0]?.toString() || '';
    return msg.includes('XNNPACK delegate') || msg.includes('TensorFlow Lite');
  };

  console.info = (...args) => { if (!isDelegateLog(args)) originalInfo(...args); };
  console.log = (...args) => { if (!isDelegateLog(args)) originalLog(...args); };
  console.warn = (...args) => { if (!isDelegateLog(args)) originalWarn(...args); };
  console.error = (...args) => { if (!isDelegateLog(args)) originalError(...args); };
};
suppressLogs();

// --- Constants ---
const DROWSY_TIME_MS = 2000; // 2 seconds
const MODEL_ASSET_PATH = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// --- Types ---
type DetectionStatus = 'initializing' | 'ready' | 'active' | 'drowsy' | 'error';
type ViewMode = 'dashboard' | 'statistics';

interface DrowsyEvent {
  id: string;
  timestamp: number;
  duration: number;
  startTime: string;
}

export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);
  const lastClosedTimeRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const alarmIntervalRef = useRef<number | null>(null);

  // State
  const [status, setStatus] = useState<DetectionStatus>('initializing');
  const [viewMode, setViewMode] = useState<ViewMode>('dashboard');
  const [isMuted, setIsMuted] = useState(false);
  const [eyeClosure, setEyeClosure] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAlarming, setIsAlarming] = useState(false);
  const [drowsyLimitMs, setDrowsyLimitMs] = useState(2000);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isWipeConfirmOpen, setIsWipeConfirmOpen] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isPowerSaving, setIsPowerSaving] = useState(false);
  const [autoPowerSaving, setAutoPowerSaving] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [brightness, setBrightness] = useState(100);
  const [isHudVisible, setIsHudVisible] = useState(true);
  const [battery, setBattery] = useState<{level: number, charging: boolean}>({level: 0, charging: false});
  const [isSafetyGuideOpen, setIsSafetyGuideOpen] = useState(true); // Open by default
  const [latency, setLatency] = useState(0);
  const [isBreakActive, setIsBreakActive] = useState(false);
  const [isAutoFocus, setIsAutoFocus] = useState(true);
  const [headPose, setHeadPose] = useState({ pitch: 0, yaw: 0, roll: 0 });
  const [fatigueLevel, setFatigueLevel] = useState(0);
  const [facialExpression, setFacialExpression] = useState('Neutral');
  
  const ignoreDrowsyUntilOpenRef = useRef(false);
  const statusRef = useRef<DetectionStatus>(status);
  const drowsyLimitMsRef = useRef(drowsyLimitMs);
  const isPausedRef = useRef(isPaused);
  const isPowerSavingRef = useRef(isPowerSaving);
  const lastDetectionTimeRef = useRef(0);
  
  // Controls
  const [sensitivity, setSensitivity] = useState(0.5);
  const [alarmVolume, setAlarmVolume] = useState(0.8);
  
  // Data
  const [history, setHistory] = useState<DrowsyEvent[]>(() => {
    try {
      const saved = localStorage.getItem('drowsy_history');
      if (!saved) return [];
      const parsed = JSON.parse(saved);
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      console.error("Failed to load history:", err);
      return [];
    }
  });

  const [fatigueHistory, setFatigueHistory] = useState<{timestamp: number, level: number}[]>(() => {
    try {
      const saved = localStorage.getItem('fatigue_history');
      if (!saved) return [];
      const parsed = JSON.parse(saved);
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      console.error("Failed to load fatigue history:", err);
      return [];
    }
  });

  const [logs, setLogs] = useState<string[]>([]);

  // Refs for real-time control access without re-creating functions
  const sensitivityRef = useRef(sensitivity);
  const alarmVolumeRef = useRef(alarmVolume);
  const isMutedRef = useRef(isMuted);
  const isRecordingEventRef = useRef(false);

  useEffect(() => { sensitivityRef.current = sensitivity; }, [sensitivity]);
  useEffect(() => { alarmVolumeRef.current = alarmVolume; }, [alarmVolume]);
  useEffect(() => { isMutedRef.current = isMuted; }, [isMuted]);
  useEffect(() => { statusRef.current = status; }, [status]);
  useEffect(() => { drowsyLimitMsRef.current = drowsyLimitMs; }, [drowsyLimitMs]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { isPowerSavingRef.current = isPowerSaving; }, [isPowerSaving]);

  // Auto Power Saver Logic (Simulated based on inactivity or toggle)
  useEffect(() => {
    if (autoPowerSaving && status === 'active') {
      const handleVisibilityChange = () => {
        setIsPowerSaving(document.hidden);
      };
      document.addEventListener('visibilitychange', handleVisibilityChange);
      return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
    }
  }, [autoPowerSaving, status]);

  // Battery Status API
  useEffect(() => {
    if ('getBattery' in navigator) {
      (navigator as any).getBattery().then((bat: any) => {
        const updateBattery = () => {
          setBattery({
            level: Math.round(bat.level * 100),
            charging: bat.charging
          });
        };
        updateBattery();
        bat.addEventListener('levelchange', updateBattery);
        bat.addEventListener('chargingchange', updateBattery);
        return () => {
          bat.removeEventListener('levelchange', updateBattery);
          bat.removeEventListener('chargingchange', updateBattery);
        };
      });
    }
  }, []);

  // Auto Focus Logic (Simulated/Constraint based)
  useEffect(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      const track = stream.getVideoTracks()[0];
      if (track && track.getCapabilities && (track.getCapabilities() as any).focusMode) {
        track.applyConstraints({
          advanced: [{ focusMode: isAutoFocus ? 'continuous' : 'manual' }] as any
        }).catch(e => console.warn("Focus control not supported:", e));
      }
    }
  }, [isAutoFocus, status]);

  // --- Logging ---
  const addLog = useCallback((msg: string) => {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    setLogs(prev => [`[${time}] ${msg}`, ...prev].slice(0, 50));
  }, []);

  // --- Persistence ---
  useEffect(() => {
    localStorage.setItem('drowsy_history', JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    localStorage.setItem('fatigue_history', JSON.stringify(fatigueHistory));
  }, [fatigueHistory]);

  // Record fatigue level every 1 minute for history trend
  useEffect(() => {
    const interval = setInterval(() => {
      if (status === 'active' || status === 'drowsy') {
        setFatigueHistory(prev => {
          const newPoint = { timestamp: Date.now(), level: fatigueLevel };
          // Keep only last 7 days of points (approx 10080 points if 1min, but we can clean up older ones)
          const sevenDaysAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
          return [...prev.filter(p => p.timestamp > sevenDaysAgo), newPoint].slice(-1000); // Limit to 1000 points for performance
        });
      }
    }, 60000);
    return () => clearInterval(interval);
  }, [status, fatigueLevel]);

  // --- Audio Logic ---
  const playAlarmTone = useCallback(() => {
    if (isMutedRef.current) return;
    
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    const ctx = audioContextRef.current;
    if (ctx.state === 'suspended') ctx.resume();

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.1);
    
    // Use the ref for volume to get the latest value even inside a stale interval closure
    gain.gain.setValueAtTime(alarmVolumeRef.current * 0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  }, []);

  const startContinuousAlarm = useCallback(() => {
    // Check the ref to avoid double-starting if state hasn't updated yet
    if (alarmIntervalRef.current) return;
    
    setIsAlarming(true);
    addLog("ALERT: Continuous alarm triggered.");
    
    // Play immediately
    playAlarmTone();
    
    // Repeat every 600ms
    alarmIntervalRef.current = window.setInterval(() => {
      playAlarmTone();
    }, 600);
  }, [playAlarmTone, addLog]);

  const stopAlarm = useCallback(() => {
    if (alarmIntervalRef.current) {
      window.clearInterval(alarmIntervalRef.current);
      alarmIntervalRef.current = null;
    }
    setIsAlarming(false);
    isRecordingEventRef.current = false;
    ignoreDrowsyUntilOpenRef.current = true;
    addLog("Alarm deactivated. System ignoring drowsiness until eyes open.");
    
    // If we were drowsy, go back to active
    setStatus(prev => prev === 'drowsy' ? 'active' : prev);
  }, [addLog]);

  const calibrateFace = useCallback(() => {
    addLog("Calibrating face landmarker...");
    lastClosedTimeRef.current = null;
    ignoreDrowsyUntilOpenRef.current = false;
    
    // Visual feedback
    const originalStatus = status;
    setStatus('initializing');
    setTimeout(() => {
      setStatus(originalStatus === 'error' ? 'ready' : originalStatus);
      addLog("Calibration complete. System optimized.");
    }, 1000);
  }, [status, addLog]);

  const testAlarm = useCallback(() => {
    addLog("Testing alarm sound...");
    playAlarmTone();
    setTimeout(playAlarmTone, 600);
  }, [playAlarmTone, addLog]);

  // --- Full Screen Logic ---
  const toggleFullScreen = useCallback(() => {
    if (!document.fullscreenElement && !isFullScreen) {
      document.documentElement.requestFullscreen().catch(err => {
        console.warn(`Browser Full-screen API suppressed: ${err.message}. Using internal UI toggle.`);
        setIsFullScreen(true);
      });
    } else {
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {
          setIsFullScreen(false);
        });
      } else {
        setIsFullScreen(false);
      }
    }
  }, [isFullScreen]);

  useEffect(() => {
    const handleFullScreenChange = () => {
      setIsFullScreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullScreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullScreenChange);
  }, []);

  // --- MediaPipe Initialization ---
  useEffect(() => {
    async function init() {
      try {
        addLog("Initializing vision engine...");
        
        // Pinning version for stability and better log handling
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
        );
        
        // SUPPRESSION: Broad log filtering is now handled globally at the top of the file.
        // This ensures even early initialization logs are captured.
        
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH,
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        
        setStatus('ready');
        addLog("Vision engine initialized successfully.");
      } catch (err) {
        console.error("Failed to initialize FaceLandmarker:", err);
        setErrorMessage("Vision system failed to load. Check connection.");
        setStatus('error');
        addLog("ERROR: Vision system failed.");
      }
    }
    init();
    
    return () => {
      faceLandmarkerRef.current?.close();
      if (alarmIntervalRef.current) clearInterval(alarmIntervalRef.current);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [addLog]);

  // --- Camera Logic ---
  const startCamera = async () => {
    if (!videoRef.current) return;
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage("Your browser does not support camera access. Please use a modern browser like Chrome or Edge.");
      setStatus('error');
      addLog("ERROR: MediaDevices API not supported.");
      return;
    }
    
    try {
      // Initialize AudioContext on user gesture to ensure it's unlocked
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      addLog("Requesting camera access...");
      
      // More flexible constraints for better compatibility across devices
      const constraints = {
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: 'user',
          aspectRatio: { ideal: window.innerWidth > window.innerHeight ? 1.777777778 : 0.5625 }
        }
      };

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (e) {
        addLog("Retrying with basic constraints...");
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => {
            console.error("Video play failed:", e);
            addLog("ERROR: Video playback failed.");
          });
          setStatus('active');
          addLog("Camera stream active.");
          startDetection();
        };
      }
    } catch (err: any) {
      console.error("Camera access failed:", err);
      
      let message = "Camera access is required for detection. Please ensure permissions are granted.";
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        message = "Camera access denied. Please click the camera icon in your browser's address bar to reset permissions, then try again.";
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        message = "No camera found on this device. Please connect a webcam and try again.";
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        message = "Camera is already in use by another application. Please close other apps and try again.";
      } else if (err.name === 'OverconstrainedError') {
        message = "Camera constraints cannot be satisfied. Retrying with default settings...";
        // This is handled by the retry block inside the try, but if it still fails:
        message = "Your camera does not support the required resolution.";
      }

      setErrorMessage(message);
      setStatus('error');
      addLog(`ERROR: Camera access ${err.name || 'failed'}.`);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    setStatus('ready');
    addLog("Monitoring stopped. Camera deactivated.");
  };

  // --- Detection Loop ---
  const startDetection = () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    
    const detect = async () => {
      if (isPausedRef.current) {
        requestRef.current = requestAnimationFrame(detect);
        return;
      }

      // Power saving: limit to ~10 FPS
      if (isPowerSavingRef.current) {
        const now = performance.now();
        if (now - lastDetectionTimeRef.current < 100) {
          requestRef.current = requestAnimationFrame(detect);
          return;
        }
        lastDetectionTimeRef.current = now;
      }

      if (
        videoRef.current && 
        videoRef.current.readyState >= 2 && // Ensure video has enough data
        faceLandmarkerRef.current && 
        statusRef.current !== 'error'
      ) {
        const startTimeMs = performance.now();
        try {
          const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
          const endTimeMs = performance.now();
          setLatency(Math.round(endTimeMs - startTimeMs));
          processResults(results);
        } catch (err) {
          console.warn("Detection frame skipped:", err);
        }
      }
      requestRef.current = requestAnimationFrame(detect);
    };
    requestRef.current = requestAnimationFrame(detect);
  };

  const processResults = (results: FaceLandmarkerResult) => {
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
      const blendshapes = results.faceBlendshapes[0].categories;
      const leftEye = blendshapes.find(c => c.categoryName === 'eyeBlinkLeft')?.score || 0;
      const rightEye = blendshapes.find(c => c.categoryName === 'eyeBlinkRight')?.score || 0;
      
      const avgClosure = (leftEye + rightEye) / 2;
      setEyeClosure(avgClosure);

      // Facial Expressions (Micro-expressions)
      const yawn = blendshapes.find(c => c.categoryName === 'jawOpen')?.score || 0;
      const browDown = (blendshapes.find(c => c.categoryName === 'browDownLeft')?.score || 0 + 
                        blendshapes.find(c => c.categoryName === 'browDownRight')?.score || 0) / 2;
      
      if (yawn > 0.5) setFacialExpression('Yawning');
      else if (browDown > 0.4) setFacialExpression('Strained');
      else setFacialExpression('Neutral');

      // Head Pose Analysis
      if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
        const matrix = results.facialTransformationMatrixes[0].data;
        // Basic extraction of pitch, yaw, roll from transformation matrix
        const pitch = Math.asin(-matrix[6]);
        const yaw = Math.atan2(matrix[2], matrix[10]);
        const roll = Math.atan2(matrix[4], matrix[5]);
        
        setHeadPose({ 
          pitch: pitch * (180 / Math.PI), 
          yaw: yaw * (180 / Math.PI), 
          roll: roll * (180 / Math.PI) 
        });
      }

      // Advanced Fatigue Prediction
      // Factors: Eye closure, Yawning, Head tilt (nodding off)
      const headTilt = Math.abs(headPose.pitch) > 20 ? 0.3 : 0;
      const yawnFactor = yawn > 0.4 ? 0.2 : 0;
      const eyeFactor = avgClosure > 0.5 ? 0.5 : 0;
      
      const predictedFatigue = Math.min((eyeFactor + headTilt + yawnFactor) * sensitivityRef.current * 2, 1);
      setFatigueLevel(predictedFatigue);

      // Use sensitivity to adjust threshold
      const adjustedThreshold = 1.0 - (sensitivityRef.current * 0.8 + 0.1); 

      if (avgClosure > adjustedThreshold || predictedFatigue > 0.8) {
        if (ignoreDrowsyUntilOpenRef.current) {
          return;
        }

        if (!lastClosedTimeRef.current) {
          lastClosedTimeRef.current = Date.now();
        } else {
          const duration = Date.now() - lastClosedTimeRef.current;
          if (duration > drowsyLimitMsRef.current && !isRecordingEventRef.current) {
            isRecordingEventRef.current = true;
            setStatus('drowsy');
            startContinuousAlarm();
            
            // Record event
            const newEvent: DrowsyEvent = {
              id: Math.random().toString(36).substr(2, 9),
              timestamp: Date.now(),
              duration: duration,
              startTime: new Date().toLocaleString()
            };
            setHistory(prev => [newEvent, ...prev]);
          }
        }
      } else {
        lastClosedTimeRef.current = null;
        if (ignoreDrowsyUntilOpenRef.current) {
          ignoreDrowsyUntilOpenRef.current = false;
          setStatus('active');
          addLog("Eyes detected open. Monitoring resumed.");
        }
      }
    } else {
      setEyeClosure(0);
      setFatigueLevel(0);
      setFacialExpression('Not Detected');
      lastClosedTimeRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // --- Statistics Logic ---
  const chartData = useMemo(() => {
    try {
      if (!Array.isArray(history)) return [];
      const last7Days = Array.from({ length: 7 }, (_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - i);
        return d.toLocaleDateString();
      }).reverse();

      return last7Days.map(date => {
        const count = history.filter(e => e && e.timestamp && new Date(e.timestamp).toLocaleDateString() === date).length;
        return { date, count };
      });
    } catch (err) {
      console.error("Chart data calculation error:", err);
      return [];
    }
  }, [history]);

  const fatigueChartData = useMemo(() => {
    try {
      if (!Array.isArray(fatigueHistory)) return [];
      const now = Date.now();
      // Generate 24 slots for the last 24 hours
      return Array.from({ length: 24 }, (_, i) => {
        const hourTimestamp = now - (23 - i) * 3600000;
        const hourLabel = new Date(hourTimestamp).getHours() + ":00";
        const pointsInHour = fatigueHistory.filter(p => p.timestamp >= hourTimestamp && p.timestamp < hourTimestamp + 3600000);
        const avg = pointsInHour.length > 0 ? pointsInHour.reduce((acc, p) => acc + p.level, 0) / pointsInHour.length : 0;
        return { time: hourLabel, fatigue: Math.round(avg * 100) };
      });
    } catch (err) {
      console.error("Failed to compute fatigue trend:", err);
      return [];
    }
  }, [fatigueHistory]);

  const exportData = () => {
    const dataStr = JSON.stringify(history, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'drowsy_guard_history.json';
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    addLog("Data exported successfully.");
  };

  const exportCSV = () => {
    if (history.length === 0) return;
    const headers = ['ID', 'Timestamp', 'Duration (ms)', 'Start Time'];
    const rows = history.map(e => [e.id, e.timestamp, e.duration, `"${e.startTime}"`]);
    const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `Dream_Catcher_data_${new Date().toISOString()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog("CSV exported successfully.");
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('drowsy_history');
    addLog("History cleared.");
    setIsWipeConfirmOpen(false);
  };

  // --- Navigation Fix ---
  useEffect(() => {
    if (viewMode === 'dashboard' && status === 'active' && videoRef.current && !videoRef.current.srcObject) {
      addLog("Resuming camera feed...");
      startCamera();
    }
  }, [viewMode, status]);

  const takeBreak = () => {
    setIsBreakActive(true);
    setIsPaused(true);
    addLog("Break initiated. Monitoring paused.");
  };

  const endBreak = () => {
    setIsBreakActive(false);
    setIsPaused(false);
    addLog("Break ended. Monitoring resumed.");
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#E0E0E0] font-mono selection:bg-cyan-500 selection:text-black flex flex-col">
      {/* Header / Status Bar */}
      {!isFullScreen && (
        <header className={`h-16 border-b border-[#1F1F1F] bg-[#0F0F0F]/80 backdrop-blur-xl px-4 sm:px-6 flex items-center justify-between sticky top-0 z-[60] transition-all duration-500`}>
          <div 
            className="flex flex-shrink-0 items-center gap-2 sm:gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => setViewMode('dashboard')}
          >
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg sm:rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(244,63,94,0.3)]">
              <ShieldAlert size={18} className="text-white sm:hidden" />
              <ShieldAlert size={24} className="text-white hidden sm:block" />
            </div>
            <div className="flex flex-col">
              <h1 className="text-sm sm:text-xl font-black tracking-tighter text-white leading-none uppercase">Dream<span className="text-rose-500"> Catcher</span></h1>
              <p className="text-[6px] sm:text-[8px] text-[#444] font-bold tracking-[0.2em] sm:tracking-[0.3em] uppercase mt-0.5 sm:mt-1">Drowsiness System v4.5</p>
            </div>
          </div>

        <nav className="hidden md:flex bg-[#151619] rounded-full border border-[#1F1F1F] p-1 scale-90 md:scale-100">
          <button 
            onClick={() => setViewMode('dashboard')}
            className={`flex items-center gap-2 px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === 'dashboard' ? 'bg-white text-black shadow-lg' : 'text-[#666] hover:text-white'}`}
          >
            <LayoutDashboard size={12} /> Dashboard
          </button>
          <button 
            onClick={() => setViewMode('statistics')}
            className={`flex items-center gap-2 px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === 'statistics' ? 'bg-white text-black shadow-lg' : 'text-[#666] hover:text-white'}`}
          >
            <BarChart3 size={12} /> Data Archive
          </button>
        </nav>

        <div className="flex items-center gap-1 sm:gap-4 md:gap-6">
          <div className="hidden lg:flex items-center gap-4 px-4 py-1.5 bg-[#151619] border border-[#1F1F1F] rounded-lg">
            <div className="flex items-center gap-2 mr-2 border-r border-[#1F1F1F] pr-4">
              <Battery size={14} className={battery.level < 20 ? 'text-rose-500' : 'text-emerald-500'} />
              <span className="text-[10px] font-black text-white">{battery.level}%</span>
              {battery.charging && <Zap size={10} className="text-amber-400 animate-pulse" />}
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[8px] text-[#444] font-bold uppercase">System Status</span>
              <span className="text-[10px] text-emerald-400 font-black">NOMINAL</span>
            </div>
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
          </div>
          <button 
            onClick={() => setIsSafetyGuideOpen(true)}
            className="p-1.5 sm:p-2 rounded-lg text-[#666] hover:text-white hover:bg-white/5 transition-all flex items-center gap-2"
            title="Operational Manual"
          >
            <Info size={16} className="sm:hidden" />
            <Info size={18} className="hidden sm:block" />
            <span className="hidden xl:inline text-[10px] font-black uppercase tracking-widest">Help</span>
          </button>
          <button 
            onClick={() => setIsHudVisible(!isHudVisible)}
            className={`p-1.5 sm:p-2 rounded-lg transition-all ${!isHudVisible ? 'text-cyan-400 bg-cyan-400/10' : 'text-[#666] hover:text-white hover:bg-white/5'}`}
            title={isHudVisible ? "Hide HUD" : "Show HUD"}
          >
            {isHudVisible ? <EyeOff size={16} className="sm:hidden" /> : <Eye size={16} className="sm:hidden" />}
            {isHudVisible ? <EyeOff size={18} className="hidden sm:block" /> : <Eye size={18} className="hidden sm:block" />}
          </button>
          <button 
            onClick={() => setIsMuted(!isMuted)}
            className={`p-1.5 sm:p-2 rounded-lg transition-all ${isMuted ? 'text-rose-500 bg-rose-500/10' : 'text-[#666] hover:text-white hover:bg-white/5'}`}
          >
            {isMuted ? <VolumeX size={16} className="sm:hidden" /> : <Volume2 size={16} className="sm:hidden" />}
            {isMuted ? <VolumeX size={18} className="hidden sm:block" /> : <Volume2 size={18} className="hidden sm:block" />}
          </button>
          <button 
            onClick={toggleFullScreen}
            className="p-1.5 sm:p-2 rounded-lg text-[#666] hover:text-white hover:bg-white/5 transition-all"
            title={isFullScreen ? "Exit Full Screen" : "Enter Full Screen"}
          >
            {isFullScreen ? <Minimize size={16} className="sm:hidden" /> : <Maximize size={16} className="sm:hidden" />}
            {isFullScreen ? <Minimize size={18} className="hidden sm:block" /> : <Maximize size={18} className="hidden sm:block" />}
          </button>
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="p-1.5 sm:p-2 rounded-lg text-[#666] hover:text-white hover:bg-white/5 transition-all"
          >
            <Settings size={16} className="sm:hidden" />
            <Settings size={18} className="hidden sm:block" />
          </button>
        </div>
      </header>
    )}

      <main className="flex-1 relative overflow-hidden">
        {/* Background Grid */}
        <div className="fixed inset-0 opacity-5 pointer-events-none z-0" 
             style={{ backgroundImage: 'radial-gradient(#FFF 1px, transparent 1px)', backgroundSize: '32px 32px' }} />

        <div className="relative z-10 h-full">
          
          {viewMode === 'dashboard' ? (
            <div className="flex flex-col h-full">
              {/* Hero Section: Camera Viewport */}
              <section className={`relative w-full bg-[#050505] transition-all duration-700 ease-in-out ${isFullScreen ? 'h-screen fixed inset-0 z-[100]' : 'h-[calc(100vh-64px)]'}`}>
                <div className="absolute inset-0">
                  {/* Mirror Video Element */}
                  <div className="w-full h-full flex items-center justify-center bg-black overflow-hidden">
                    <video 
                      ref={videoRef} 
                      className={`w-full h-full transition-all duration-700 ease-out ${status === 'active' ? 'opacity-100' : 'opacity-0'}`}
                      style={{ 
                        objectFit: 'cover',
                        transform: `scaleX(-1) scale(${zoomLevel})`,
                        filter: `brightness(${isPowerSaving ? Math.min(brightness, 40) : brightness}%)`
                      }}
                      playsInline
                      muted
                      autoPlay
                    />
                  </div>

                  {/* Exit Full Screen Button Tip */}
                  {isFullScreen && (
                    <button 
                      onClick={toggleFullScreen}
                      className="absolute top-8 right-8 z-[110] p-4 bg-black/60 backdrop-blur-md rounded-2xl border border-white/20 text-white hover:bg-black/80 transition-all pointer-events-auto"
                      title="Exit Full Screen"
                    >
                      <Minimize size={24} />
                    </button>
                  )}

                  {/* HUD Overlay */}
                  <AnimatePresence>
                    {isHudVisible && (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 pointer-events-none"
                      >
                        {/* Status Warnings on HUD */}
                        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 pointer-events-none">
                          <AnimatePresence mode="popLayout">
                            {Math.abs(headPose.pitch) > 15 && (
                              <motion.div 
                                initial={{ opacity: 0, scale: 0.5, y: -10 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.5, y: -10 }}
                                className="bg-rose-600/40 backdrop-blur-md px-4 py-1 rounded-full border border-rose-500/50 flex items-center gap-2"
                              >
                                <AlertTriangle size={12} className="text-white" />
                                <span className="text-[10px] font-black text-white uppercase tracking-widest">
                                  {headPose.pitch > 0 ? "HEAD UP" : "HEAD DOWN"}
                                </span>
                              </motion.div>
                            )}
                            {(Math.abs(headPose.yaw) > 20 || Math.abs(headPose.roll) > 15) && (
                              <motion.div 
                                initial={{ opacity: 0, scale: 0.5, y: -10 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.5, y: -10 }}
                                className="bg-rose-600/40 backdrop-blur-md px-4 py-1 rounded-full border border-rose-500/50 flex items-center gap-2"
                              >
                                <ShieldAlert size={12} className="text-white" />
                                <span className="text-[10px] font-black text-white uppercase tracking-widest">POSTURE DEVIATION</span>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>

                        {/* Corner Brackets */}
                        <div className="absolute top-12 left-12 w-16 h-16 border-t-2 border-l-2 border-white/20" />
                        <div className="absolute top-12 right-12 w-16 h-16 border-t-2 border-r-2 border-white/20" />
                        <div className="absolute bottom-24 left-12 w-16 h-16 border-b-2 border-l-2 border-white/20" />
                        <div className="absolute bottom-24 right-12 w-16 h-16 border-b-2 border-r-2 border-white/20" />

                        {/* HUD Content */}
                        <div className="absolute inset-0 flex flex-col justify-between p-4 sm:p-8 md:p-12 pb-56 sm:pb-28 pointer-events-none">
                          <div className="flex justify-between items-start">
                            <div className="flex flex-col gap-2 sm:gap-3">
                              <div className="bg-black/60 backdrop-blur-2xl border border-white/10 px-4 sm:px-6 py-2 sm:py-3 rounded-xl sm:rounded-2xl text-[10px] sm:text-[12px] font-black flex items-center gap-3 sm:gap-4">
                                <div className={`w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full animate-pulse ${isPaused ? 'bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.8)]' : status === 'active' ? 'bg-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.8)]' : status === 'drowsy' ? 'bg-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.8)]' : 'bg-amber-400 shadow-[0_0_15px_rgba(251,191,36,0.8)]'}`} />
                                <span className="tracking-[0.2em] sm:tracking-[0.3em] uppercase">{isPaused ? 'PAUSED' : status}</span>
                              </div>
                              {isPowerSaving && (
                                <div className="bg-emerald-500/20 backdrop-blur-md border border-emerald-500/30 px-4 py-2 rounded-xl text-[10px] text-emerald-400 font-black animate-pulse uppercase tracking-widest flex items-center gap-2">
                                  <BatteryLow size={12} /> Power Saving Active
                                </div>
                              )}
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-mono text-white/60">{new Date().toLocaleTimeString()}</div>
                            </div>
                          </div>

                          <div className="flex justify-between items-end gap-4 overflow-hidden">
                            <div className="flex flex-col gap-4 sm:gap-6 w-full max-w-[240px] sm:max-w-none">
                              <div className="flex flex-col gap-2 sm:gap-3">
                                <div className="flex justify-between items-end">
                                  <span className="text-[8px] sm:text-[10px] text-white/40 uppercase font-black tracking-[0.2em] sm:tracking-[0.3em] truncate pr-2">Fatigue Level</span>
                                  <span className={`text-[10px] sm:text-xs font-black ${(fatigueLevel * 100) > 70 ? 'text-rose-500' : 'text-cyan-400'}`}>{(fatigueLevel * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full sm:w-64 h-1.5 sm:h-2 bg-white/5 rounded-full overflow-hidden border border-white/10">
                                  <motion.div 
                                    className={`h-full transition-colors duration-300 ${fatigueLevel > 0.7 ? 'bg-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.8)]' : 'bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.8)]'}`}
                                    animate={{ width: `${fatigueLevel * 100}%` }}
                                  />
                                </div>
                              </div>
                              <div className="flex flex-col gap-2 sm:gap-3">
                                <div className="flex justify-between items-end">
                                  <span className="text-[8px] sm:text-[10px] text-white/40 uppercase font-black tracking-[0.2em] sm:tracking-[0.3em] truncate pr-2">Eye Closure</span>
                                  <span className={`text-[10px] sm:text-xs font-black ${(eyeClosure * 100) > 80 ? 'text-rose-500' : 'text-cyan-400'}`}>{(eyeClosure * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full sm:w-64 h-1.5 sm:h-2 bg-white/5 rounded-full overflow-hidden border border-white/10">
                                  <motion.div 
                                    className={`h-full transition-colors duration-300 ${eyeClosure > (1.0 - (sensitivity * 0.8 + 0.1)) ? 'bg-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.8)]' : 'bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.8)]'}`}
                                    animate={{ width: `${eyeClosure * 100}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                            <div className="hidden sm:flex flex-col items-end gap-2 shrink-0">
                              <div className="flex gap-2">
                                {[1,2,3,4,5,6].map(i => <div key={i} className="w-1.5 h-1.5 bg-cyan-500/30 rounded-full" />)}
                              </div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Bottom Control Bar */}
                  <div className="absolute bottom-4 sm:bottom-10 inset-x-2 sm:inset-x-0 flex justify-center z-50 pointer-events-none">
                    <AnimatePresence>
                      {isHudVisible && (
                        <motion.div 
                          initial={{ y: 80, opacity: 0 }}
                          animate={{ y: 0, opacity: 1 }}
                          exit={{ y: 80, opacity: 0 }}
                          className="px-5 py-3.5 bg-black/60 backdrop-blur-3xl border border-white/10 flex flex-wrap items-center justify-center gap-4 rounded-[32px] shadow-[0_20px_50px_rgba(0,0,0,0.5)] max-w-full pointer-events-auto"
                        >
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 bg-white/5 pr-4 pl-3 py-1.5 rounded-2xl border border-white/5">
                              <Sun size={14} className="text-amber-400" />
                              <input 
                                type="range"
                                min="10"
                                max="200"
                                value={brightness}
                                onChange={(e) => setBrightness(parseInt(e.target.value))}
                                className="w-16 sm:w-24 h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-white"
                              />
                            </div>
                            <div className="flex items-center gap-1.5 bg-white/5 px-2.5 py-1.5 rounded-2xl border border-white/5">
                              <ZoomIn size={14} className="text-cyan-400" />
                              <div className="flex items-center gap-1">
                                <button onClick={() => setZoomLevel(prev => Math.max(prev - 0.2, 1))} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                                  <Minus size={12} className="text-white" />
                                </button>
                                <span className="text-[10px] font-black text-white/60 w-8 text-center">{zoomLevel.toFixed(1)}x</span>
                                <button onClick={() => setZoomLevel(prev => Math.min(prev + 0.2, 4))} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                                  <Plus size={12} className="text-white" />
                                </button>
                              </div>
                            </div>
                          </div>

                          <div className="w-[1px] h-6 bg-white/10 hidden sm:block" />

                          <div className="flex items-center gap-2">
                            <button 
                              onClick={takeBreak}
                              className="flex items-center gap-2 px-5 py-2.5 bg-white/5 hover:bg-white/10 rounded-2xl text-[9px] font-black uppercase tracking-widest text-white transition-all border border-white/5"
                            >
                              <Coffee size={14} /> <span className="hidden sm:inline">Break</span>
                            </button>
                            <button 
                              onClick={() => setIsPaused(!isPaused)}
                              className={`p-2.5 rounded-2xl transition-all ${isPaused ? 'bg-emerald-500 text-white' : 'bg-white/5 text-white hover:bg-white/10'}`}
                            >
                              {isPaused ? <Play size={18} /> : <Pause size={18} />}
                            </button>
                            <button 
                              onClick={stopCamera}
                              className="p-2.5 bg-rose-600/20 border border-rose-500/20 rounded-2xl text-rose-500 hover:bg-rose-600/40 transition-all shadow-lg"
                            >
                              <RefreshCw size={18} />
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

              {/* Initial State Overlay */}
                  {status === 'ready' && (
                    <div className="absolute inset-0 flex items-center justify-center bg-[#0A0A0A]/95 backdrop-blur-3xl">
                      <div className="flex flex-col items-center gap-10">
                        <div className="w-24 h-24 bg-white/5 rounded-[40px] border border-white/10 flex items-center justify-center text-white/40 shadow-2xl">
                          <Camera size={48} />
                        </div>
                        <div className="text-center">
                          <h3 className="text-3xl font-black text-white tracking-tighter mb-3 uppercase">Vision Engine Offline</h3>
                          <p className="text-[10px] text-[#666] uppercase tracking-[0.4em] font-bold">Initialize secure feed to begin monitoring</p>
                        </div>
                        <button 
                          onClick={startCamera}
                          className="group relative px-12 py-6 bg-white text-black font-black rounded-3xl overflow-hidden transition-all hover:scale-105 active:scale-95 shadow-[0_30px_60px_rgba(255,255,255,0.1)]"
                        >
                          <span className="relative flex items-center gap-4 uppercase tracking-[0.2em] text-sm">
                            <Zap size={20} fill="currentColor" />
                            Activate Guard
                          </span>
                        </button>
                      </div>
                    </div>
                  )}

                  {status === 'initializing' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0A0A0A] gap-8">
                      <div className="relative">
                        <div className="w-20 h-20 border-4 border-white/5 rounded-full" />
                        <div className="absolute inset-0 w-20 h-20 border-4 border-t-cyan-400 rounded-full animate-spin shadow-[0_0_20px_rgba(34,211,238,0.4)]" />
                      </div>
                      <div className="text-center">
                        <span className="text-[12px] font-black tracking-[0.5em] text-cyan-400 uppercase animate-pulse">Synchronizing Neural Models</span>
                        <p className="text-[10px] text-[#444] mt-3 uppercase tracking-[0.3em] font-bold">Calibrating biometric sensors...</p>
                      </div>
                    </div>
                  )}

                  {status === 'error' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-rose-950/30 backdrop-blur-3xl p-12 text-center">
                      <div className="w-24 h-24 bg-rose-500/20 rounded-[40px] flex items-center justify-center text-rose-500 mb-8 border border-rose-500/30 shadow-2xl">
                        <AlertTriangle size={48} />
                      </div>
                      <h2 className="text-3xl font-black text-white mb-4 uppercase tracking-tighter">System Critical Error</h2>
                      <p className="text-sm text-rose-400/60 max-w-sm leading-relaxed font-bold uppercase tracking-widest mb-10">{errorMessage}</p>
                      <div className="flex flex-col sm:flex-row gap-4">
                        <button 
                          onClick={startCamera}
                          className="px-12 py-5 bg-white text-black font-black rounded-3xl hover:scale-105 active:scale-95 transition-all uppercase text-xs tracking-[0.2em] shadow-xl"
                        >
                          Retry Initialization
                        </button>
                        <button 
                          onClick={() => window.location.reload()}
                          className="px-12 py-5 bg-rose-600/20 border border-rose-500/50 text-rose-500 font-black rounded-3xl hover:bg-rose-600/30 transition-all uppercase text-xs tracking-[0.2em]"
                        >
                          Reboot Core
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Scroll Indicator */}
                {!isFullScreen && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 2 }}
                    className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-white/20"
                  >
                    <span className="text-[8px] font-black uppercase tracking-[0.5em]">Scroll for Data</span>
                    <motion.div 
                      animate={{ y: [0, 10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-[1px] h-8 bg-gradient-to-b from-white/20 to-transparent"
                    />
                  </motion.div>
                )}
              </section>

              {/* Content Section: Telemetry, Controls, Logs */}
              <div className="max-w-7xl mx-auto w-full p-6 md:p-12 grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column: Telemetry & History */}
                <div className="lg:col-span-8 flex flex-col gap-8">
                  {/* Telemetry Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-[40px] relative overflow-hidden group hover:border-cyan-500/30 transition-all shadow-xl">
                      <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Wifi size={48} className="text-cyan-400" />
                      </div>
                      <div className="text-[10px] text-[#444] uppercase font-black tracking-[0.3em] mb-4 flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.6)]" /> Latency
                      </div>
                      <div className="text-3xl font-black text-white tracking-tighter">{latency}ms</div>
                      <div className="mt-3 text-[9px] text-cyan-400/60 font-bold uppercase tracking-widest">Neural Processing</div>
                    </div>
                    
                    <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-[40px] relative overflow-hidden group hover:border-amber-500/30 transition-all shadow-xl">
                      <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Clock size={48} className="text-amber-400" />
                      </div>
                      <div className="text-[10px] text-[#444] uppercase font-black tracking-[0.3em] mb-4 flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.6)]" /> Alert Limit
                      </div>
                      <div className="text-3xl font-black text-white tracking-tighter">{(drowsyLimitMs / 1000).toFixed(1)}s</div>
                      <div className="mt-3 text-[9px] text-amber-400/60 font-bold uppercase tracking-widest">Closure Threshold</div>
                    </div>

                    <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-[40px] relative overflow-hidden group hover:border-emerald-500/30 transition-all shadow-xl">
                      <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Activity size={48} className="text-emerald-400" />
                      </div>
                      <div className="text-[10px] text-[#444] uppercase font-black tracking-[0.3em] mb-4 flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.6)]" /> Accuracy
                      </div>
                      <div className="text-3xl font-black text-emerald-400 tracking-tighter">99.8%</div>
                      <div className="mt-3 text-[9px] text-emerald-400/60 font-bold uppercase tracking-widest">Model Confidence</div>
                    </div>
                  </div>

                  {/* Last Incident Card */}
                  {history.length > 0 && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      className="bg-gradient-to-r from-rose-600/10 via-[#151619] to-[#151619] border border-rose-500/20 p-8 rounded-[40px] flex flex-col md:flex-row items-center justify-between shadow-2xl gap-8"
                    >
                      <div className="flex items-center gap-6 w-full md:w-auto">
                        <div className="w-16 h-16 bg-rose-500/10 rounded-[24px] flex items-center justify-center text-rose-500 border border-rose-500/20 shadow-lg">
                          <History size={28} />
                        </div>
                        <div>
                          <div className="text-[11px] text-rose-500 uppercase font-black tracking-[0.4em] mb-2">Recent Security Breach</div>
                          <div className="text-xl font-black text-white tracking-tight">{history[0].startTime}</div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-10 w-full md:w-auto justify-between md:justify-end">
                        <div className="text-right">
                          <div className="text-[10px] text-[#444] uppercase font-black tracking-widest mb-2">Duration</div>
                          <div className="text-3xl font-black text-rose-500 tracking-tighter">{(history[0].duration / 1000).toFixed(2)}s</div>
                        </div>
                        <button 
                          onClick={() => setViewMode('statistics')}
                          className="px-8 py-4 bg-white text-black text-[11px] font-black uppercase tracking-[0.2em] rounded-2xl hover:bg-rose-500 hover:text-white transition-all shadow-xl active:scale-95"
                        >
                          Archive
                        </button>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Right Column: Controls & Logs */}
                <div className="lg:col-span-4 flex flex-col gap-8">
                  {/* Control Panel */}
                  <div className="bg-[#151619] border border-[#1F1F1F] rounded-[40px] overflow-hidden shadow-2xl">
                    <div className="bg-[#1F1F1F]/50 px-10 py-6 border-b border-[#1F1F1F] flex justify-between items-center">
                      <span className="text-[11px] font-black tracking-[0.4em] uppercase text-white">Agent Config</span>
                      <Settings2 size={16} className="text-[#444]" />
                    </div>
                    <div className="p-10 flex flex-col gap-10">
                      <div className="space-y-6">
                        <div className="flex justify-between items-center">
                          <span className="text-[11px] text-[#666] uppercase font-black tracking-[0.2em]">Sensitivity</span>
                          <span className="text-sm font-black text-white">{(sensitivity * 100).toFixed(0)}%</span>
                        </div>
                        <p className="text-[9px] text-[#444] uppercase font-bold leading-tight">
                          Higher sensitivity makes the AI more strict. It will trigger alerts faster for smaller eye closures or slight head tilts.
                        </p>
                        <input 
                          type="range" 
                          min="0" 
                          max="1" 
                          step="0.01" 
                          value={sensitivity}
                          onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                          className="w-full h-2 bg-[#1F1F1F] rounded-full appearance-none cursor-pointer accent-white"
                        />
                      </div>

                      <div className="space-y-6">
                        <div className="flex justify-between items-center">
                          <span className="text-[11px] text-[#666] uppercase font-black tracking-[0.2em]">Alarm Volume</span>
                          <span className="text-sm font-black text-white">{(alarmVolume * 100).toFixed(0)}%</span>
                        </div>
                        <input 
                          type="range" 
                          min="0" 
                          max="1" 
                          step="0.01" 
                          value={alarmVolume}
                          onChange={(e) => setAlarmVolume(parseFloat(e.target.value))}
                          className="w-full h-2 bg-[#1F1F1F] rounded-full appearance-none cursor-pointer accent-white"
                        />
                      </div>

                      <div className="space-y-6">
                        <div className="flex justify-between items-center">
                          <span className="text-[11px] text-[#666] uppercase font-black tracking-[0.2em]">Response Time</span>
                          <span className="text-sm font-black text-white">{(drowsyLimitMs / 1000).toFixed(1)}s</span>
                        </div>
                        <input 
                          type="range" 
                          min="500" 
                          max="5000" 
                          step="100" 
                          value={drowsyLimitMs}
                          onChange={(e) => setDrowsyLimitMs(parseInt(e.target.value))}
                          className="w-full h-2 bg-[#1F1F1F] rounded-full appearance-none cursor-pointer accent-white"
                        />
                      </div>

                      <div className="pt-8 border-t border-[#1F1F1F] flex flex-col gap-4">
                        <button 
                          onClick={calibrateFace}
                          className="w-full py-5 bg-[#1F1F1F] hover:bg-[#252525] text-white text-[11px] font-black rounded-2xl transition-all uppercase tracking-[0.2em] flex items-center justify-center gap-4 border border-transparent hover:border-white/10"
                        >
                          <RefreshCw size={16} /> Calibrate Sensors
                        </button>
                        <button 
                          onClick={testAlarm}
                          className="w-full py-5 bg-[#1F1F1F] hover:bg-[#252525] text-white text-[11px] font-black rounded-2xl transition-all uppercase tracking-[0.2em] flex items-center justify-center gap-4 border border-transparent hover:border-white/10"
                        >
                          <Volume2 size={16} /> Test Audio
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* System Logs */}
                  <div className="bg-[#151619] border border-[#1F1F1F] rounded-[40px] overflow-hidden flex flex-col shadow-2xl">
                    <div className="bg-[#1F1F1F]/50 px-10 py-5 text-[11px] font-black tracking-[0.4em] uppercase text-white border-b border-[#1F1F1F]">
                      System Console
                    </div>
                    <div className="p-8 font-mono text-[11px] text-[#444] space-y-4 overflow-y-auto max-h-[400px] custom-scrollbar">
                      {logs.map((log, i) => (
                        <div key={i} className={`flex gap-4 ${log.includes('ALERT') ? 'text-rose-500' : log.includes('ERROR') ? 'text-rose-700' : ''}`}>
                          <span className="opacity-30">[{i.toString().padStart(3, '0')}]</span>
                          <span className="flex-1 leading-relaxed">{log}</span>
                        </div>
                      ))}
                      <div className="text-cyan-400 animate-pulse">_</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Statistics View - Creative Redesign */
            <div className="max-w-7xl mx-auto p-6 md:p-12">
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-12"
              >
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-l-4 border-rose-500 pl-6 py-2">
                <div className="flex flex-col gap-2">
                  <button 
                    onClick={() => setViewMode('dashboard')}
                    className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-[#666] hover:text-white transition-colors mb-2"
                  >
                    <LayoutDashboard size={12} /> Back to Dashboard
                  </button>
                  <h2 className="text-4xl font-black uppercase tracking-tighter text-white">Data Archive</h2>
                  <p className="text-xs text-rose-500/60 font-bold tracking-[0.2em] uppercase">Intelligence & Historical Analysis</p>
                </div>
                <div className="flex gap-3">
                  <div className="flex bg-[#151619] rounded-xl border border-[#1F1F1F] p-1">
                    <button 
                      onClick={exportData}
                      className="flex items-center gap-2 px-4 py-2 bg-rose-600 text-white rounded-lg text-[10px] font-black uppercase hover:bg-rose-500 transition-all shadow-lg shadow-rose-600/20"
                    >
                      <Download size={14} /> Export JSON
                    </button>
                    <div className="w-[1px] bg-[#1F1F1F] my-2" />
                    <button 
                      onClick={exportCSV}
                      className="flex items-center gap-2 px-4 py-2 text-white rounded-lg text-[10px] font-black uppercase hover:bg-white/5 transition-all"
                    >
                      <FileSpreadsheet size={14} /> CSV
                    </button>
                  </div>
                  <button 
                    onClick={() => setIsWipeConfirmOpen(true)}
                    className="flex items-center gap-2 px-6 py-3 bg-rose-600/10 border border-rose-600/20 text-rose-500 rounded-xl text-[11px] font-black uppercase hover:bg-rose-600/20 transition-colors"
                  >
                    <Trash2 size={14} /> Wipe Memory
                  </button>
                </div>
              </div>

              {/* Bento Grid Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="md:col-span-2 bg-[#151619] border border-[#1F1F1F] p-8 rounded-3xl relative overflow-hidden group">
                  <div className="absolute inset-0 opacity-10 pointer-events-none" 
                       style={{ backgroundImage: 'linear-gradient(45deg, #1F1F1F 25%, transparent 25%, transparent 50%, #1F1F1F 50%, #1F1F1F 75%, transparent 75%, transparent)', backgroundSize: '20px 20px' }} />
                  <div 
                    className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-40 transition-opacity cursor-pointer"
                    onClick={() => setViewMode('dashboard')}
                  >
                    <ShieldAlert size={80} className="text-rose-500" />
                  </div>
                  <div className="relative z-10">
                    <div className="text-[10px] text-rose-500 uppercase font-black tracking-[0.3em] mb-4 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
                      Total Breaches
                    </div>
                    <div className="text-7xl font-black text-white tracking-tighter">{history.length}</div>
                    <div className="mt-4 text-[10px] text-[#444] uppercase font-bold">Cumulative incidents recorded since initialization</div>
                  </div>
                </div>
                
                <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-3xl flex flex-col justify-between relative overflow-hidden group">
                  <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="text-[10px] text-cyan-400 uppercase font-black tracking-[0.3em]">24H Activity</div>
                  <div className="text-5xl font-black text-white mt-4 relative z-10">
                    {history.filter(e => Date.now() - e.timestamp < 86400000).length}
                  </div>
                  <div className="mt-4 text-[9px] text-[#444] uppercase font-bold relative z-10">Active alerts</div>
                </div>

                <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-3xl flex flex-col justify-between relative overflow-hidden group">
                  <div className="absolute inset-0 bg-amber-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="text-[10px] text-amber-400 uppercase font-black tracking-[0.3em]">Avg Response</div>
                  <div className="text-5xl font-black text-white mt-4 relative z-10">
                    {history.length > 0 
                      ? (history.reduce((acc, e) => acc + e.duration, 0) / history.length / 1000).toFixed(1)
                      : 0}s
                  </div>
                  <div className="mt-4 text-[9px] text-[#444] uppercase font-bold relative z-10">Seconds to alert</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Drowsiness Events Chart */}
                <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-3xl relative overflow-hidden">
                  <div className="flex justify-between items-center mb-8">
                    <div className="text-[10px] text-[#666] uppercase font-black tracking-[0.3em]">Temporal Distribution</div>
                    <div className="flex gap-2">
                      <div className="w-2 h-2 rounded-full bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
                    </div>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" vertical={false} />
                        <XAxis 
                          dataKey="date" 
                          stroke="#333" 
                          fontSize={9} 
                          tickLine={false} 
                          axisLine={false}
                          tick={{ fill: '#444' }}
                        />
                        <YAxis 
                          stroke="#333" 
                          fontSize={9} 
                          tickLine={false} 
                          axisLine={false}
                          allowDecimals={false}
                          tick={{ fill: '#444' }}
                        />
                        <Tooltip 
                          cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                          contentStyle={{ backgroundColor: '#0F0F0F', border: '1px solid #1F1F1F', borderRadius: '12px', fontSize: '10px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}
                          itemStyle={{ color: '#E0E0E0', fontWeight: 'bold' }}
                        />
                        <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.count > 0 ? '#F43F5E' : '#2A2A2A'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Fatigue Level (Line Chart) */}
                <div className="bg-[#151619] border border-[#1F1F1F] p-8 rounded-3xl relative overflow-hidden">
                  <div className="flex justify-between items-center mb-8">
                    <div className="text-[10px] text-[#666] uppercase font-black tracking-[0.3em]">Fatigue Profile (24H)</div>
                    <div className="flex gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]" />
                    </div>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={fatigueChartData}>
                        <defs>
                          <linearGradient id="colorFatigue" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#22D3EE" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" vertical={false} />
                        <XAxis 
                          dataKey="time" 
                          stroke="#333" 
                          fontSize={8} 
                          tickLine={false} 
                          axisLine={false}
                          tick={{ fill: '#444' }}
                        />
                        <YAxis 
                          stroke="#333" 
                          fontSize={9} 
                          tickLine={false} 
                          axisLine={false}
                          domain={[0, 100]}
                          tick={{ fill: '#444' }}
                          tickFormatter={(val) => `${val}%`}
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0F0F0F', border: '1px solid #1F1F1F', borderRadius: '12px', fontSize: '10px' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="fatigue" 
                          stroke="#22D3EE" 
                          strokeWidth={2}
                          fillOpacity={1} 
                          fill="url(#colorFatigue)" 
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Recent Activity Feed */}
                <div className="bg-[#151619] border border-[#1F1F1F] rounded-3xl overflow-hidden flex flex-col">
                  <div className="bg-[#1F1F1F]/50 px-6 py-4 text-[10px] font-black tracking-[0.3em] uppercase border-b border-[#1F1F1F]">
                    Live Feed
                  </div>
                  <div className="flex-1 overflow-y-auto max-h-[400px] custom-scrollbar">
                    {history.length > 0 ? (
                      <div className="divide-y divide-[#1F1F1F]">
                        {history.slice(0, 20).map((event) => (
                          <div key={event.id} className="p-6 hover:bg-white/[0.02] transition-colors group">
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-500 group-hover:bg-rose-500 group-hover:text-white transition-all">
                                  <Clock size={14} />
                                </div>
                                <div>
                                  <div className="text-[11px] font-black text-white">
                                    {event.startTime?.includes(',') ? event.startTime.split(',')[1] : event.startTime}
                                  </div>
                                  <div className="text-[9px] text-[#444] uppercase tracking-widest">
                                    {event.startTime?.includes(',') ? event.startTime.split(',')[0] : 'Recorded'}
                                  </div>
                                </div>
                              </div>
                              <div className="text-[10px] font-black text-rose-500 bg-rose-500/10 px-2 py-0.5 rounded uppercase">
                                {(event.duration / 1000).toFixed(2)}s
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center p-12 text-center opacity-20 grayscale">
                        <BarChart3 size={48} className="mb-4" />
                        <div className="text-[10px] font-black uppercase tracking-[0.3em]">No Data Found</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Raw Data Table - Creative View */}
              <div className="bg-[#151619] border border-[#1F1F1F] rounded-3xl overflow-hidden">
                <div className="bg-[#1F1F1F]/50 px-8 py-6 flex justify-between items-center border-b border-[#1F1F1F]">
                  <div className="text-[10px] font-black tracking-[0.3em] uppercase">Detailed Log Explorer</div>
                  <div className="text-[9px] text-[#444] uppercase tracking-widest">{history.length} Entries Total</div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="text-[9px] text-[#444] uppercase tracking-[0.2em] border-b border-[#1F1F1F]">
                        <th className="px-8 py-4 font-black">Event ID</th>
                        <th className="px-8 py-4 font-black">Timestamp</th>
                        <th className="px-8 py-4 font-black">Duration</th>
                        <th className="px-8 py-4 font-black text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody className="text-[10px]">
                      {history.length > 0 ? (
                        history.map((event) => (
                          <tr key={event.id} className="border-b border-[#1F1F1F]/50 hover:bg-white/[0.01] transition-colors group">
                            <td className="px-8 py-4 font-mono text-[#666] group-hover:text-white transition-colors">{event.id}</td>
                            <td className="px-8 py-4 text-[#888]">{event.startTime}</td>
                            <td className="px-8 py-4 font-black text-white">{(event.duration / 1000).toFixed(3)}s</td>
                            <td className="px-8 py-4 text-right">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-rose-500 mr-2 shadow-[0_0_5px_rgba(244,63,94,0.5)]" />
                              <span className="text-rose-500 font-black uppercase tracking-widest">Breach</span>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="px-8 py-20 text-center text-[#333] uppercase tracking-[0.5em]">Empty Archive</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
              </motion.div>
            </div>
          )}
        </div>
      </main>

      {/* Global Alarm Overlay */}
      <AnimatePresence>
        {isAlarming && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-rose-600/40 backdrop-blur-xl p-4"
          >
            <motion.div 
              initial={{ scale: 0.8, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.8, opacity: 0, y: 20 }}
              className="bg-[#0F0F0F] border-2 border-rose-600 p-8 md:p-16 rounded-[40px] md:rounded-[60px] shadow-[0_0_150px_rgba(244,63,94,0.6)] flex flex-col items-center gap-8 text-center max-w-2xl w-full"
            >
              <div className="w-24 h-24 md:w-32 md:h-32 bg-rose-600 rounded-full flex items-center justify-center animate-pulse shadow-[0_0_60px_rgba(244,63,94,0.8)]">
                <ShieldAlert size={64} className="text-white" />
              </div>
              <div className="space-y-4">
                <h2 className="text-6xl md:text-8xl font-black text-white tracking-tighter uppercase leading-none">Wake Up!</h2>
                <p className="text-rose-500 font-black tracking-[0.5em] uppercase text-xs md:text-sm">Drowsiness Threshold Exceeded</p>
              </div>
              <div className="w-full max-w-xs h-1 bg-white/10 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-rose-500"
                  animate={{ width: ['0%', '100%'] }}
                  transition={{ duration: 0.6, repeat: Infinity }}
                />
              </div>
              <button 
                onClick={stopAlarm}
                className="group relative mt-6 px-16 py-6 bg-rose-600 text-white font-black rounded-3xl hover:bg-rose-500 transition-all shadow-2xl shadow-rose-600/40 uppercase text-sm tracking-[0.3em] active:scale-95"
              >
                I am awake
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-[#151619] border border-[#1F1F1F] w-full max-w-md rounded-[32px] overflow-hidden shadow-2xl"
            >
              <div className="bg-[#1F1F1F]/50 px-8 py-6 border-b border-[#1F1F1F] flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <Settings2 className="text-rose-500" size={20} />
                  <h3 className="text-sm font-black uppercase tracking-widest text-white">System Settings</h3>
                </div>
                <button onClick={() => setIsSettingsOpen(false)} className="text-[#444] hover:text-white transition-colors">
                  <X size={20} />
                </button>
              </div>
              <div className="p-8 space-y-6 max-h-[60vh] overflow-y-auto custom-scrollbar">
                {/* Visual Settings */}
                <div className="grid grid-cols-1 gap-6 pb-6 border-b border-[#1F1F1F]">
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-black uppercase tracking-widest text-[#666]">Screen Brightness</span>
                      <span className="text-[10px] font-black text-amber-500">{brightness}%</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <Sun size={14} className="text-amber-400" />
                      <input 
                        type="range" min="10" max="200" value={brightness}
                        onChange={(e) => setBrightness(parseInt(e.target.value))}
                        className="flex-1 h-1 bg-[#1F1F1F] rounded-full appearance-none accent-amber-500 cursor-pointer"
                      />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-black uppercase tracking-widest text-[#666]">Zoom Magnification</span>
                      <span className="text-[10px] font-black text-cyan-400">{zoomLevel.toFixed(1)}x</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <ZoomIn size={14} className="text-cyan-400" />
                      <input 
                        type="range" min="1" max="3" step="0.1" value={zoomLevel}
                        onChange={(e) => setZoomLevel(parseFloat(e.target.value))}
                        className="flex-1 h-1 bg-[#1F1F1F] rounded-full appearance-none accent-cyan-400 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-end">
                    <span className="text-[10px] font-black uppercase tracking-widest text-[#666]">Alarm Volume</span>
                    <span className="text-[10px] font-black text-rose-500">{(alarmVolume * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="1" step="0.01" 
                    value={alarmVolume} 
                    onChange={(e) => setAlarmVolume(parseFloat(e.target.value))}
                    className="w-full h-1 bg-[#1F1F1F] rounded-lg appearance-none cursor-pointer accent-rose-500"
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-end">
                    <span className="text-[10px] font-black uppercase tracking-widest text-[#666]">Detection Sensitivity</span>
                    <span className="text-[10px] font-black text-cyan-400">{(sensitivity * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0.1" max="0.9" step="0.01" 
                    value={sensitivity} 
                    onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                    className="w-full h-1 bg-[#1F1F1F] rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-end">
                    <span className="text-[10px] font-black uppercase tracking-widest text-[#666]">Drowsy Threshold</span>
                    <span className="text-[10px] font-black text-amber-400">{(drowsyLimitMs / 1000).toFixed(1)}s</span>
                  </div>
                  <input 
                    type="range" min="500" max="5000" step="100" 
                    value={drowsyLimitMs} 
                    onChange={(e) => setDrowsyLimitMs(parseInt(e.target.value))}
                    className="w-full h-1 bg-[#1F1F1F] rounded-lg appearance-none cursor-pointer accent-amber-400"
                  />
                </div>
                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-black uppercase tracking-widest text-white">System Operation</span>
                      <span className="text-[8px] text-[#666] uppercase font-bold">Control active monitoring state</span>
                    </div>
                    <div className="flex gap-2">
                      <button 
                        onClick={() => setIsPaused(!isPaused)}
                        className={`px-4 py-2 rounded-xl text-[9px] font-black uppercase transition-all ${isPaused ? 'bg-emerald-500 text-white' : 'bg-[#1F1F1F] text-white hover:bg-white/5'}`}
                      >
                        {isPaused ? 'Resume' : 'Pause'}
                      </button>
                      <button 
                        onClick={stopCamera}
                        className="px-4 py-2 bg-rose-600/20 border border-rose-500/30 text-rose-500 rounded-xl text-[9px] font-black uppercase hover:bg-rose-600/30 transition-all font-mono"
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-black uppercase tracking-widest text-white">Focus Mode</span>
                      <span className="text-[8px] text-[#666] uppercase font-bold">Keeps your face clear automatically</span>
                    </div>
                    <button 
                      onClick={() => setIsAutoFocus(!isAutoFocus)}
                      className={`w-12 h-6 rounded-full transition-all relative ${isAutoFocus ? 'bg-cyan-500' : 'bg-[#1F1F1F]'}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${isAutoFocus ? 'left-7' : 'left-1'}`} />
                    </button>
                  </div>
                </div>
                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-black uppercase tracking-widest text-white">Save Battery</span>
                      <span className="text-[8px] text-[#666] uppercase font-bold">Dims screen to save power</span>
                    </div>
                    <button 
                      onClick={() => setIsPowerSaving(!isPowerSaving)}
                      className={`w-12 h-6 rounded-full transition-all relative ${isPowerSaving ? 'bg-emerald-500' : 'bg-[#1F1F1F]'}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${isPowerSaving ? 'left-7' : 'left-1'}`} />
                    </button>
                  </div>
                </div>
                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-black uppercase tracking-widest text-white">Auto Power Saver</span>
                      <span className="text-[8px] text-[#666] uppercase font-bold">Enable when app is in background</span>
                    </div>
                    <button 
                      onClick={() => setAutoPowerSaving(!autoPowerSaving)}
                      className={`w-12 h-6 rounded-full transition-all relative ${autoPowerSaving ? 'bg-cyan-500' : 'bg-[#1F1F1F]'}`}
                    >
                      <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${autoPowerSaving ? 'left-7' : 'left-1'}`} />
                    </button>
                  </div>
                </div>
                <button 
                  onClick={testAlarm}
                  className="w-full py-4 bg-[#1F1F1F] hover:bg-[#25262b] text-white rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] transition-all border border-[#2a2a2a]"
                >
                  Test Alarm System
                </button>
              </div>
              <div className="bg-[#1F1F1F]/20 p-6 border-t border-[#1F1F1F]">
                <button 
                  onClick={() => setIsSettingsOpen(false)}
                  className="w-full py-4 bg-white text-black rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] hover:bg-opacity-90 transition-all"
                >
                  Apply & Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Break Overlay */}
      <AnimatePresence>
        {isBreakActive && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-[#0A0A0A]/95 backdrop-blur-2xl p-4"
          >
            <div className="max-w-md w-full text-center space-y-8">
              <div className="w-24 h-24 bg-cyan-500/10 rounded-[40px] flex items-center justify-center text-cyan-400 mx-auto border border-cyan-500/20 shadow-2xl">
                <Coffee size={48} />
              </div>
              <div className="space-y-2">
                <h2 className="text-4xl font-black text-white uppercase tracking-tighter">Break Time</h2>
                <p className="text-xs text-[#666] uppercase tracking-[0.3em] font-bold">Rest your eyes and stretch</p>
              </div>
              <button 
                onClick={endBreak}
                className="w-full py-6 bg-white text-black font-black rounded-3xl uppercase tracking-[0.2em] text-sm hover:scale-105 transition-all shadow-2xl"
              >
                Resume Monitoring
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Safety Guide Modal */}
      <AnimatePresence>
        {isSafetyGuideOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/90 backdrop-blur-xl"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-[#151619] border border-[#1F1F1F] w-full max-w-2xl rounded-[40px] overflow-hidden shadow-2xl flex flex-col max-h-[90vh]"
            >
              <div className="bg-[#1F1F1F]/50 px-10 py-8 border-b border-[#1F1F1F] flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-rose-500/10 rounded-xl flex items-center justify-center text-rose-500">
                    <ShieldAlert size={24} />
                  </div>
                  <div>
                    <h3 className="text-lg font-black uppercase tracking-tighter text-white leading-none">Safety Protocol</h3>
                    <p className="text-[9px] text-[#666] uppercase tracking-widest mt-1">Operational Guidelines v2.1</p>
                  </div>
                </div>
                <button onClick={() => setIsSafetyGuideOpen(false)} className="p-2 text-[#444] hover:text-white transition-colors">
                  <X size={24} />
                </button>
              </div>
              
              <div className="p-10 overflow-y-auto custom-scrollbar space-y-10">
                <section className="space-y-6">
                  <h4 className="text-[11px] font-black text-rose-500 uppercase tracking-[0.3em] flex items-center gap-3">
                    <AlertTriangle size={14} /> Critical Warning
                  </h4>
                  <p className="text-sm text-[#888] leading-relaxed">
                    Dream Catcher is an assistive tool and <span className="text-white font-bold">NOT a substitute for responsible driving or operation of heavy machinery.</span> If you feel tired, pull over immediately in a safe location.
                  </p>
                </section>

                <section className="space-y-6">
                  <h4 className="text-[11px] font-black text-cyan-400 uppercase tracking-[0.3em] flex items-center gap-3">
                    <BookOpen size={14} /> How to use (Simple Guide)
                  </h4>
                  <div className="space-y-4 text-xs text-[#888] leading-relaxed">
                    <div className="bg-[#1F1F1F]/20 p-4 rounded-2xl border border-[#1F1F1F]">
                      <p className="text-white font-bold mb-2 uppercase tracking-widest text-[10px]">Step 1: Start</p>
                      <p>Click the big "Activate Guard" button. If your computer asks to use the camera, click "Allow".</p>
                    </div>
                    <div className="bg-[#1F1F1F]/20 p-4 rounded-2xl border border-[#1F1F1F]">
                      <p className="text-white font-bold mb-2 uppercase tracking-widest text-[10px]">Step 2: Position</p>
                      <p>Sit so you can see yourself clearly in the box. Make sure your face is well-lit.</p>
                    </div>
                    <div className="bg-[#1F1F1F]/20 p-4 rounded-2xl border border-[#1F1F1F]">
                      <p className="text-white font-bold mb-2 uppercase tracking-widest text-[10px]">Step 3: The Alarm</p>
                      <p>If you close your eyes for too long, a high-frequency alarm will play. The HUD will also flash warnings if your head posture deviates significantly (e.g., nodding off).</p>
                    </div>
                    <div className="bg-[#1F1F1F]/20 p-4 rounded-2xl border border-[#1F1F1F]">
                      <p className="text-white font-bold mb-2 uppercase tracking-widest text-[10px]">Step 4: Customization</p>
                      <p>Use the "Eye" icon to toggle a Clean View. Brightness and Zoom can be adjusted in the bottom bar or within System Settings for optimal clarity.</p>
                    </div>
                  </div>
                </section>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-[#1F1F1F]/30 p-6 rounded-3xl border border-[#1F1F1F] space-y-4">
                    <div className="w-10 h-10 bg-cyan-500/10 rounded-xl flex items-center justify-center text-cyan-400">
                      <Maximize size={20} />
                    </div>
                    <h5 className="text-[10px] font-black text-white uppercase tracking-widest">Optimal Setup</h5>
                    <ul className="text-[10px] text-[#666] space-y-2 list-disc pl-4 uppercase font-bold tracking-wider">
                      <li>Position camera at eye level</li>
                      <li>Ensure face is well-lit</li>
                      <li>Maintain 50-70cm distance</li>
                      <li>Avoid wearing dark sunglasses</li>
                    </ul>
                  </div>

                  <div className="bg-[#1F1F1F]/30 p-6 rounded-3xl border border-[#1F1F1F] space-y-4">
                    <div className="w-10 h-10 bg-amber-500/10 rounded-xl flex items-center justify-center text-amber-400">
                      <Zap size={20} />
                    </div>
                    <h5 className="text-[10px] font-black text-white uppercase tracking-widest">Enhanced Capabilities</h5>
                    <ul className="text-[10px] text-[#666] space-y-2 list-disc pl-4 uppercase font-bold tracking-wider">
                      <li>Real-time Ocular Tracking</li>
                      <li>Postural Alignment HUD</li>
                      <li>24H Fatigue Trend Analysis</li>
                      <li>Adaptive Power Management</li>
                    </ul>
                  </div>
                </div>

                <section className="bg-rose-500/5 border border-rose-500/10 p-8 rounded-3xl space-y-4">
                  <div className="flex items-center gap-3 text-rose-500">
                    <Info size={18} />
                    <span className="text-[11px] font-black uppercase tracking-widest">Legal Disclaimer</span>
                  </div>
                  <p className="text-[10px] text-[#666] leading-relaxed uppercase font-bold tracking-wider">
                    By using this application, you acknowledge that the developers are not liable for any accidents, injuries, or damages resulting from the use or failure of this software. Always prioritize physical rest over software monitoring.
                  </p>
                </section>
              </div>

              <div className="p-8 bg-[#1F1F1F]/20 border-t border-[#1F1F1F]">
                <button 
                  onClick={() => setIsSafetyGuideOpen(false)}
                  className="w-full py-5 bg-white text-black font-black rounded-2xl uppercase tracking-[0.2em] text-xs hover:bg-opacity-90 transition-all shadow-xl"
                >
                  I Understand & Accept
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Wipe Confirmation Modal */}
      <AnimatePresence>
        {isWipeConfirmOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/90 backdrop-blur-md"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-[#151619] border border-rose-500/30 w-full max-w-sm rounded-[32px] overflow-hidden shadow-[0_0_50px_rgba(244,63,94,0.2)]"
            >
              <div className="p-10 text-center">
                <div className="w-20 h-20 bg-rose-500/10 rounded-full flex items-center justify-center text-rose-500 mx-auto mb-6">
                  <Trash2 size={40} />
                </div>
                <h3 className="text-xl font-black uppercase tracking-tighter text-white mb-2">Wipe Memory?</h3>
                <p className="text-xs text-[#666] leading-relaxed mb-8">
                  This action will permanently delete all historical data and logs. This cannot be undone.
                </p>
                <div className="flex flex-col gap-3">
                  <button 
                    onClick={clearHistory}
                    className="w-full py-4 bg-rose-600 text-white rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] hover:bg-rose-500 transition-all shadow-lg shadow-rose-600/20"
                  >
                    Confirm Wipe
                  </button>
                  <button 
                    onClick={() => setIsWipeConfirmOpen(false)}
                    className="w-full py-4 bg-[#1F1F1F] text-[#666] hover:text-white rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer */}
      <footer className="h-10 border-t border-[#1F1F1F] bg-[#0F0F0F] px-6 flex items-center justify-between text-[9px] text-[#444] uppercase tracking-widest">
        <div>© 2026 Dream Catcher Systems Inc.</div>
        <div className="flex gap-6">
          <span className="opacity-50 tracking-tighter">Privacy Secured</span>
        </div>
      </footer>
    </div>
  );
}

