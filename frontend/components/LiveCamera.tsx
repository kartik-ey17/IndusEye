"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE, FrameResult } from "../lib/api";

type Props = { onFrame: (frame: FrameResult) => void; onError: (message: string) => void };

export default function LiveCamera({ onFrame, onError }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [display, setDisplay] = useState<string | null>(null);
  const [cameraState, setCameraState] = useState("Starting camera…");

  useEffect(() => {
    let stream: MediaStream | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    const sendFrame = async () => {
      if (stopped) return;
      // Some cameras need longer than the initial 200 ms before dimensions are available.
      if (!video.current || !canvas.current || video.current.readyState < 2) {
        timer = setTimeout(sendFrame, 100);
        return;
      }
      const v = video.current;
      const width = Math.min(v.videoWidth || 640, 640);
      const height = Math.round(width * (v.videoHeight || 480) / (v.videoWidth || 640));
      canvas.current.width = width; canvas.current.height = height;
      canvas.current.getContext("2d")?.drawImage(v, 0, 0, width, height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.current?.toBlob(resolve, "image/jpeg", 0.7));
      if (!blob) return;
      try {
        const response = await fetch(`${API_BASE}/api/process-frame`, { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
        if (!response.ok) throw new Error(`Backend returned ${response.status}`);
        const result = (await response.json()) as FrameResult;
        setDisplay(result.display_frame_data_url); onFrame(result); setCameraState("Live — privacy filter active");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Frame processing failed";
        setCameraState("Backend unavailable — camera is not displayed"); onError(message);
      } finally { if (!stopped) timer = setTimeout(sendFrame, 80); }
    };
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((value) => { stream = value; if (video.current) video.current.srcObject = value; setCameraState("Camera connected — processing…"); timer = setTimeout(sendFrame, 200); })
      .catch(() => { setCameraState("Camera permission denied or unavailable"); onError("Camera permission denied or unavailable"); });
    return () => { stopped = true; if (timer) clearTimeout(timer); stream?.getTracks().forEach((track) => track.stop()); };
  }, [onFrame, onError]);

  return <section className="panel live-panel"><div className="panel-title"><span>LIVE CAMERA</span><small>{cameraState}</small></div><div className="feed">{display ? <img src={display} alt="Privacy-preserving processed camera feed" /> : <div className="feed-placeholder">Awaiting privacy-safe processed frames</div>}<video ref={video} muted playsInline className="capture-source" /><canvas ref={canvas} hidden /></div><p className="privacy-note">RAW CAMERA NEVER RENDERS IN THIS DASHBOARD · FACES BLURRED SERVER-SIDE</p></section>;
}
