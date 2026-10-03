"use client";

import { useCallback, useEffect, useState } from "react";
import IncidentPanel from "../components/IncidentPanel";
import LiveCamera from "../components/LiveCamera";
import SystemStatus from "../components/SystemStatus";
import { API_BASE, FrameResult, Incident, Metrics, request } from "../lib/api";

type Health = { status: string; model?: { class_names?: string[]; weights?: string }; model_error?: string | null };

export default function Dashboard() {
  const [online, setOnline] = useState(false);
  const [model, setModel] = useState("Checking backend…");
  const [metrics, setMetrics] = useState<Metrics>({});
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState("Connecting to local SentinelAI backend…");
  const [locationMode, setLocationMode] = useState("Demo fallback");

  const refresh = useCallback(async () => {
    try {
      const [health, rows] = await Promise.all([request<Health>("/api/health"), request<Incident[]>("/api/incidents")]);
      setOnline(health.status === "ok");
      setModel(health.model?.weights?.split(/[\\/]/).pop() || health.model_error || "Model unavailable");
      setIncidents((previous) => { const previousIds = new Set(previous.map((item) => item.id)); setNewIds(new Set(rows.filter((item) => !previousIds.has(item.id)).map((item) => item.id))); return rows; });
      setMessage(health.status === "ok" ? "Backend connected" : `Backend degraded: ${health.model_error || "model unavailable"}`);
    } catch { setOnline(false); setModel("Backend offline"); setMessage(`Cannot reach ${API_BASE}. Start FastAPI first.`); }
  }, []);

  useEffect(() => { refresh(); const timer = setInterval(refresh, 3000); return () => clearInterval(timer); }, [refresh]);
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(async (position) => {
      try { await request("/api/location", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ latitude: position.coords.latitude, longitude: position.coords.longitude }) }); setLocationMode("Browser location"); }
      catch { setLocationMode("Demo fallback"); }
    }, () => setLocationMode("Demo fallback (permission denied)"), { enableHighAccuracy: false, timeout: 6000 });
  }, []);

  const onFrame = useCallback((frame: FrameResult) => { setMetrics(frame.metrics); if (frame.incident) refresh(); }, [refresh]);
  const onError = useCallback((error: string) => setMessage(error), []);

  return <main><header><div><p className="eyebrow">SENTINELAI / OPERATOR CONSOLE</p><h1>Privacy-first incident monitoring</h1></div><div className={`connection ${online ? "online" : "offline"}`}>{online ? "● BACKEND ONLINE" : "● BACKEND OFFLINE"}</div></header><p className="flow">LIVE VIDEO <b>→</b> FACE BLUR <b>→</b> WEAPON INDICATOR <b>→</b> INCIDENT <b>→</b> EVIDENCE <b>→</b> LOCATION</p><p className="message">{message}</p><div className="dashboard"><LiveCamera onFrame={onFrame} onError={onError} /><aside><SystemStatus online={online} model={model} metrics={metrics} locationMode={locationMode} /><IncidentPanel incidents={incidents} newIds={newIds} /></aside></div></main>;
}
