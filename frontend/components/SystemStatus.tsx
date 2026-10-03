import { Metrics } from "../lib/api";

export default function SystemStatus({ online, model, metrics, locationMode }: { online: boolean; model: string; metrics: Metrics; locationMode: string }) {
  return <section className="panel status"><div className="panel-title"><span>SYSTEM STATUS</span><small className={online ? "good" : "bad"}>{online ? "ONLINE" : "OFFLINE"}</small></div><dl><div><dt>MODEL</dt><dd>{model || "Unavailable"}</dd></div><div><dt>PRIVACY</dt><dd className="good">ACTIVE</dd></div><div><dt>FPS</dt><dd>{metrics.fps?.toFixed(1) ?? "—"}</dd></div><div><dt>LATENCY</dt><dd>{metrics.total_latency_ms ? `${metrics.total_latency_ms.toFixed(0)} ms` : "—"}</dd></div><div><dt>INFERENCE</dt><dd>{metrics.inference_latency_ms ? `${metrics.inference_latency_ms.toFixed(0)} ms` : "—"}</dd></div><div><dt>LOCATION</dt><dd>{locationMode}</dd></div></dl></section>;
}
