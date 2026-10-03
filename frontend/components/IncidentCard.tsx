import { evidenceUrl, Incident } from "../lib/api";

export default function IncidentCard({ incident, fresh }: { incident: Incident; fresh: boolean }) {
  const evidence = evidenceUrl(incident);
  const coordinates = incident.latitude !== null && incident.longitude !== null ? `${incident.latitude.toFixed(4)}, ${incident.longitude.toFixed(4)}` : "Location unavailable";
  return <article className={`incident-card ${fresh ? "fresh" : ""}`}><div className="incident-top"><strong>{incident.event_type.replace(/_/g, " ")}</strong><span>{Math.round(incident.confidence * 100)}%</span></div><p className="weapon">{incident.weapon || "Weapon indicator"}</p><p>{new Date(incident.timestamp).toLocaleString()} · {incident.camera_id}</p><p>{coordinates}</p>{evidence ? <a href={evidence} target="_blank" rel="noreferrer"><img src={evidence} alt={`Evidence for ${incident.id}`} /><span>Open evidence</span></a> : <div className="missing-evidence">Evidence image unavailable</div>}</article>;
}
