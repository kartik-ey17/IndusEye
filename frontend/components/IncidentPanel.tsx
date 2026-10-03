import { Incident } from "../lib/api";
import IncidentCard from "./IncidentCard";

export default function IncidentPanel({ incidents, newIds }: { incidents: Incident[]; newIds: Set<string> }) {
  return <section className="panel incidents"><div className="panel-title"><span>INCIDENTS</span><small>{incidents.length} recorded</small></div>{incidents.length ? <div className="incident-list">{incidents.map((incident) => <IncidentCard key={incident.id} incident={incident} fresh={newIds.has(incident.id)} />)}</div> : <div className="empty">No weapon incident indicators recorded.<br />The stream remains private and monitored.</div>}</section>;
}
