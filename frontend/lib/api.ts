export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export type Metrics = { fps?: number; total_latency_ms?: number; inference_latency_ms?: number; preprocessing_latency_ms?: number };
export type Incident = { id: string; event_type: string; confidence: number; timestamp: string; camera_id: string; latitude: number | null; longitude: number | null; evidence_url?: string | null; weapon?: string | null };
export type FrameResult = { display_frame_data_url: string; detections: unknown[]; faces: number[][]; incident: Incident | null; metrics: Metrics };

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}

export const evidenceUrl = (incident: Incident) => incident.evidence_url ? `${API_BASE}${incident.evidence_url}` : null;
