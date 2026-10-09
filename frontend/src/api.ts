import { useEffect, useState } from 'react';

export class ApiError extends Error {
  constructor(message: string, public requestId: string | null = null, public offline = false) { super(message); }
}
export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  const timer = window.setTimeout(abort, 10000);
  try {
    const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new ApiError(data.error?.message ?? `Request failed (${response.status})`, data.request_id ?? response.headers.get('X-Request-ID'), response.status === 502 || response.status === 504);
    return data as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError('Backend unreachable or request timed out.', null, true);
  } finally { window.clearTimeout(timer); signal?.removeEventListener('abort', abort); }
}
export function active(status?: string) { return status === 'queued' || status === 'running'; }
export function usePoll<T extends { status: string }>(path: string | null, connected: boolean, epoch: number, fail: (error: ApiError) => void) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => { setData(null); setError(null); }, [path]);
  useEffect(() => {
    if (!path || !connected) { setLoading(false); return; }
    const controller = new AbortController(); let timer: number | undefined; let alive = true;
    const poll = async () => {
      setLoading(true); setError(null);
      try { const result = await api<T>(path, undefined, controller.signal); if (!alive) return; setData(result); if (active(result.status)) timer = window.setTimeout(poll, 2000); }
      catch (error) { if (alive) { const e = error as ApiError; setError(e); fail(e); } }
      finally { if (alive) setLoading(false); }
    };
    void poll();
    return () => { alive = false; controller.abort(); window.clearTimeout(timer); };
  }, [path, connected, epoch, fail]);
  return { data, error, loading };
}
