import type { Session } from "./types";

const DB_NAME = "corner-local-v1";
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore("sessions", { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(
        new Error(
          "Local storage is unavailable. You can still export this round.",
        ),
      );
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  op: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("sessions", mode);
    const request = op(tx.objectStore("sessions"));
    tx.oncomplete = () => {
      resolve(request.result);
      db.close();
    };
    tx.onerror = tx.onabort = () => {
      reject(
        new Error(
          "Could not save locally. Storage may be full; export your round before closing.",
        ),
      );
      db.close();
    };
  });
}
export async function saveSession(session: Session) {
  await transaction("readwrite", (store) => store.put(session));
}
export async function listSessions(): Promise<Session[]> {
  const all = await transaction<Session[]>("readonly", (store) =>
    store.getAll(),
  );
  return all.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}
export async function deleteSession(id: string) {
  await transaction("readwrite", (store) => store.delete(id));
}
export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function exportSession(session: Session) {
  const { video: _video, ...data } = session;
  downloadBlob(
    new Blob([JSON.stringify(data)], { type: "application/json" }),
    `corner-${session.id}.json`,
  );
}
export function percentile(values: number[], p: number): number {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length
    ? sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)]
    : 0;
}
export function formatTime(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}
