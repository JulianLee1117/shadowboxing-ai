import type { Session } from "./types";
import type { RoundAnalysisReport } from "./roundAnalysis";

const DB_NAME = "corner-local-v1";
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let abandoned = false;
    const request = indexedDB.open(DB_NAME, 2);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("sessions"))
        db.createObjectStore("sessions", { keyPath: "id" });
      if (!db.objectStoreNames.contains("analyses"))
        db.createObjectStore("analyses", { keyPath: "sourceSessionId" });
    };
    request.onsuccess = () => {
      if (abandoned) {
        request.result.close();
        return;
      }
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onblocked = () => {
      abandoned = true;
      reject(new Error("Close other Corner tabs, then retry saving."));
    };
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
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(["sessions", "analyses"], "readwrite");
    tx.objectStore("sessions").delete(id);
    tx.objectStore("analyses").delete(id);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(new Error("Could not delete this round. Please retry."));
    };
  });
}

/** Derived video analysis is separate from original evidence and annotations. */
export async function loadRoundAnalysis(
  id: string,
): Promise<RoundAnalysisReport | undefined> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("analyses", "readonly");
    const request = tx.objectStore("analyses").get(id);
    tx.oncomplete = () => {
      db.close();
      resolve(request.result);
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(new Error("Could not load this round’s analysis."));
    };
  });
}

export async function saveRoundAnalysis(
  report: RoundAnalysisReport,
): Promise<void> {
  const db = await database();
  return new Promise((resolve, reject) => {
    // Check existence and write in one transaction: a completed background job
    // must never resurrect analysis after its round was deleted elsewhere.
    const tx = db.transaction(["sessions", "analyses"], "readwrite");
    const exists = tx.objectStore("sessions").getKey(report.sourceSessionId);
    let missing = false;
    exists.onsuccess = () => {
      if (exists.result === undefined) {
        missing = true;
        tx.abort();
      } else tx.objectStore("analyses").put(report);
    };
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(
        new Error(
          missing
            ? "Save the original round before saving its analysis."
            : "Analysis could not be saved. Export it before closing.",
        ),
      );
    };
  });
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
