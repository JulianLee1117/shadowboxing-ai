import type { Session } from "./types";
import type { RoundAnalysisReport } from "./roundAnalysis";

const DB_NAME = "corner-local-v1";
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let abandoned = false;
    const request = indexedDB.open(DB_NAME, 3);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("sessions"))
        db.createObjectStore("sessions", { keyPath: "id" });
      if (!db.objectStoreNames.contains("analyses"))
        db.createObjectStore("analyses", { keyPath: "sourceSessionId" });
      if (!db.objectStoreNames.contains("trash"))
        db.createObjectStore("trash", { keyPath: "id" });
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
/** A small tombstone keeps the original video, evidence and report recoverable. */
export interface TrashedRound {
  id: string;
  deletedAt: string;
  createdAt: string;
  durationMs: number;
  eventCount: number;
  hasVideo: boolean;
}
export async function saveSession(
  session: Session,
  options: { requireExisting?: boolean } = {},
) {
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(["sessions", "trash"], "readwrite");
    const exists = tx.objectStore("sessions").getKey(session.id);
    const removed = tx.objectStore("trash").getKey(session.id);
    let inTrash = false;
    let missing = false;
    removed.onsuccess = () => {
      if (removed.result !== undefined) {
        inTrash = true;
        tx.abort();
      } else if (options.requireExisting && exists.result === undefined) {
        // An annotation edit in a stale tab must not recreate a purged video.
        missing = true;
        tx.abort();
      } else tx.objectStore("sessions").put(session);
    };
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(
        new Error(
          inTrash
            ? "Restore this round from Recently deleted before editing it."
            : missing
              ? "This round was deleted in another tab. Export your edits before closing."
              : "Could not save locally. Storage may be full; export your round before closing.",
        ),
      );
    };
  });
}
export async function listSessions(): Promise<Session[]> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "trash"], "readonly");
    const sessions = tx.objectStore("sessions").getAll();
    const removed = tx.objectStore("trash").getAllKeys();
    tx.oncomplete = () => {
      const ids = new Set(removed.result);
      db.close();
      resolve(
        (sessions.result as Session[])
          .filter((session) => !ids.has(session.id))
          .sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
      );
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(new Error("Could not load saved rounds."));
    };
  });
}
export async function listTrashedRounds(): Promise<TrashedRound[]> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("trash", "readonly");
    const request = tx.objectStore("trash").getAll();
    tx.oncomplete = () => {
      db.close();
      resolve(
        (request.result as TrashedRound[]).sort((a, b) =>
          b.deletedAt.localeCompare(a.deletedAt),
        ),
      );
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(new Error("Could not load Recently deleted."));
    };
  });
}
export async function moveSessionToTrash(id: string): Promise<TrashedRound> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "trash"], "readwrite");
    const request = tx.objectStore("sessions").get(id);
    let entry: TrashedRound | undefined;
    request.onsuccess = () => {
      const session = request.result as Session | undefined;
      if (!session) {
        tx.abort();
        return;
      }
      entry = {
        id,
        deletedAt: new Date().toISOString(),
        createdAt: session.createdAt,
        durationMs: session.durationMs,
        eventCount: session.events.length,
        hasVideo: !!session.video,
      };
      tx.objectStore("trash").put(entry);
    };
    tx.oncomplete = () => {
      db.close();
      resolve(entry!);
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(
        new Error(
          "Could not remove this round. Make sure it has finished saving, then retry.",
        ),
      );
    };
  });
}
export async function restoreSession(id: string): Promise<Session> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["sessions", "trash"], "readwrite");
    const request = tx.objectStore("sessions").get(id);
    request.onsuccess = () => {
      if (!request.result) {
        tx.abort();
        return;
      }
      tx.objectStore("trash").delete(id);
    };
    tx.oncomplete = () => {
      db.close();
      resolve(request.result as Session);
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(
        new Error(
          "Could not restore this round. Its saved evidence may no longer be available.",
        ),
      );
    };
  });
}
/** Permanent deletion is only allowed from Recently deleted, after UI confirmation. */
export async function deleteSession(id: string) {
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(["sessions", "analyses", "trash"], "readwrite");
    const removed = tx.objectStore("trash").getKey(id);
    let notRemoved = false;
    removed.onsuccess = () => {
      if (removed.result === undefined) {
        notRemoved = true;
        tx.abort();
        return;
      }
      tx.objectStore("sessions").delete(id);
      tx.objectStore("analyses").delete(id);
      tx.objectStore("trash").delete(id);
    };
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = tx.onabort = () => {
      db.close();
      reject(
        new Error(
          notRemoved
            ? "Move this round to Recently deleted before deleting it permanently."
            : "Could not delete this round. Please retry.",
        ),
      );
    };
  });
}

/** Derived video analysis is separate from original evidence and annotations. */
export async function loadRoundAnalysis(
  id: string,
): Promise<RoundAnalysisReport | undefined> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["analyses", "trash"], "readonly");
    const request = tx.objectStore("analyses").get(id);
    const removed = tx.objectStore("trash").getKey(id);
    tx.oncomplete = () => {
      db.close();
      resolve(removed.result === undefined ? request.result : undefined);
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
    const tx = db.transaction(["sessions", "analyses", "trash"], "readwrite");
    const exists = tx.objectStore("sessions").getKey(report.sourceSessionId);
    const removed = tx.objectStore("trash").getKey(report.sourceSessionId);
    let missing = false;
    let inTrash = false;
    removed.onsuccess = () => {
      if (exists.result === undefined || removed.result !== undefined) {
        missing = exists.result === undefined;
        inTrash = removed.result !== undefined;
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
          inTrash
            ? "This round is in Recently deleted. Restore it before saving analysis."
            : missing
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
