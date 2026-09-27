import { useEffect, useId, useRef, useState } from "react";
import { RotateCcw, Trash2 } from "lucide-react";
import type { Session } from "../lib/types";
import {
  deleteSession,
  formatTime,
  listTrashedRounds,
  moveSessionToTrash,
  restoreSession,
  type TrashedRound,
} from "../lib/storage";

const dateLabel = (createdAt: string) =>
  new Date(createdAt).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });

export function RoundLibrary({
  sessions,
  selectedId,
  unsavedIds,
  onSelect,
  onRemoving,
  onRemoved,
  onRestored,
  onTrashCount,
  onNotice,
}: {
  sessions: Session[];
  selectedId?: string;
  unsavedIds: readonly string[];
  onSelect: (session: Session) => void;
  onRemoving: (id: string) => void;
  onRemoved: (id: string) => void;
  onRestored: (session: Session) => void;
  onTrashCount: (count: number) => void;
  onNotice: (message: string) => void;
}) {
  const [trashed, setTrashed] = useState<TrashedRound[]>([]);
  const [showTrash, setShowTrash] = useState(false);
  const [undoId, setUndoId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const mutating = useRef(false);
  const revision = useRef(0);
  const mounted = useRef(true);
  const library = useRef<HTMLDivElement>(null);
  const trashId = useId();
  useEffect(() => {
    mounted.current = true;
    let current = true;
    const initialRevision = revision.current;
    void listTrashedRounds()
      .then((items) => {
        if (current && revision.current === initialRevision) setTrashed(items);
      })
      .catch((error: Error) => {
        if (current) onNotice(error.message);
      });
    return () => {
      current = false;
      mounted.current = false;
    };
  }, [onNotice]);
  useEffect(() => {
    onTrashCount(trashed.length);
    if (!trashed.length) setShowTrash(false);
  }, [trashed.length, onTrashCount]);
  useEffect(() => {
    const element = library.current;
    if (!element) return;
    const update = () =>
      setOverflows(element.scrollWidth > element.clientWidth + 2);
    const observer = new ResizeObserver(update);
    observer.observe(element);
    update();
    return () => observer.disconnect();
  }, [sessions.length]);
  useEffect(() => {
    const element = library.current;
    const selected = element?.querySelector<HTMLElement>(".is-selected");
    if (!element || !selected) return;
    // Only scroll the library; choosing a round must not scroll the whole page.
    const item = selected.getBoundingClientRect();
    const strip = element.getBoundingClientRect();
    if (item.left < strip.left) element.scrollLeft += item.left - strip.left;
    else if (item.right > strip.right)
      element.scrollLeft += item.right - strip.right;
  }, [selectedId, sessions.length]);

  const mutate = async (operation: () => Promise<void>) => {
    if (mutating.current) return;
    mutating.current = true;
    revision.current++;
    setBusy(true);
    try {
      await operation();
    } catch (error) {
      onNotice(
        error instanceof Error
          ? error.message
          : "Could not update this round. Please retry.",
      );
    } finally {
      mutating.current = false;
      if (mounted.current) setBusy(false);
    }
  };
  const remove = (session: Session) =>
    void mutate(async () => {
      onRemoving(session.id);
      const entry = await moveSessionToTrash(session.id);
      if (mounted.current) {
        setTrashed((old) => [
          entry,
          ...old.filter((item) => item.id !== session.id),
        ]);
        setUndoId(session.id);
      }
      onRemoved(session.id);
    });
  const restore = (id: string) =>
    void mutate(async () => {
      const session = await restoreSession(id);
      if (mounted.current) {
        setTrashed((old) => old.filter((item) => item.id !== id));
        setUndoId((old) => (old === id ? null : old));
      }
      onRestored(session);
    });
  const purge = (entry: TrashedRound) => {
    if (
      !window.confirm(
        `Permanently delete the ${dateLabel(entry.createdAt)} round, its video and analysis? This cannot be undone.`,
      )
    )
      return;
    void mutate(async () => {
      await deleteSession(entry.id);
      if (mounted.current) {
        setTrashed((old) => old.filter((item) => item.id !== entry.id));
        setUndoId((old) => (old === entry.id ? null : old));
      }
    });
  };

  return (
    <div className="round-library-area">
      {sessions.length > 0 && (
        <div className="round-library" aria-label="Saved rounds" ref={library}>
          {sessions.map((session) => (
            <div
              className={`session-card ${session.id === selectedId ? "is-selected" : ""}`}
              key={session.id}
            >
              <button
                className={`session-item ${session.id === selectedId ? "selected" : ""}`}
                aria-pressed={session.id === selectedId}
                onClick={() => onSelect(session)}
              >
                <span>{dateLabel(session.createdAt)}</span>
                <small>
                  {formatTime(session.durationMs)} ·{" "}
                  <strong>{session.events.length} detected</strong>
                  {!session.video && " · Tracking only"}
                  {unsavedIds.includes(session.id) && " · Not saved"}
                </small>
              </button>
              <button
                className="session-remove icon-button"
                disabled={busy || unsavedIds.includes(session.id)}
                aria-label={`Remove round from ${dateLabel(session.createdAt)}`}
                title={
                  unsavedIds.includes(session.id)
                    ? "This round has unsaved changes. Export before closing."
                    : "Move to Recently deleted"
                }
                onClick={() => remove(session)}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      )}
      {sessions.length > 0 && overflows && (
        <p className="library-scroll-hint">Scroll for more rounds</p>
      )}
      {(undoId || trashed.length > 0) && (
        <div className="library-actions">
          {undoId && (
            <div className="round-undo" role="status">
              <span>Round removed.</span>
              <button
                className="text-button"
                disabled={busy}
                onClick={() => restore(undoId)}
              >
                Undo
              </button>
            </div>
          )}
          {trashed.length > 0 && (
            <button
              className="text-button trash-toggle"
              aria-expanded={showTrash}
              aria-controls={trashId}
              onClick={() => setShowTrash((old) => !old)}
            >
              {showTrash
                ? "Close recently deleted"
                : `Recently deleted (${trashed.length})`}
            </button>
          )}
        </div>
      )}
      {showTrash && trashed.length > 0 && (
        <div
          className="recently-deleted"
          id={trashId}
          aria-label="Recently deleted rounds"
        >
          <p>Videos and analysis are kept until you delete them permanently.</p>
          {trashed.map((entry) => (
            <div className="trashed-round" key={entry.id}>
              <div>
                <strong>{dateLabel(entry.createdAt)}</strong>
                <span>
                  {formatTime(entry.durationMs)} · {entry.eventCount} detected
                  {entry.hasVideo ? " · Video saved" : " · Motion only"}
                </span>
              </div>
              <button
                className="text-button"
                disabled={busy}
                onClick={() => restore(entry.id)}
              >
                <RotateCcw size={14} />
                Restore
              </button>
              <button
                className="text-button permanent-delete"
                disabled={busy}
                onClick={() => purge(entry)}
              >
                Delete permanently
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
