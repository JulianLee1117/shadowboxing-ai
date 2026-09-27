const fingerprints = new WeakMap<Blob, Promise<string>>();

/** Blobs are immutable. Cache by object identity without retaining recordings. */
export function fingerprintVideo(video: Blob): Promise<string> {
  let pending = fingerprints.get(video);
  if (!pending) {
    pending = video
      .arrayBuffer()
      .then((bytes) => crypto.subtle.digest("SHA-256", bytes))
      .then((hash) =>
        Array.from(new Uint8Array(hash), (b) =>
          b.toString(16).padStart(2, "0"),
        ).join(""),
      );
    fingerprints.set(video, pending);
    void pending.catch(() => fingerprints.delete(video));
  }
  return pending;
}
