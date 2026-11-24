// The page's side of the worker: one request at a time, answers matched by id, progress forwarded.
export class Engine {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.pending = new Map();
    this.next = 1;
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = (event) => this.receive(event.data);
    this.worker.onerror = (event) => this.onEvent({ type: "error", what: "boot", name: "WorkerError", message: String(event.message || "") });
  }

  receive(msg) {
    if (msg.type === "result" || (msg.type === "error" && this.pending.has(msg.id))) {
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.type === "error") reject(msg); else resolve(msg);
      return;
    }
    this.onEvent(msg);
  }

  boot(manifest) { this.worker.postMessage({ type: "boot", manifest }); }

  prefetch(mode) { this.worker.postMessage({ type: "prefetch", mode }); }

  run(op, bytes, mode) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const copy = bytes.slice();
      this.worker.postMessage({ type: "run", id, op, mode, bytes: copy }, [copy.buffer]);
    });
  }
}
