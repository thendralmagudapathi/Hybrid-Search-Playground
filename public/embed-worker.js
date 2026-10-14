// Web Worker: loads Transformers.js lazily and turns text into normalised embedding vectors.
// Messages in:  { type: 'load', model, pooling }
//               { type: 'embed', id, texts, batchSize }
//               { type: 'cancel', id }
// Messages out: { type: 'progress', loaded, total, file }  (download progress, bytes)
//               { type: 'ready', model, device, dims }
//               { type: 'batch', id, start, vectors: Float32Array[] }
//               { type: 'done', id } | { type: 'cancelled', id } | { type: 'error', id?, message }

// Pinned exact version (Transformers.js v3).
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';

let lib = null;
let extractor = null;
let current = { model: null, pooling: 'mean', device: null };
const cancelled = new Set();

// Run one model call at a time; other jobs (e.g. a query) can slip in between document batches.
let chain = Promise.resolve();
const exclusive = (fn) => {
  const next = chain.then(fn, fn);
  chain = next.catch(() => {});
  return next;
};
const yieldToEvents = () => new Promise((r) => setTimeout(r, 0));

async function hasWebGPU() {
  try {
    if (!self.navigator?.gpu) return false;
    return !!(await self.navigator.gpu.requestAdapter());
  } catch { return false; }
}

async function load({ model, pooling }) {
  if (!lib) {
    lib = await import(TRANSFORMERS_URL);
    lib.env.allowLocalModels = false; // never look for models on our own static host
    lib.env.useBrowserCache = true;   // cache model files with the Cache API
  }
  if (extractor && current.model === model) {
    postMessage({ type: 'ready', model, device: current.device, dims: current.dims });
    return;
  }
  if (extractor?.dispose) { try { await extractor.dispose(); } catch { /* ignore */ } }
  extractor = null;

  const files = new Map();
  const progress_callback = (p) => {
    if (p.status === 'progress' && p.total) {
      files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      postMessage({ type: 'progress', loaded, total, file: p.file });
    }
  };

  const devices = (await hasWebGPU()) ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr;
  for (const device of devices) {
    try {
      const pipe = await lib.pipeline('feature-extraction', model, { dtype: 'q8', device, progress_callback });
      // Smoke test: make sure this backend can actually run the model.
      const t = await pipe(['hello world'], { pooling, normalize: true });
      extractor = pipe;
      current = { model, pooling, device, dims: t.dims[1] };
      postMessage({ type: 'ready', model, device, dims: t.dims[1] });
      return;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error('Could not load the model.');
}

async function embed({ id, texts, batchSize = 16 }) {
  if (!extractor) throw new Error('Model is not loaded.');
  for (let start = 0; start < texts.length; start += batchSize) {
    await yieldToEvents(); // lets a 'cancel' message arrive
    if (cancelled.has(id)) { cancelled.delete(id); postMessage({ type: 'cancelled', id }); return; }
    const batch = texts.slice(start, start + batchSize);
    const vectors = await exclusive(async () => {
      const out = await extractor(batch, { pooling: current.pooling, normalize: true });
      const [n, d] = out.dims;
      const data = out.data;
      const vecs = [];
      for (let i = 0; i < n; i++) vecs.push(Float32Array.from(data.subarray(i * d, (i + 1) * d)));
      return vecs;
    });
    postMessage({ type: 'batch', id, start, vectors }, vectors.map((v) => v.buffer));
  }
  postMessage({ type: 'done', id });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') await load(msg);
    else if (msg.type === 'embed') await embed(msg);
    else if (msg.type === 'cancel') cancelled.add(msg.id);
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, stage: msg.type, message: String(err?.message || err) });
  }
};
