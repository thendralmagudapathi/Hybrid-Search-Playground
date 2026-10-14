// Hybrid Search Playground: UI, state, rendering and tooltips.
import * as BM from './bm25.js';
import { rrf, weightedFusion } from './fusion.js';
import * as MT from './metrics.js';
import { DATASETS, getDataset } from './datasets.js';

const MAX_DOCS = 2000;
const MAX_BYTES = 2 * 1024 * 1024;
const FUSION_DEPTH = 100;
const STORAGE_KEY = 'hybrid-search-playground:v1';
const TERM_COLORS = ['#0f766e', '#c2700c', '#6d4ad6', '#be185d', '#2563eb', '#65a30d', '#b45309', '#0891b2', '#7c3aed', '#dc2626'];

export const MODELS = {
  'Xenova/all-MiniLM-L6-v2': {
    label: 'all-MiniLM-L6-v2 (default, ~23 MB)', size: '~23 MB', pooling: 'mean', queryPrefix: '',
    python: 'sentence-transformers/all-MiniLM-L6-v2',
    note: 'Small, fast general-purpose sentence model. 384 numbers per text. Mean pooling. No query prefix needed.',
  },
  'Xenova/bge-small-en-v1.5': {
    label: 'bge-small-en-v1.5 (~34 MB)', size: '~34 MB', pooling: 'cls',
    queryPrefix: 'Represent this sentence for searching relevant passages: ',
    python: 'BAAI/bge-small-en-v1.5',
    note: 'Stronger retrieval model, a bit larger. 384 numbers per text. Uses CLS pooling (as in the original model) and adds the instruction "Represent this sentence for searching relevant passages: " in front of queries (not documents), as its authors recommend for retrieval.',
  },
};
const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const defaultState = () => ({
  datasetId: DATASETS[0].id,
  customText: '',
  split: 'blank',
  separator: '---',
  query: DATASETS[0].queries[0].q,
  k: 5,
  text: { ...BM.DEFAULT_TEXT_OPTIONS },
  bm25: { ...BM.DEFAULT_BM25 },
  modelId: DEFAULT_MODEL,
  fusion: { method: 'rrf', rrfK: 60, alpha: 0.5 },
  labels: {},   // datasetKey -> { normalisedQuery: [docNumbers] }
  selected: null,
});
let state = defaultState();

const derived = { key: null, docs: [], total: 0, tokens: [], index: null, indexMs: 0, textKey: null };
let results = null; // last computed results for the current query

const vec = {
  worker: null,
  status: 'idle',           // idle | loading | ready | error
  loadedModel: null,
  device: null,
  progress: { loaded: 0, total: 0 },
  error: '',
  cache: new Map(),         // cacheKey -> Float32Array
  pending: new Map(),       // request id -> { onBatch, resolve, reject }
  inflight: new Map(),      // cacheKey -> Promise (dedupe)
  docJob: null,             // { id, total, done, docsKey }
  docJobCancelled: false,
};
let reqId = 0;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '–');
const normQ = (q) => q.trim().toLowerCase().replace(/\s+/g, ' ');
const dsKey = () => state.datasetId;
const currentDataset = () => getDataset(state.datasetId);
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const plural = (n, w) => `${n} ${n === 1 ? w : /[^aeiou]y$/.test(w) ? w.slice(0, -1) + 'ies' : w + 's'}`;

// ---------------------------------------------------------------------------
// Documents and index
// ---------------------------------------------------------------------------
function docsText() {
  const ds = currentDataset();
  return ds ? ds.docs.join('\n\n') : state.customText;
}

function splitDocs(text, mode, sep) {
  let parts;
  if (mode === 'line') parts = text.split(/\r?\n/);
  else if (mode === 'custom' && sep) parts = text.split(sep);
  else parts = text.split(/\r?\n\s*\r?\n/);
  return parts.map((p) => p.trim()).filter(Boolean);
}

function ensureIndex() {
  const ds = currentDataset();
  const text = docsText();
  const key = (ds ? ds.id : 'custom') + '\u0000' + state.split + '\u0000' + state.separator + '\u0000' + text;
  const textKey = JSON.stringify(state.text);
  if (key === derived.key && textKey === derived.textKey) return;
  const t0 = performance.now();
  if (key !== derived.key) {
    const all = ds ? ds.docs.slice() : splitDocs(text, state.split, state.separator);
    derived.total = all.length;
    derived.docs = all.slice(0, MAX_DOCS);
    derived.key = key;
    if (state.selected !== null && state.selected >= derived.docs.length) state.selected = null;
  }
  derived.tokens = derived.docs.map((d) => BM.tokenize(d, state.text));
  derived.index = BM.buildIndex(derived.tokens);
  derived.textKey = textKey;
  derived.indexMs = performance.now() - t0;
  ensureDocEmbeddings();
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------
function getLabels(q) {
  const nq = normQ(q);
  const store = state.labels[dsKey()];
  if (store && Object.prototype.hasOwnProperty.call(store, nq)) return new Set(store[nq]);
  const ex = currentDataset()?.queries.find((x) => normQ(x.q) === nq);
  return new Set(ex ? ex.relevant : []);
}
function setLabel(q, docNum, on) {
  if (!q.trim()) return;
  const s = getLabels(q);
  if (on) s.add(docNum); else s.delete(docNum);
  (state.labels[dsKey()] ||= {})[normQ(q)] = [...s].sort((a, b) => a - b);
  save();
}
function labelledQueries() {
  const out = new Map();
  for (const ex of currentDataset()?.queries || []) out.set(normQ(ex.q), ex.q);
  for (const [nq, list] of Object.entries(state.labels[dsKey()] || {})) if (list.length && !out.has(nq)) out.set(nq, nq);
  return [...out.values()].map((q) => ({ q, relevant: getLabels(q) })).filter((x) => x.relevant.size);
}

// ---------------------------------------------------------------------------
// Vector search (worker)
// ---------------------------------------------------------------------------
const model = () => MODELS[state.modelId] || MODELS[DEFAULT_MODEL];
const vectorReady = () => vec.status === 'ready' && vec.loadedModel === state.modelId;
const docKey = (text) => `${state.modelId}\u0000d\u0000${text}`;
const queryKey = (q) => `${state.modelId}\u0000q\u0000${model().queryPrefix}${q}`;
const wordKey = (w) => `${state.modelId}\u0000w\u0000${w}`;

function ensureWorker() {
  if (vec.worker) return vec.worker;
  const w = new Worker(new URL('./embed-worker.js', import.meta.url), { type: 'module' });
  w.onmessage = onWorkerMessage;
  w.onerror = (e) => {
    e.preventDefault?.();
    failModel(e.message || 'The embedding worker could not start.');
  };
  vec.worker = w;
  return w;
}

function loadModel() {
  if (vec.status === 'loading') return;
  if (derived.docs.length > MAX_DOCS) return;
  vec.status = 'loading';
  vec.error = '';
  vec.progress = { loaded: 0, total: 0 };
  cancelDocJob();
  try {
    ensureWorker().postMessage({ type: 'load', model: state.modelId, pooling: model().pooling });
  } catch (err) {
    failModel(String(err?.message || err));
  }
  renderVectorStatus();
}

function failModel(message) {
  vec.status = 'error';
  vec.error = message;
  for (const p of vec.pending.values()) p.reject(new Error(message));
  vec.pending.clear();
  vec.inflight.clear();
  vec.docJob = null;
  if (vec.worker) { vec.worker.terminate(); vec.worker = null; }
  update();
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === 'progress') {
    vec.progress = { loaded: m.loaded, total: m.total };
    renderVectorStatus();
  } else if (m.type === 'ready') {
    vec.status = 'ready';
    vec.loadedModel = m.model;
    vec.device = m.device;
    ensureDocEmbeddings();
    update();
  } else if (m.type === 'batch') {
    vec.pending.get(m.id)?.onBatch?.(m.start, m.vectors);
  } else if (m.type === 'done') {
    vec.pending.get(m.id)?.resolve({ cancelled: false });
    vec.pending.delete(m.id);
  } else if (m.type === 'cancelled') {
    vec.pending.get(m.id)?.resolve({ cancelled: true });
    vec.pending.delete(m.id);
  } else if (m.type === 'error') {
    if (m.stage === 'load') failModel(m.message);
    else { vec.pending.get(m.id)?.reject(new Error(m.message)); vec.pending.delete(m.id); }
  }
}

function embedRequest(texts, onBatch, batchSize = 16) {
  const id = ++reqId;
  const promise = new Promise((resolve, reject) => vec.pending.set(id, { onBatch, resolve, reject }));
  vec.worker.postMessage({ type: 'embed', id, texts, batchSize });
  return { id, promise };
}

/** Embed texts not yet cached (keys computed by keyFn). Resolves when all are cached. */
function embedMissing(texts, keyFn) {
  if (!vectorReady()) return Promise.resolve();
  const todo = [...new Set(texts)].filter((t) => !vec.cache.has(keyFn(t)) && !vec.inflight.has(keyFn(t)));
  const waits = [...new Set(texts)].map((t) => vec.inflight.get(keyFn(t))).filter(Boolean);
  if (todo.length) {
    const modelId = state.modelId;
    const prefixFn = keyFn === queryKey ? (t) => model().queryPrefix + t : (t) => t;
    const { promise } = embedRequest(todo.map(prefixFn), (start, vectors) => {
      vectors.forEach((v, i) => {
        const t = todo[start + i];
        if (modelId === state.modelId) vec.cache.set(keyFn(t), v);
      });
    });
    const p = promise.finally(() => todo.forEach((t) => vec.inflight.delete(keyFn(t))));
    todo.forEach((t) => vec.inflight.set(keyFn(t), p));
    waits.push(p);
  }
  return Promise.all(waits).catch(() => {});
}

function docsEmbedded() {
  return vectorReady() && derived.docs.every((d) => vec.cache.has(docKey(d)));
}

function ensureDocEmbeddings(force = false) {
  if (!vectorReady() || derived.docs.length > MAX_DOCS) return;
  const missing = [...new Set(derived.docs.filter((d) => !vec.cache.has(docKey(d))))];
  if (!missing.length) return;
  if (vec.docJob && vec.docJob.docsKey === derived.key) return; // already running for these docs
  if (vec.docJobCancelled && !force && vec.docJob === null && vec.lastCancelledKey === derived.key) return;
  cancelDocJob();
  vec.docJobCancelled = false;
  const modelId = state.modelId;
  const job = { id: 0, total: missing.length, done: 0, docsKey: derived.key };
  const { id, promise } = embedRequest(missing, (start, vectors) => {
    vectors.forEach((v, i) => { if (modelId === state.modelId) vec.cache.set(`${modelId}\u0000d\u0000${missing[start + i]}`, v); });
    job.done = Math.min(job.total, start + vectors.length);
    renderVectorStatus();
  });
  job.id = id;
  vec.docJob = job;
  promise.then((r) => {
    if (vec.docJob !== job) return;
    vec.docJob = null;
    if (r.cancelled) { vec.docJobCancelled = true; vec.lastCancelledKey = job.docsKey; }
    update();
  }).catch((err) => {
    if (vec.docJob === job) vec.docJob = null;
    vec.error = String(err?.message || err);
    update();
  });
  renderVectorStatus();
}

function cancelDocJob(userCancel = false) {
  if (!vec.docJob) return;
  vec.worker?.postMessage({ type: 'cancel', id: vec.docJob.id });
  if (userCancel) { vec.docJobCancelled = true; vec.lastCancelledKey = vec.docJob.docsKey; }
  vec.pending.get(vec.docJob.id)?.resolve({ cancelled: true });
  vec.pending.delete(vec.docJob.id);
  vec.docJob = null;
}

function vectorRanking(q) {
  if (!q.trim() || !derived.docs.length || !docsEmbedded()) return null;
  const qv = vec.cache.get(queryKey(q));
  if (!qv) return null;
  return derived.docs.map((d, idx) => {
    const dv = vec.cache.get(docKey(d));
    let s = 0;
    for (let i = 0; i < qv.length; i++) s += qv[i] * dv[i];
    return { idx, score: s };
  }).sort((a, b) => b.score - a.score || a.idx - b.idx);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
function fuse(bmList, vecList) {
  if (!vecList) return null;
  if (state.fusion.method === 'weighted') {
    return weightedFusion(
      bmList.map((r) => ({ id: r.idx, score: r.score })),
      vecList.map((r) => ({ id: r.idx, score: r.score })),
      state.fusion.alpha, FUSION_DEPTH,
    );
  }
  return rrf([bmList.map((r) => r.idx), vecList.map((r) => r.idx)], state.fusion.rrfK, FUSION_DEPTH);
}

function runQuery(q) {
  const t0 = performance.now();
  const terms = BM.queryTerms(q, state.text);
  const bm = q.trim() ? BM.rank(derived.index, terms, state.bm25) : [];
  const searchMs = performance.now() - t0;
  const vecList = vectorRanking(q);
  const hyb = fuse(bm, vecList);
  const rankMap = (list, key = 'idx') => new Map((list || []).map((r, i) => [r[key], i + 1]));
  return {
    q, terms, bm, vec: vecList, hyb, searchMs,
    bmRank: rankMap(bm), vecRank: rankMap(vecList), hybRank: rankMap(hyb, 'id'),
    bmScore: new Map(bm.map((r) => [r.idx, r.score])),
    vecScore: new Map((vecList || []).map((r) => [r.idx, r.score])),
  };
}

function update() {
  ensureIndex();
  if (vectorReady() && state.query.trim() && !vec.cache.has(queryKey(state.query))) {
    const q = state.query;
    embedMissing([q], queryKey).then(() => { if (q === state.query) update(); });
  }
  if (vectorReady() && docsEmbedded()) {
    const lq = labelledQueries().map((x) => x.q).filter((q) => !vec.cache.has(queryKey(q)));
    if (lq.length) embedMissing(lq, queryKey).then(() => renderEvaluation());
  }
  results = runQuery(state.query);
  renderAll();
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
function highlightRange(text, spans, termSet, from = 0, to = text.length) {
  let out = '', last = from;
  for (const s of spans) {
    if (s.end <= from || s.start >= to) continue;
    if (s.terms.some((t) => termSet.has(t))) {
      out += esc(text.slice(last, s.start)) + '<mark>' + esc(s.raw) + '</mark>';
      last = s.end;
    }
  }
  return out + esc(text.slice(last, to));
}

function snippet(text, termSet, len = 170) {
  const spans = BM.analyzeWithSpans(text, state.text);
  if (text.length <= len) return highlightRange(text, spans, termSet);
  const first = spans.find((s) => s.terms.some((t) => termSet.has(t)));
  let from = first ? Math.max(0, first.start - 50) : 0;
  if (from > 0) { const sp = text.indexOf(' ', from); from = sp >= 0 && sp < first.start ? sp + 1 : from; }
  let to = Math.min(text.length, from + len);
  if (to < text.length) { const sp = text.lastIndexOf(' ', to); if (sp > from + 40) to = sp; }
  return (from > 0 ? '… ' : '') + highlightRange(text, spans, termSet, from, to) + (to < text.length ? ' …' : '');
}

function rawWordForTerm(q) {
  const map = new Map();
  for (const s of BM.analyzeWithSpans(q, state.text)) for (const t of s.terms) if (!map.has(t)) map.set(t, s.raw);
  return map;
}

function bm25ScoreTip(idx) {
  const ex = BM.explain(derived.index, idx, results.terms, state.bm25);
  const parts = ex.rows.filter((r) => r.tf > 0);
  return `BM25 score = the sum of each matching word's contribution:\n${parts.map((r) => `${r.term} ${fmt(r.contribution)}`).join(' + ')} = ${fmt(ex.total)}\nClick the document for the full breakdown.`;
}

function rankDelta(baseRank, rank, baseName) {
  if (baseRank === undefined) {
    return `<span class="badge new" data-tip="${esc(baseName === 'BM25'
      ? 'BM25 did not find this document: it shares no words with the query after text processing.'
      : 'Vector search did not rank this document in its list.')}">new vs ${baseName}</span>`;
  }
  const d = baseRank - rank;
  if (d === 0) return `<span class="badge" data-tip="Same position as in the ${baseName} column (#${baseRank}).">= ${baseName}</span>`;
  const up = d > 0;
  return `<span class="badge ${up ? 'up' : 'down'}" data-tip="${up ? 'Higher' : 'Lower'} than in the ${baseName} column: #${baseRank} there, #${rank} here (${Math.abs(d)} place${Math.abs(d) === 1 ? '' : 's'} ${up ? 'up' : 'down'}).">${up ? '▲' : '▼'}${Math.abs(d)} vs ${baseName}</span>`;
}

function onlyBadge(idx) {
  if (!results.vec) return '';
  const k = state.k;
  const inBm = (results.bmRank.get(idx) ?? Infinity) <= k;
  const inVec = (results.vecRank.get(idx) ?? Infinity) <= k;
  if (inBm && !inVec) return `<span class="badge only-bm25" data-tip="In BM25's top ${k} but not in vector search's top ${k}.\nUsually means an exact word or code matched that the embedding model doesn't weigh much.">Only BM25 found this</span>`;
  if (inVec && !inBm) return `<span class="badge only-vec" data-tip="In vector search's top ${k} but not in BM25's top ${k}.\nUsually means the document says the same thing in different words.">Only vector search found this</span>`;
  return '';
}

function resultCard(method, rank, idx, score, maxScore, extraMeta, scoreTip) {
  const termSet = new Set(results.terms);
  const rel = getLabels(state.query).has(idx + 1);
  const width = maxScore > 0 ? Math.max(2, Math.min(100, (Math.max(0, score) / maxScore) * 100)) : 0;
  return `<li class="result${state.selected === idx ? ' selected' : ''}${rel ? ' relevant' : ''}" data-idx="${idx}" tabindex="0">
    <div class="r-top">
      <span class="r-rank" data-tip="Position in the ${method} ranking.">#${rank}</span>
      <span class="r-doc" data-tip="Document number in the dataset (its position in the list).">Doc ${idx + 1}</span>
      <span class="r-score" data-tip="${esc(scoreTip)}">${fmt(score, method === 'Hybrid' && state.fusion.method === 'rrf' ? 4 : 3)}</span>
    </div>
    <div class="r-bar" data-tip="Bar length = this score compared with the top result in this column."><i style="width:${width}%"></i></div>
    <p class="r-snip">${snippet(derived.docs[idx], termSet)}</p>
    <div class="r-meta">${extraMeta}
      <label class="rel-check" data-tip="Tick if this document is a good answer for the query.\nThe evaluation table uses these ticks to score each method."><input type="checkbox" data-rel="${idx}"${rel ? ' checked' : ''}> relevant</label>
    </div>
  </li>`;
}

// ---------------------------------------------------------------------------
// Render: inputs
// ---------------------------------------------------------------------------
function renderInputs() {
  const ds = currentDataset();
  $('dataset').value = state.datasetId;
  $('dataset-lesson').textContent = ds ? ds.lesson : 'Your own documents. Paste text below or open a file. Nothing is uploaded.';
  const ta = $('docs');
  const text = docsText();
  if (ta.value !== text && document.activeElement !== ta) ta.value = text;
  $('split').value = ds ? 'blank' : state.split;
  $('separator').hidden = (ds ? 'blank' : state.split) !== 'custom';
  if (document.activeElement !== $('separator')) $('separator').value = state.separator;
  const capped = derived.total > MAX_DOCS;
  $('doc-count').textContent = capped ? `${MAX_DOCS.toLocaleString()} of ${derived.total.toLocaleString()} docs (limit)` : plural(derived.docs.length, 'doc');
  $('doc-count').className = 'small ' + (capped ? 'msg warn' : 'muted');
  if (document.activeElement !== $('query')) $('query').value = state.query;
  $('topk').value = String(state.k);

  // processed query tokens
  const spans = BM.analyzeWithSpans(state.query, state.text);
  $('query-tokens').innerHTML = spans.map((s) => (s.terms.length
    ? s.terms.map((t) => `<span class="tok">${esc(t)}</span>`).join('')
    : `<span class="tok drop">${esc(s.raw)}</span>`)).join('');

  // example queries
  const list = ds ? ds.queries.map((x) => x.q) : labelledQueries().map((x) => x.q);
  $('examples-wrap').hidden = !list.length;
  $('examples').innerHTML = list.map((q) => {
    const ex = ds?.queries.find((x) => x.q === q);
    const n = getLabels(q).size;
    const tip = (ex?.note ? ex.note + '\n' : '') + `${plural(n, 'document')} marked relevant.`;
    return `<li><button type="button" data-q="${esc(q)}" class="${normQ(q) === normQ(state.query) ? 'active' : ''}" data-tip="${esc(tip)}">${esc(q)}<span class="lbl">${n} rel.</span></button></li>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Render: results
// ---------------------------------------------------------------------------
function renderResults() {
  const k = state.k;
  const r = results;
  $('timing').textContent = `BM25: index ${fmt(derived.indexMs, 1)} ms · search ${fmt(r.searchMs, 2)} ms · ${plural(derived.docs.length, 'doc')}`;

  // BM25
  if (!r.q.trim()) $('list-bm25').innerHTML = '<li class="empty-col">Type a query to see results.</li>';
  else if (!r.bm.length) $('list-bm25').innerHTML = `<li class="empty-col">No document contains any of the query words${r.terms.length ? ` (${r.terms.map(esc).join(', ')})` : ''}. BM25 only matches exact words after text processing.</li>`;
  else {
    const max = r.bm[0].score;
    $('list-bm25').innerHTML = r.bm.slice(0, k).map((x, i) => resultCard('BM25', i + 1, x.idx, x.score, max,
      (r.vec ? rankDelta(r.vecRank.get(x.idx), i + 1, 'Vector') : '') + onlyBadge(x.idx), bm25ScoreTip(x.idx))).join('');
  }

  // Vector
  $('vector-sub').innerHTML = vectorReady()
    ? `<span class="device-pill" data-tip="${vec.device === 'webgpu' ? 'Running on your graphics card with WebGPU (fast).' : 'Running on your processor with WebAssembly (WASM). WebGPU is not available in this browser, or it could not run this model.'}">${vec.device === 'webgpu' ? 'WebGPU' : 'WASM'}</span>`
    : 'embeddings';
  if (!r.vec) $('list-vector').innerHTML = (vectorReady() && docsEmbedded() && !r.q.trim()) ? '<li class="empty-col">Type a query to see results.</li>' : '';
  else {
    const max = r.vec[0].score;
    $('list-vector').innerHTML = r.vec.slice(0, k).map((x, i) => resultCard('Vector', i + 1, x.idx, x.score, max,
      rankDelta(r.bmRank.get(x.idx), i + 1, 'BM25') + onlyBadge(x.idx),
      `Cosine similarity between the query and this document = ${fmt(x.score, 3)}.\nThe vectors have length 1, so this equals their dot product.\nRange −1 to 1; higher = closer in meaning.`)).join('');
  }
  renderVectorStatus();

  // Hybrid
  $('hybrid-sub').textContent = state.fusion.method === 'rrf' ? `RRF, k = ${state.fusion.rrfK}` : `weighted, α = ${fmt(state.fusion.alpha)}`;
  if (!r.hyb) {
    $('list-hybrid').innerHTML = `<li class="empty-col">${r.q.trim() ? 'Hybrid search combines BM25 with vector results, so it needs the embedding model. Load it in the Vector column.' : 'Type a query to see results.'}</li>`;
  } else {
    const max = r.hyb[0]?.score || 0;
    $('list-hybrid').innerHTML = r.hyb.slice(0, k).map((e, i) => {
      const [br, vr] = e.ranks;
      const origin = br && vr ? `BM25 #${br} + Vector #${vr}` : br ? `BM25 #${br} only` : `Vector #${vr} only`;
      const originTip = br && vr ? 'Where this result came from: its position in the BM25 list and in the vector list.'
        : br ? 'Only in the BM25 list (vector search ranked it below its top 100).' : 'Only in the vector list: BM25 gave it 0 (no shared words).';
      let tip;
      if (state.fusion.method === 'rrf') {
        const kk = state.fusion.rrfK;
        const terms = [br && `1/(${kk} + ${br})`, vr && `1/(${kk} + ${vr})`].filter(Boolean);
        const vals = [br && 1 / (kk + br), vr && 1 / (kk + vr)].filter(Boolean);
        tip = `RRF score = ${terms.join(' + ')}${vals.length > 1 ? ` = ${vals.map((v) => fmt(v, 4)).join(' + ')}` : ''} = ${fmt(e.score, 4)}\nEach list adds 1 / (k + rank). A list that didn't find the document adds nothing.`;
      } else {
        const a = state.fusion.alpha;
        tip = `Weighted score = α × vector + (1 − α) × BM25\n= ${fmt(a)} × ${fmt(e.parts.vector, 3)} + ${fmt(1 - a)} × ${fmt(e.parts.bm25, 3)} = ${fmt(e.score, 3)}\n(both scores first rescaled to 0–1 within their own list)`;
      }
      const meta = `<span class="badge origin" data-tip="${esc(originTip)}">${origin}</span>`
        + rankDelta(r.bmRank.get(e.id), i + 1, 'BM25') + rankDelta(r.vecRank.get(e.id), i + 1, 'Vector') + onlyBadge(e.id);
      return resultCard('Hybrid', i + 1, e.id, e.score, max, meta, tip);
    }).join('');
  }
}

function renderVectorStatus() {
  const box = $('vector-status');
  const m = model();
  let html = '';
  const loadBtn = (label) => `<button class="btn primary" data-action="load-model" data-tip="Downloads the ${esc(state.modelId.split('/')[1])} model (${m.size}) from Hugging Face via the jsDelivr-hosted Transformers.js library.\nIt runs inside a background worker, so the page never freezes. Your browser caches it, so next time it loads instantly.\nYour documents are not sent anywhere.">${label}</button>`;
  if (vec.status === 'loading') {
    const { loaded, total } = vec.progress;
    const pct = total ? (loaded / total) * 100 : 0;
    html = `<div class="vs-box"><p>${total ? `Downloading model… ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB` : 'Loading Transformers.js…'}</p>
      <div class="progress" data-tip="Download progress of the model files. Cached by your browser afterwards."><i style="width:${pct}%"></i></div></div>`;
  } else if (vec.status === 'error') {
    const offline = /fetch|network|import|load|cdn|resolve|failed/i.test(vec.error);
    html = `<div class="vs-box"><p><strong>The embedding model could not be loaded.</strong> ${offline ? 'You may be offline, or a firewall, VPN or content blocker may be blocking cdn.jsdelivr.net or huggingface.co.' : ''} BM25 keeps working.</p>
      <p class="muted small" data-tip="The technical error message, in case it helps.">Details: ${esc(vec.error.slice(0, 220))}</p>${loadBtn('Try again')}</div>`;
  } else if (!vectorReady()) {
    const other = vec.status === 'ready' && vec.loadedModel !== state.modelId;
    html = `<div class="vs-box"><p>${other ? `You picked a different model (${esc(state.modelId.split('/')[1])}).` : 'Vector search turns text into embeddings with a small neural network, right here in your browser.'}</p>
      ${loadBtn(`Load embedding model (${m.size}, one time, cached by your browser)`)}</div>`;
  } else if (vec.docJob) {
    const { done, total } = vec.docJob;
    html = `<div class="vs-box"><p>Embedding documents… ${done} / ${total}</p>
      <div class="progress" data-tip="Documents are embedded in batches of 16 in a background worker. Results appear when all are done."><i style="width:${total ? (done / total) * 100 : 0}%"></i></div>
      <button class="btn small" data-action="cancel-embed" data-tip="Stop embedding. Documents already embedded are kept, so you can resume later.">Cancel</button></div>`;
  } else if (!docsEmbedded()) {
    const done = derived.docs.filter((d) => vec.cache.has(docKey(d))).length;
    html = `<div class="vs-box"><p>Embedding was stopped (${done} of ${derived.docs.length} documents done).</p>
      <button class="btn small" data-action="resume-embed" data-tip="Continue embedding the remaining documents.">Resume</button></div>`;
  } else if (state.query.trim() && !vec.cache.has(queryKey(state.query))) {
    html = '<div class="vs-box"><p class="muted">Embedding the query…</p></div>';
  }
  box.innerHTML = html;

  // mirror in settings
  const mc = $('model-controls');
  if (vectorReady()) mc.innerHTML = `<p class="small">✅ Loaded, running on <strong>${vec.device === 'webgpu' ? 'WebGPU' : 'WASM'}</strong>${vec.docJob ? ` · embedding ${vec.docJob.done}/${vec.docJob.total}` : ''}.</p>`;
  else if (vec.status === 'loading') mc.innerHTML = '<p class="small">Loading…</p>';
  else mc.innerHTML = loadBtn(vec.status === 'error' ? 'Try again' : `Load model (${m.size})`);
}

// ---------------------------------------------------------------------------
// Render: detail panel
// ---------------------------------------------------------------------------
function rarity(df, N) {
  const share = N ? df / N : 0;
  if (df === 0) return 'does not appear in any document';
  if (share <= 0.1) return `is rare (appears in ${df} of ${N} documents), so it counts a lot`;
  if (share <= 0.35) return `is fairly uncommon (appears in ${df} of ${N} documents), so it counts a fair amount`;
  return `is common (appears in ${df} of ${N} documents), so it counts only a little`;
}

function renderDetail() {
  const idx = state.selected;
  const has = idx !== null && idx < derived.docs.length;
  $('detail-empty').hidden = has;
  $('detail').hidden = !has;
  if (!has) return;
  const r = results;
  const text = derived.docs[idx];
  const termSet = new Set(r.terms);
  const spans = BM.analyzeWithSpans(text, state.text);
  const ex = BM.explain(derived.index, idx, r.terms, state.bm25);
  const { N, avgdl, docLen, k1, b } = ex;
  const rel = getLabels(state.query).has(idx + 1);
  const bmR = r.bmRank.get(idx), vR = r.vecRank.get(idx), hR = r.hybRank.get(idx);

  let html = `<div class="detail-head"><h3>Document ${idx + 1}</h3>
    <label class="rel-check" data-tip="Tick if this document is a good answer for the current query. Used by the evaluation table."><input type="checkbox" data-rel="${idx}"${rel ? ' checked' : ''}> relevant</label></div>
    <div class="detail-ranks">
      <span class="badge only-bm25" data-tip="Position and score in the BM25 ranking.">BM25 ${bmR ? `#${bmR} · ${fmt(ex.total, 3)}` : 'no match (0)'}</span>
      <span class="badge only-vec" data-tip="Position and cosine similarity in the vector ranking.">${r.vec ? `Vector #${vR} · ${fmt(r.vecScore.get(idx), 3)}` : 'Vector: not loaded'}</span>
      ${r.hyb ? `<span class="badge origin" data-tip="Position in the hybrid ranking.">Hybrid #${hR ?? '–'}</span>` : ''}
    </div>
    <div class="detail-text" data-tip="Full text. Words that match the query (after text processing) are highlighted.">${highlightRange(text, spans, termSet)}</div>`;

  // processed tokens
  const toks = derived.tokens[idx];
  const shown = toks.slice(0, 300);
  html += `<h4>Processed tokens <span class="help" data-tip="This document after lowercasing, punctuation removal, stopword removal and stemming (as set in Settings).\nThis is exactly what BM25 compares with the query. Highlighted tokens match the query.">?</span> <span class="muted small">${docLen} tokens · average ${fmt(avgdl, 1)}</span></h4>
    <div class="tokens">${shown.map((t) => `<span class="tok${termSet.has(t) ? ' hit' : ''}">${esc(t)}</span>`).join('')}${toks.length > shown.length ? `<span class="muted small">… +${toks.length - shown.length} more</span>` : ''}</div>`;

  // BM25 breakdown
  if (ex.rows.length) {
    const rows = ex.rows.map((row, i) => ({ ...row, color: TERM_COLORS[i % TERM_COLORS.length] }));
    const lf = rows[0].lengthFactor;
    const lfTip = `Length factor = 1 − b + b × |d| / avgdl\n= 1 − ${fmt(b)} + ${fmt(b)} × ${docLen} / ${fmt(avgdl, 1)} = ${fmt(lf, 3)}\nAbove 1 = longer than average (scores shrink). Below 1 = shorter (scores grow).`;
    html += `<h4>BM25 score breakdown <span class="help" data-tip="One row per query word (after processing). The document's BM25 score is the sum of the last column.">?</span></h4>
    <table class="breakdown"><thead><tr>
      <th data-tip="The query word after text processing.">Term</th>
      <th data-tip="Term frequency: how many times the term appears in this document.">tf</th>
      <th data-tip="Document frequency: how many documents contain the term.">df</th>
      <th data-tip="Inverse document frequency: rare terms get a higher value.\nIDF = ln(1 + (N − df + 0.5) / (df + 0.5))">IDF</th>
      <th data-tip="${esc(lfTip)}">Length</th>
      <th data-tip="This term's share of the score:\nIDF × tf × (k1 + 1) / (tf + k1 × length factor)">Contribution</th>
    </tr></thead><tbody>
    ${rows.map((row) => {
      const idfTip = `IDF = ln(1 + (N − df + 0.5) / (df + 0.5))\n= ln(1 + (${N} − ${row.df} + 0.5) / (${row.df} + 0.5)) = ${fmt(row.idf, 3)}`;
      const cTip = row.tf ? `${fmt(row.idf, 3)} × ${row.tf} × (${fmt(k1)} + 1) / (${row.tf} + ${fmt(k1)} × ${fmt(row.lengthFactor, 3)}) = ${fmt(row.contribution, 3)}` : 'tf = 0, so this term adds nothing.';
      return `<tr class="${row.tf ? '' : 'zero'}"><td class="term" data-tip="A query word after text processing. The colour matches its slice of the bar below."><span class="swatch" style="background:${row.color}"></span>${esc(row.term)}</td>
        <td data-tip="&quot;${esc(row.term)}&quot; appears ${row.tf} time${row.tf === 1 ? '' : 's'} in this document.">${row.tf}</td>
        <td data-tip="${row.df} of ${N} documents contain &quot;${esc(row.term)}&quot;.">${row.df}</td>
        <td data-tip="${esc(idfTip)}">${fmt(row.idf)}</td>
        <td data-tip="${esc(lfTip)}">${fmt(row.lengthFactor)}</td>
        <td data-tip="${esc(cTip)}">${fmt(row.contribution, 3)}</td></tr>`;
    }).join('')}
    </tbody><tfoot><tr><td colspan="5" data-tip="BM25 score = sum of all contributions.">Total</td><td data-tip="${esc(`${rows.filter((x) => x.tf).map((x) => fmt(x.contribution, 3)).join(' + ') || '0'} = ${fmt(ex.total, 3)}`)}">${fmt(ex.total, 3)}</td></tr></tfoot></table>`;
    if (ex.total > 0) {
      html += `<div class="stack" data-tip="Each colour is one query term; its width is that term's share of the total score.">${rows.filter((x) => x.contribution > 0).map((x) => `<i style="width:${(x.contribution / ex.total) * 100}%;background:${x.color}" data-tip="${esc(x.term)}: ${fmt(x.contribution, 3)} of ${fmt(ex.total, 3)} (${fmt((x.contribution / ex.total) * 100, 0)}%)"></i>`).join('')}</div>`;
    }
    html += `<ul class="sentences">${rows.map((row) => {
      if (!row.tf) return `<li><strong>${esc(row.term)}</strong> is not in this document, so it adds nothing.</li>`;
      const rep = row.tf === 1 ? 'It appears once here.' : `It appears ${row.tf} times here, but repeats add less and less (k1 = ${fmt(k1)}).`;
      const share = ex.total ? ` It gives ${fmt((row.contribution / ex.total) * 100, 0)}% of this document's score.` : '';
      return `<li><strong>${esc(row.term)}</strong> ${rarity(row.df, N)}. ${rep}${share}</li>`;
    }).join('')}</ul>`;
  }

  html += renderWhy(idx, ex);
  $('detail').innerHTML = html;
  maybeFindSynonyms(idx, ex);
}

// "Why didn't this match?"
function renderWhy(idx, ex) {
  const r = results;
  if (!r.q.trim()) return '';
  const items = [];
  const raw = rawWordForTerm(r.q);
  const bmR = r.bmRank.get(idx);
  const missing = ex.rows.filter((x) => x.tf === 0);
  const docSpans = BM.analyzeWithSpans(derived.docs[idx], state.text);
  const docWords = docSpans.map((s) => s.raw);
  const needed = !bmR || bmR > state.k || missing.length > 0;

  if (!bmR) items.push(`BM25 gives this document <strong>0</strong>: after text processing it shares no words with the query.`);
  else if (bmR > state.k) items.push(`BM25 ranks it <strong>#${bmR}</strong> of ${r.bm.length} matching documents (score ${fmt(ex.total, 2)}), outside the top ${state.k}.`);

  if (missing.length) {
    const words = missing.map((x) => `“${esc(raw.get(x.term) || x.term)}”`);
    items.push(`Missing from this document: ${words.join(', ')}${missing.some((x) => raw.get(x.term) && raw.get(x.term).toLowerCase() !== x.term) ? ` (processed: ${missing.map((x) => esc(x.term)).join(', ')})` : ''}.`);
  }

  // stopwords dropped from the query
  if (state.text.stopwords) {
    const dropped = BM.analyzeWithSpans(r.q, state.text).filter((s) => !s.terms.length).map((s) => s.raw);
    if (dropped.length && missing.length) items.push(`${dropped.map((w) => `“${esc(w)}”`).join(', ')} ${dropped.length === 1 ? 'was' : 'were'} ignored because ${dropped.length === 1 ? 'it is a stopword' : 'they are stopwords'} (very common words).`);
  }

  // effects of text-processing toggles on the missing words
  for (const m of missing) {
    const qw = raw.get(m.term) || m.term;
    const ql = qw.toLowerCase();
    if (!state.text.stem && /^[a-z]+$/i.test(qw)) {
      const st = BM.porterStem(ql);
      const hit = docWords.find((w) => /^[a-z]+$/i.test(w) && w.toLowerCase() !== ql && BM.porterStem(w.toLowerCase()) === st);
      if (hit) items.push(`“${esc(qw)}” was not matched to “${esc(hit)}” because stemming is off. Turn on Stemming and both become “${esc(st)}”.`);
    }
    if (!state.text.lowercase) {
      const hit = docWords.find((w) => w !== qw && w.toLowerCase() === ql);
      if (hit) items.push(`“${esc(qw)}” did not match “${esc(hit)}” because Lowercase is off, so capital letters make them different words.`);
    }
    if (!state.text.removePunct) {
      const strip = (w) => w.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
      const hit = docWords.find((w) => w !== qw && strip(w) === strip(qw));
      if (hit) items.push(`“${esc(qw)}” did not match “${esc(hit)}” because punctuation is kept (Remove punctuation is off).`);
    }
  }

  // positive stemming note
  if (state.text.stem) {
    const qSpans = BM.analyzeWithSpans(r.q, state.text);
    for (const qs of qSpans) {
      for (const t of qs.terms) {
        if (ex.rows.find((x) => x.term === t && x.tf > 0)) {
          const other = docSpans.find((s) => s.terms.includes(t) && s.raw.toLowerCase() !== qs.raw.toLowerCase());
          if (other && /^[a-z]+$/i.test(qs.raw)) { items.push(`✓ “${esc(qs.raw)}” matched “${esc(other.raw)}” thanks to stemming (both become “${esc(t)}”).`); break; }
        }
      }
    }
  }

  // length penalty
  const ratio = ex.avgdl ? ex.docLen / ex.avgdl : 1;
  if (ratio >= 1.5 && state.bm25.b > 0) {
    items.push(`This document is <strong>${fmt(ratio, 1)}× longer</strong> than average (${ex.docLen} vs ${fmt(ex.avgdl, 1)} tokens), so with b = ${fmt(state.bm25.b)} its scores are reduced (length factor ${fmt(ex.rows[0]?.lengthFactor ?? 0, 2)} instead of 1). Lower b to reduce this penalty.`);
  } else if (ratio <= 0.6 && state.bm25.b > 0 && ex.total > 0) {
    items.push(`This document is shorter than average (${ex.docLen} vs ${fmt(ex.avgdl, 1)} tokens), which gives its matches a small boost.`);
  }

  // vocabulary mismatch (vector)
  if (missing.length) {
    if (r.vec) {
      const vR = r.vecRank.get(idx);
      const sim = r.vecScore.get(idx);
      if (vR <= Math.max(state.k, 5)) {
        items.push(`<span id="why-synonyms">Vector search ranks it <strong>#${vR}</strong> (similarity ${fmt(sim, 2)}), so it probably says the same thing in other words. Looking for the matching words…</span>`);
      }
    } else if (needed) {
      items.push('Load the embedding model to check whether this document uses different words for the same idea (vocabulary mismatch).');
    }
  }

  if (!needed && items.length === 0) {
    return `<div class="why ok"><h4>Why did this match?</h4><ul><li>Every query word was found, and BM25 ranks it #${bmR}.</li></ul></div>`;
  }
  if (!items.length) return '';
  return `<div class="why${needed ? '' : ' ok'}"><h4>${needed ? 'Why didn\'t this match?' : 'Why did this match?'} <span class="help" data-tip="Plain-English reasons for this document's BM25 result: missing words, text-processing effects, length, and words with the same meaning (from the embedding model).">?</span></h4><ul>${items.map((x) => `<li>${x}</li>`).join('')}</ul></div>`;
}

// Find which document words mean the same as the missing query words, using the embedding model.
async function maybeFindSynonyms(idx, ex) {
  const el = document.getElementById('why-synonyms');
  if (!el || !vectorReady()) return;
  const r = results;
  const raw = rawWordForTerm(r.q);
  const missingWords = [...new Set(ex.rows.filter((x) => x.tf === 0).map((x) => (raw.get(x.term) || x.term).toLowerCase()))];
  const docWords = [...new Set(BM.analyzeWithSpans(derived.docs[idx], { ...BM.DEFAULT_TEXT_OPTIONS, stem: false })
    .filter((s) => s.terms.length && /^\p{L}[\p{L}-]{2,}$/u.test(s.raw)).map((s) => s.raw.toLowerCase()))].slice(0, 150);
  if (!missingWords.length || !docWords.length) return;
  const q = state.query;
  await embedMissing([...missingWords, ...docWords], wordKey);
  if (q !== state.query || state.selected !== idx || !document.getElementById('why-synonyms')) return;
  const pairs = [];
  for (const mw of missingWords) {
    const a = vec.cache.get(wordKey(mw));
    if (!a) continue;
    let best = null;
    for (const dw of docWords) {
      if (dw === mw || BM.porterStem(dw) === BM.porterStem(mw)) continue; // same root: explained by stemming
      const v = vec.cache.get(wordKey(dw));
      if (!v) continue;
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * v[i];
      if (!best || s > best.s) best = { dw, s };
    }
    if (best && best.s >= 0.45) pairs.push({ mw, ...best });
  }
  const vR = r.vecRank.get(idx);
  const lead = `Vector search ranks it <strong>#${vR}</strong> (similarity ${fmt(r.vecScore.get(idx), 2)}). `;
  document.getElementById('why-synonyms').innerHTML = pairs.length
    ? lead + pairs.map((p) => `Your query says “${esc(p.mw)}”, this document says “${esc(p.dw)}” (word similarity ${fmt(p.s, 2)}).`).join(' ') + ' BM25 only matches exact words; vector search can catch this.'
    : lead + 'It expresses the idea with different words, which BM25 cannot see. BM25 only matches exact words; vector search can catch this.';
}

// ---------------------------------------------------------------------------
// Render: evaluation
// ---------------------------------------------------------------------------
const METRIC_INFO = {
  precision: { name: 'Precision@k', tip: (k) => `Of the top ${k} results, what share is relevant?\nFormula: (relevant in top ${k}) ÷ ${k}\nExample: 2 of the top 5 are relevant → 2 ÷ 5 = 0.40` },
  recall: { name: 'Recall@k', tip: (k) => `Of all relevant documents, what share made it into the top ${k}?\nFormula: (relevant in top ${k}) ÷ (all relevant)\nExample: 3 relevant documents, 2 of them in the top 5 → 2 ÷ 3 = 0.67\n0 when no document is marked relevant.` },
  mrr: { name: 'MRR', tip: (k) => `How high is the first relevant result? 1 ÷ its position (within the top ${k}), or 0 if there is none.\nExample: first relevant result at #2 → 1 ÷ 2 = 0.50\nAveraged over queries this is the Mean Reciprocal Rank.` },
  ndcg: { name: 'nDCG@k', tip: (k) => `Rewards relevant results more when they are near the top.\nA relevant result at position i is worth 1 ÷ log2(i + 1); the sum is divided by the best possible sum.\nExample: relevant docs at #1 and #3, 2 relevant in total: (1 + 0.5) ÷ (1 + 0.63) = 0.92\n1 = perfect order.` },
};

function metricValueTip(key, ranked, rel, k) {
  const top = ranked.slice(0, k);
  const hits = top.filter((id) => rel.has(id)).length;
  if (key === 'precision') return `Precision@${k} = ${hits} ÷ ${k} = ${fmt(hits / k)}`;
  if (key === 'recall') return rel.size ? `Recall@${k} = ${hits} ÷ ${rel.size} = ${fmt(hits / rel.size)}` : 'No relevant documents marked, so recall is 0.';
  if (key === 'mrr') { const p = top.findIndex((id) => rel.has(id)); return p >= 0 ? `First relevant result at #${p + 1} → 1 ÷ ${p + 1} = ${fmt(1 / (p + 1))}` : `No relevant result in the top ${k} → 0`; }
  const dcg = MT.dcgAtK(ranked, rel, k);
  const ideal = Math.min(rel.size, k);
  let idcg = 0; for (let i = 0; i < ideal; i++) idcg += 1 / Math.log2(i + 2);
  const parts = top.map((id, i) => (rel.has(id) ? `1/log2(${i + 2})` : null)).filter(Boolean);
  return idcg ? `DCG = ${parts.join(' + ') || '0'} = ${fmt(dcg, 3)}\nIdeal DCG (${ideal} relevant at the top) = ${fmt(idcg, 3)}\nnDCG = ${fmt(dcg, 3)} ÷ ${fmt(idcg, 3)} = ${fmt(dcg / idcg)}` : 'No relevant documents marked, so nDCG is 0.';
}

function metricsTable(rowsByMethod, k, tipFn) {
  const methods = [['bm25', 'BM25'], ['vector', 'Vector'], ['hybrid', 'Hybrid']];
  const keys = ['precision', 'recall', 'mrr', 'ndcg'];
  const best = {};
  for (const key of keys) {
    const vals = methods.map(([m]) => rowsByMethod[m]?.[key]).filter((v) => v !== undefined);
    best[key] = vals.length ? Math.max(...vals) : null;
  }
  return `<table class="metrics"><thead><tr><th data-tip="Search method.">Method</th>${keys.map((key) => `<th data-tip="${esc(METRIC_INFO[key].tip(k))}">${METRIC_INFO[key].name.replace('@k', `@${k}`)}</th>`).join('')}</tr></thead><tbody>
    ${methods.map(([m, label]) => `<tr><td class="m-${m}" data-tip="${esc({ bm25: 'BM25 keyword search.', vector: 'Vector (embedding) search.', hybrid: 'Hybrid: BM25 and vector lists fused together.' }[m])} Green cells are the best for that metric.">${label}</td>${keys.map((key) => {
      const v = rowsByMethod[m]?.[key];
      if (v === undefined) return '<td class="muted" data-tip="Needs the embedding model. Load it in the Vector column.">–</td>';
      const isBest = best[key] > 0 && Math.abs(v - best[key]) < 1e-9;
      return `<td class="${isBest ? 'best' : ''}" data-tip="${esc((tipFn ? tipFn(m, key) : '') + (isBest ? '\nBest (or tied best) for this metric.' : ''))}">${fmt(v)}</td>`;
    }).join('')}</tr>`).join('')}
  </tbody></table>`;
}

function rankingsFor(q) {
  const terms = BM.queryTerms(q, state.text);
  const bm = BM.rank(derived.index, terms, state.bm25);
  const v = vectorRanking(q);
  const h = fuse(bm, v);
  return { bm: bm.map((x) => x.idx + 1), vec: v?.map((x) => x.idx + 1), hyb: h?.map((x) => x.id + 1) };
}

function renderEvaluation() {
  const k = state.k;
  $('eval-k').textContent = state.query.trim() ? `“${state.query}” at k = ${k}` : '';
  const rel = getLabels(state.query);
  const ins = [];
  if (!state.query.trim()) {
    $('eval-table').innerHTML = '<p class="muted">Type a query first.</p>';
  } else if (!rel.size) {
    $('eval-table').innerHTML = '<p class="muted">No documents are marked relevant for this query yet. Tick “relevant” on the results that answer it.</p>';
  } else {
    const rk = { bm: results.bm.map((x) => x.idx + 1), vec: results.vec?.map((x) => x.idx + 1), hyb: results.hyb?.map((x) => x.id + 1) };
    const lists = { bm25: rk.bm, vector: rk.vec, hybrid: rk.hyb };
    const rows = {};
    for (const [m, list] of Object.entries(lists)) if (list) rows[m] = MT.evaluate(list, rel, k);
    $('eval-table').innerHTML = metricsTable(rows, k, (m, key) => metricValueTip(key, lists[m], rel, k));
    ins.push(...insights(rk, rel, k));
  }

  // all labelled queries
  const lq = labelledQueries();
  if (!lq.length) {
    $('eval-all').innerHTML = '<p class="muted">No labelled queries yet.</p>';
  } else {
    const sums = { bm25: [], vector: [], hybrid: [] };
    let vecMissing = false;
    for (const { q, relevant } of lq) {
      const rk = rankingsFor(q);
      sums.bm25.push(MT.evaluate(rk.bm, relevant, k));
      if (rk.vec) { sums.vector.push(MT.evaluate(rk.vec, relevant, k)); sums.hybrid.push(MT.evaluate(rk.hyb, relevant, k)); } else vecMissing = true;
    }
    const avg = (arr) => (arr.length ? Object.fromEntries(['precision', 'recall', 'mrr', 'ndcg'].map((key) => [key, arr.reduce((s, x) => s + x[key], 0) / arr.length])) : undefined);
    const rows = { bm25: avg(sums.bm25) };
    if (!vecMissing) { rows.vector = avg(sums.vector); rows.hybrid = avg(sums.hybrid); }
    $('eval-all').innerHTML = metricsTable(rows, k, (m, key) => `Average ${METRIC_INFO[key].name.replace('@k', `@${k}`)} over ${plural(lq.length, 'labelled query')}.`)
      + `<p class="muted small">${plural(lq.length, 'query')}: ${lq.map((x) => esc(x.q)).join(' · ')}</p>`;
    if (!vecMissing && rows.hybrid && ins.length < 3) {
      const h = rows.hybrid.ndcg, b = rows.bm25.ndcg, v = rows.vector.ndcg;
      if (h >= b && h >= v) ins.push(`Across all ${lq.length} labelled queries, hybrid has the best average nDCG@${k} (${fmt(h)} vs BM25 ${fmt(b)} and vector ${fmt(v)}).`);
      else ins.push(`Across all ${lq.length} labelled queries, average nDCG@${k}: BM25 ${fmt(b)}, vector ${fmt(v)}, hybrid ${fmt(h)}.`);
    }
  }
  $('insights').innerHTML = ins.slice(0, 3).map((s) => `<li>${s}</li>`).join('');
}

function insights(rk, rel, k) {
  const out = [];
  const found = (list) => (list ? list.slice(0, k).filter((d) => rel.has(d)) : null);
  const f = { BM25: found(rk.bm), Vector: found(rk.vec), Hybrid: found(rk.hyb) };
  const n = rel.size;
  const avail = Object.entries(f).filter(([, v]) => v);
  const bestCount = Math.max(...avail.map(([, v]) => v.length));
  const winnerList = avail.filter(([, v]) => v.length === bestCount).map(([m]) => m);
  const winners = winnerList.length === avail.length && avail.length > 1
    ? (avail.length === 2 ? 'Both methods' : 'All three methods')
    : winnerList.length > 1 ? `${winnerList.slice(0, -1).join(', ')} and ${winnerList.at(-1)}` : winnerList[0];
  if (bestCount === 0) out.push(`No method found ${n === 1 ? 'the relevant document' : 'any of the relevant documents'} in the top ${k}.`);
  else if (bestCount === n) out.push(`${winners} found ${n === 1 ? 'the relevant document' : `all ${n} relevant documents`} in the top ${k}.`);
  else out.push(`${winners} found the most relevant documents: ${bestCount} of ${n} in the top ${k}.`);

  const raw = rawWordForTerm(state.query);
  if (f.Vector) {
    const missedByBm = [...rel].filter((d) => !f.BM25.includes(d) && f.Vector.includes(d));
    for (const d of missedByBm.slice(0, 1)) {
      const ex = BM.explain(derived.index, d - 1, results.terms, state.bm25);
      const miss = ex.rows.filter((x) => x.tf === 0).map((x) => `“${esc(raw.get(x.term) || x.term)}”`);
      out.push(`BM25 missed doc ${d} because it does not contain ${miss.length ? miss.join(', ') : 'enough of the query words'}; vector search ranked it #${results.vecRank.get(d - 1)}${miss.length ? ' because it says the same thing in other words' : ''}.`);
    }
    const missedByVec = [...rel].filter((d) => f.BM25.includes(d) && !f.Vector.includes(d));
    for (const d of missedByVec.slice(0, 1)) {
      const ex = BM.explain(derived.index, d - 1, results.terms, state.bm25);
      const top = ex.rows.filter((x) => x.tf > 0).sort((a, b) => b.contribution - a.contribution)[0];
      out.push(`Vector search ranked doc ${d} only #${results.vecRank.get(d - 1)}; BM25 found it at #${results.bmRank.get(d - 1)} thanks to the exact term “${esc(top ? raw.get(top.term) || top.term : '')}”.`);
    }
  } else if (n && f.BM25.length < n) {
    out.push('Load the embedding model to see whether vector or hybrid search finds the documents BM25 missed.');
  }
  return out.slice(0, 3);
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
let exportTab = 'python';

function pythonExport() {
  const m = model();
  const t = state.text;
  const pyBool = (x) => (x ? 'True' : 'False');
  const docs = derived.docs.map((d) => '    ' + JSON.stringify(d) + ',').join('\n');
  const stop = [...BM.STOPWORDS].sort().join(' ');
  const weighted = state.fusion.method === 'weighted';
  return `"""
Hybrid Search Playground export: BM25 + vector search + ${weighted ? 'weighted score fusion' : 'Reciprocal Rank Fusion'}.

    pip install sentence-transformers nltk numpy

Settings: k1=${state.bm25.k1}, b=${state.bm25.b}, lowercase=${t.lowercase}, remove_punctuation=${t.removePunct},
stopwords=${t.stopwords}, stemming=${t.stem}, model=${m.python}, ${weighted ? `alpha=${state.fusion.alpha}` : `rrf_k=${state.fusion.rrfK}`}

BM25 below is hand-written with the Lucene / Elasticsearch formula:
    IDF = ln(1 + (N - n + 0.5) / (n + 0.5))
Note: rank_bm25's BM25Okapi uses IDF = ln((N - n + 0.5) / (n + 0.5)) with an epsilon floor for
negative values, so its absolute scores (and sometimes its order) differ from Elasticsearch and
from this playground. That is why it is not used here.
Vector scores differ slightly from the browser, which runs an 8-bit quantised copy of the model.
"""
import math
import re
from collections import Counter

import numpy as np
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer

DOCS = [
${docs}
]
QUERY = ${JSON.stringify(state.query)}
TOP_K = ${state.k}

K1, B = ${state.bm25.k1}, ${state.bm25.b}
LOWERCASE, REMOVE_PUNCT, REMOVE_STOPWORDS, STEM = ${pyBool(t.lowercase)}, ${pyBool(t.removePunct)}, ${pyBool(t.stopwords)}, ${pyBool(t.stem)}
MODEL_NAME = ${JSON.stringify(m.python)}
QUERY_PREFIX = ${JSON.stringify(m.queryPrefix)}${m.queryPrefix ? '  # instruction the model expects in front of queries' : ''}
${weighted ? `ALPHA = ${state.fusion.alpha}  # weight of vector search in the fusion` : `RRF_K = ${state.fusion.rrfK}`}
FUSION_DEPTH = ${FUSION_DEPTH}  # fuse the top 100 of each list

STOPWORDS = set("""${stop}""".split())
# MARTIN_EXTENSIONS = the version published on Martin Porter's site (same as the playground).
_stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
WORD_RE = re.compile(r"[^\\W_]+(?:[-._][^\\W_]+)*")  # keeps ERR-504 and v2.3.1 whole


def process_word(raw):
    word = raw.lower() if LOWERCASE else raw
    pieces = [word]
    if REMOVE_PUNCT and re.search(r"[-._]", word):
        pieces += [p for p in re.split(r"[-._]+", word) if len(p) >= 2]  # ERR-504 -> err-504, err, 504
    out = []
    for p in pieces:
        if REMOVE_STOPWORDS and p.lower() in STOPWORDS:
            continue
        if STEM and re.fullmatch(r"[a-z]+", p):
            p = _stemmer.stem(p)
        if p:
            out.append(p)
    return out


def tokenize(text):
    words = WORD_RE.findall(text) if REMOVE_PUNCT else text.split()
    return [t for w in words for t in process_word(w)]


class BM25:
    """Lucene / Elasticsearch BM25 (IDF never negative)."""

    def __init__(self, corpus_tokens, k1=K1, b=B):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.tfs = [Counter(toks) for toks in corpus_tokens]
        self.lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = sum(self.lens) / self.N if self.N else 0.0
        self.df = Counter(term for tf in self.tfs for term in tf)

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query_terms):
        out = [0.0] * self.N
        for term in dict.fromkeys(query_terms):  # each distinct query term once
            if term not in self.df:
                continue
            idf = self.idf(term)
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f:
                    length = 1 - self.b + self.b * (self.lens[i] / self.avgdl if self.avgdl else 1)
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * length)
        return out


def rrf(ranked_lists, k=60, depth=FUSION_DEPTH):
    """Reciprocal Rank Fusion: sum of 1 / (k + rank), ranks start at 1."""
    fused = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking[:depth], start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def min_max(scores):
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: (1.0 if hi > 0 else 0.0) for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def weighted_fusion(bm25_top, vec_top, alpha=0.5):
    """alpha * normalised vector score + (1 - alpha) * normalised BM25 score."""
    nb, nv = min_max(dict(bm25_top)), min_max(dict(vec_top))
    ids = list(dict.fromkeys([d for d, _ in bm25_top] + [d for d, _ in vec_top]))
    fused = {d: alpha * nv.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in ids}
    return sorted(fused.items(), key=lambda kv: -kv[1])


# --- BM25 -------------------------------------------------------------------
bm25 = BM25([tokenize(d) for d in DOCS])
bm25_scores = bm25.scores(list(dict.fromkeys(tokenize(QUERY))))
bm25_ranking = [i for i in sorted(range(len(DOCS)), key=lambda i: -bm25_scores[i]) if bm25_scores[i] > 0]

# --- Vector -----------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)
doc_emb = model.encode(DOCS, normalize_embeddings=True, batch_size=32)
query_emb = model.encode([QUERY_PREFIX + QUERY], normalize_embeddings=True)[0]
vec_scores = doc_emb @ query_emb  # cosine similarity (vectors have length 1)
vec_ranking = [int(i) for i in sorted(range(len(DOCS)), key=lambda i: -vec_scores[i])]

# --- Hybrid -----------------------------------------------------------------
${weighted
    ? `hybrid = weighted_fusion(
    [(i, bm25_scores[i]) for i in bm25_ranking[:FUSION_DEPTH]],
    [(i, float(vec_scores[i])) for i in vec_ranking[:FUSION_DEPTH]],
    alpha=ALPHA,
)`
    : 'hybrid = rrf([bm25_ranking, vec_ranking], k=RRF_K)'}

for name, ranking in [
    ("BM25", [(i, bm25_scores[i]) for i in bm25_ranking]),
    ("Vector", [(i, float(vec_scores[i])) for i in vec_ranking]),
    ("Hybrid", hybrid),
]:
    print(f"\\n{name}")
    for rank, (i, score) in enumerate(ranking[:TOP_K], start=1):
        print(f"  #{rank}  doc {i + 1:<4} {score:.4f}  {DOCS[i][:70]}")
`;
}

function esExport() {
  const t = state.text;
  const filters = [];
  if (t.lowercase) filters.push('lowercase');
  if (t.stopwords) filters.push('playground_stop');
  if (t.stem) filters.push('porter_stem');
  const analysis = { analyzer: { playground_analyzer: { type: 'custom', tokenizer: t.removePunct ? 'standard' : 'whitespace', filter: filters } } };
  if (t.stopwords) analysis.filter = { playground_stop: { type: 'stop', stopwords: [...BM.STOPWORDS].sort(), ignore_case: true } };
  const body = {
    settings: {
      index: { similarity: { playground_bm25: { type: 'BM25', k1: state.bm25.k1, b: state.bm25.b } } },
      analysis,
    },
    mappings: {
      properties: {
        text: { type: 'text', analyzer: 'playground_analyzer', similarity: 'playground_bm25' },
      },
    },
  };
  return JSON.stringify(body, null, 2);
}

function esNote() {
  const notes = ['Use as the body of PUT /my-index (Elasticsearch 7+/8+ and OpenSearch). Search the "text" field with a match query.'];
  if (state.text.removePunct) notes.push('Difference: the standard tokenizer splits ERR-504 into "err" and "504" but does not also keep "err-504" as one token like this playground does.');
  if (state.text.stem && !state.text.lowercase) notes.push('Warning: porter_stem expects lowercase input; with Lowercase off, capitalised words will not be stemmed (same as here).');
  return notes.join(' ');
}

function workspaceObject() {
  return {
    app: 'hybrid-search-playground',
    version: 1,
    savedAt: new Date().toISOString(),
    datasetId: state.datasetId,
    documentsText: state.datasetId === 'custom' ? state.customText : undefined,
    split: state.split,
    separator: state.separator,
    query: state.query,
    labels: state.labels,
    settings: { k: state.k, text: state.text, bm25: state.bm25, modelId: state.modelId, fusion: state.fusion },
  };
}

function csvExport() {
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rel = getLabels(state.query);
  const lines = ['method,rank,doc,score,relevant,text'];
  const add = (method, list, getId) => (list || []).slice(0, state.k).forEach((x, i) => {
    const id = getId(x);
    lines.push([method, i + 1, id + 1, x.score.toFixed(6), rel.has(id + 1) ? 1 : 0, q(derived.docs[id].replace(/\s+/g, ' '))].join(','));
  });
  add('bm25', results.bm, (x) => x.idx);
  add('vector', results.vec, (x) => x.idx);
  add('hybrid', results.hyb, (x) => x.id);
  return lines.join('\n') + '\n';
}

function exportContent() {
  if (exportTab === 'python') return { text: pythonExport(), name: 'hybrid_search.py', type: 'text/x-python' };
  if (exportTab === 'es') return { text: esExport(), name: 'elasticsearch-index-settings.json', type: 'application/json' };
  if (exportTab === 'csv') return { text: csvExport(), name: 'search-results.csv', type: 'text/csv' };
  return { text: JSON.stringify(workspaceObject(), null, 2), name: 'search-workspace.json', type: 'application/json' };
}

function renderExport() {
  document.querySelectorAll('[data-etab]').forEach((b) => b.classList.toggle('active', b.dataset.etab === exportTab));
  $('workspace-open').hidden = exportTab !== 'workspace';
  $('workspace-reset').hidden = exportTab !== 'workspace';
  const { text } = exportContent();
  $('export-code').textContent = text.length > 200000 ? text.slice(0, 200000) + '\n… (preview cut; Copy / Download include everything)' : text;
  const msg = $('export-msg');
  if (!msg.dataset.sticky) {
    msg.className = 'msg';
    msg.textContent = exportTab === 'es' ? esNote()
      : exportTab === 'python' ? 'Python 3.9+. Uses the same documents, query, settings and model as the page right now.'
        : exportTab === 'csv' ? `Top ${state.k} of each method for the current query.`
          : 'Also auto-saved in this browser (localStorage), so a reload keeps your work.';
  }
}
const renderExportSoon = debounce(renderExport, 300);

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
const save = debounce(() => {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(workspaceObject())); } catch { /* quota or private mode: ignore */ }
}, 400);

function applyWorkspace(ws) {
  if (!ws || ws.app !== 'hybrid-search-playground') throw new Error('This is not a Hybrid Search Playground workspace file.');
  const s = defaultState();
  s.datasetId = ws.datasetId === 'custom' || getDataset(ws.datasetId) ? ws.datasetId : s.datasetId;
  s.customText = typeof ws.documentsText === 'string' ? ws.documentsText : '';
  s.split = ['blank', 'line', 'custom'].includes(ws.split) ? ws.split : 'blank';
  s.separator = typeof ws.separator === 'string' ? ws.separator : '---';
  s.query = typeof ws.query === 'string' ? ws.query : '';
  s.labels = ws.labels && typeof ws.labels === 'object' ? ws.labels : {};
  const st = ws.settings || {};
  if ([3, 5, 10].includes(st.k)) s.k = st.k;
  if (st.text) for (const key of Object.keys(s.text)) if (typeof st.text[key] === 'boolean') s.text[key] = st.text[key];
  if (st.bm25) {
    if (Number.isFinite(st.bm25.k1)) s.bm25.k1 = Math.min(3, Math.max(0, st.bm25.k1));
    if (Number.isFinite(st.bm25.b)) s.bm25.b = Math.min(1, Math.max(0, st.bm25.b));
  }
  if (MODELS[st.modelId]) s.modelId = st.modelId;
  if (st.fusion) {
    if (['rrf', 'weighted'].includes(st.fusion.method)) s.fusion.method = st.fusion.method;
    if (Number.isFinite(st.fusion.rrfK)) s.fusion.rrfK = Math.min(100, Math.max(1, Math.round(st.fusion.rrfK)));
    if (Number.isFinite(st.fusion.alpha)) s.fusion.alpha = Math.min(1, Math.max(0, st.fusion.alpha));
  }
  state = s;
}

function syncSettingsControls() {
  $('k1').value = state.bm25.k1; $('k1-out').textContent = fmt(state.bm25.k1);
  $('b').value = state.bm25.b; $('b-out').textContent = fmt(state.bm25.b);
  $('opt-lowercase').checked = state.text.lowercase;
  $('opt-punct').checked = state.text.removePunct;
  $('opt-stop').checked = state.text.stopwords;
  $('opt-stem').checked = state.text.stem;
  $('model').value = state.modelId;
  $('model-note').textContent = model().note;
  document.querySelectorAll('input[name=fusion]').forEach((r) => { r.checked = r.value === state.fusion.method; });
  $('rrfk').value = state.fusion.rrfK; $('rrfk-out').textContent = state.fusion.rrfK;
  $('alpha').value = state.fusion.alpha; $('alpha-out').textContent = fmt(state.fusion.alpha);
  $('rrfk-row').hidden = state.fusion.method !== 'rrf';
  $('alpha-row').hidden = state.fusion.method !== 'weighted';
}

function renderAll() {
  renderInputs();
  renderResults();
  renderDetail();
  renderEvaluation();
  renderExportSoon();
  save();
}

// ---------------------------------------------------------------------------
// Tooltips: one shared floating tooltip for every [data-tip] element
// ---------------------------------------------------------------------------
function initTooltips() {
  const tip = $('tooltip');
  let timer = null, target = null, suppressed = false;
  const mouse = { x: 0, y: 0 };
  const place = () => {
    const pad = 12;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = mouse.x + 14, y = mouse.y + 18;
    if (x + w + pad > window.innerWidth) x = Math.max(pad, mouse.x - w - 14);
    if (y + h + pad > window.innerHeight) y = Math.max(pad, mouse.y - h - 14);
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  };
  const show = (el) => {
    const text = el.getAttribute('data-tip');
    if (!text || suppressed) return;
    tip.textContent = text;
    tip.hidden = false;
    place();
  };
  const hide = () => { clearTimeout(timer); tip.hidden = true; };
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest?.('[data-tip]');
    if (el === target) return;
    hide();
    suppressed = false;
    target = el;
    if (el) timer = setTimeout(() => show(el), 350);
  });
  document.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX; mouse.y = e.clientY;
    if (!tip.hidden) place();
  }, { passive: true });
  document.addEventListener('mouseleave', () => { hide(); target = null; });
  const dismiss = () => { hide(); suppressed = true; };
  document.addEventListener('mousedown', dismiss, true);
  document.addEventListener('keydown', dismiss, true);
  window.addEventListener('scroll', dismiss, { capture: true, passive: true });
  window.addEventListener('blur', hide);
  // keyboard users: show tips on focus of "?" markers
  document.addEventListener('focusin', (e) => {
    const el = e.target.closest?.('.help[data-tip]');
    if (!el) return;
    const r = el.getBoundingClientRect();
    mouse.x = r.right; mouse.y = r.bottom;
    suppressed = false;
    show(el);
  });
  document.addEventListener('focusout', (e) => { if (e.target.closest?.('.help[data-tip]')) hide(); });
  document.querySelectorAll('.help').forEach((el) => el.setAttribute('tabindex', '0'));
}

// ---------------------------------------------------------------------------
// Learn: similarity metrics canvas
// ---------------------------------------------------------------------------
function initSimilarityCanvas() {
  const canvas = $('sim-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, S = 85; // pixels per unit
  const O = { x: W / 2, y: H / 2 };
  const vecs = [
    { name: 'Q', label: 'Query', color: '#0f766e', x: 1.0, y: 0.3 },
    { name: 'A', label: 'Doc A', color: '#c2700c', x: 0.6, y: 0.25 },
    { name: 'B', label: 'Doc B', color: '#6d4ad6', x: 1.2, y: 1.9 },
  ];
  let drag = null;
  const norm = () => $('sim-normalise').checked;
  const shown = (v) => {
    if (!norm()) return v;
    const l = Math.hypot(v.x, v.y) || 1;
    return { ...v, x: v.x / l, y: v.y / l };
  };
  const toPx = (v) => ({ x: O.x + v.x * S, y: O.y - v.y * S });

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#eceef1'; ctx.lineWidth = 1;
    for (let i = -2; i <= 2; i++) {
      ctx.beginPath(); ctx.moveTo(O.x + i * S, 0); ctx.lineTo(O.x + i * S, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, O.y + i * S); ctx.lineTo(W, O.y + i * S); ctx.stroke();
    }
    ctx.strokeStyle = '#c9ced6';
    ctx.beginPath(); ctx.moveTo(0, O.y); ctx.lineTo(W, O.y); ctx.moveTo(O.x, 0); ctx.lineTo(O.x, H); ctx.stroke();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#9aa3ae';
    ctx.beginPath(); ctx.arc(O.x, O.y, S, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#9aa3ae'; ctx.font = '11px system-ui'; ctx.fillText('length 1', O.x + S * 0.72, O.y - S * 0.74);
    for (const v of vecs.map(shown)) {
      const p = toPx(v);
      const ang = Math.atan2(p.y - O.y, p.x - O.x);
      ctx.strokeStyle = v.color; ctx.fillStyle = v.color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(O.x, O.y); ctx.lineTo(p.x, p.y); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - 12 * Math.cos(ang - 0.4), p.y - 12 * Math.sin(ang - 0.4));
      ctx.lineTo(p.x - 12 * Math.cos(ang + 0.4), p.y - 12 * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 2.5; ctx.stroke();
      ctx.fillStyle = v.color; ctx.font = 'bold 13px system-ui';
      ctx.fillText(`${v.name}`, p.x + 10, p.y - 10);
    }
    renderSimTable();
  }

  function renderSimTable() {
    const [q, a, b] = vecs.map(shown);
    const dot = (u, v) => u.x * v.x + u.y * v.y;
    const len = (u) => Math.hypot(u.x, u.y);
    const m = [a, b].map((d) => ({
      d, cos: dot(q, d) / ((len(q) * len(d)) || 1), dot: dot(q, d), dist: Math.hypot(q.x - d.x, q.y - d.y),
    }));
    const win = { cos: m[0].cos >= m[1].cos ? 0 : 1, dot: m[0].dot >= m[1].dot ? 0 : 1, dist: m[0].dist <= m[1].dist ? 0 : 1 };
    const f = (x) => x.toFixed(3);
    const v2 = (u) => `(${u.x.toFixed(2)}, ${u.y.toFixed(2)})`;
    $('sim-table').innerHTML = `<thead><tr><th data-tip="Each document arrow compared with the query arrow Q.">Compared with Q</th>
      <th data-tip="cos = (Q · D) / (|Q| × |D|). Only the angle matters. Higher = more similar.">Cosine ↑</th>
      <th data-tip="Q · D = Qx × Dx + Qy × Dy. Angle and length both matter. Higher = more similar.">Dot product ↑</th>
      <th data-tip="√((Qx − Dx)² + (Qy − Dy)²): the gap between the arrow tips. Lower = more similar.">Euclidean distance ↓</th></tr></thead><tbody>
      ${m.map((r, i) => `<tr><td style="color:${r.d.color};font-weight:700" data-tip="${esc(`${r.d.label} as (x, y)${norm() ? ', scaled to length 1' : ''}. Its length is ${len(r.d).toFixed(2)}; the query Q is (${q.x.toFixed(2)}, ${q.y.toFixed(2)}).`)}">${r.d.label} ${v2(r.d)}</td>
        <td class="${win.cos === i ? 'win' : ''}" data-tip="${esc(`cos = (${f(q.x)}×${f(r.d.x)} + ${f(q.y)}×${f(r.d.y)}) / (${f(len(q))} × ${f(len(r.d))}) = ${f(r.cos)}`)}">${f(r.cos)}</td>
        <td class="${win.dot === i ? 'win' : ''}" data-tip="${esc(`Q · ${r.d.name} = ${f(q.x)}×${f(r.d.x)} + ${f(q.y)}×${f(r.d.y)} = ${f(r.dot)}`)}">${f(r.dot)}</td>
        <td class="${win.dist === i ? 'win' : ''}" data-tip="${esc(`√((${f(q.x)} − ${f(r.d.x)})² + (${f(q.y)} − ${f(r.d.y)})²) = ${f(r.dist)}`)}">${f(r.dist)}</td></tr>`).join('')}
      </tbody>`;
    const names = ['A', 'B'];
    const agree = win.cos === win.dot && win.dot === win.dist;
    $('sim-verdict').innerHTML = agree
      ? `✅ All three measures agree: <strong>Doc ${names[win.cos]}</strong> is closest to the query.${norm() ? ' (With normalised vectors they always agree.)' : ''}`
      : `⚠️ They disagree: cosine picks Doc ${names[win.cos]}, dot product picks Doc ${names[win.dot]}, distance picks Doc ${names[win.dist]}. Tick “Normalise vectors” to make them agree.`;
  }

  const pointer = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) };
  };
  canvas.addEventListener('pointerdown', (e) => {
    const p = pointer(e);
    let best = null, bd = 20;
    vecs.forEach((v) => { const t = toPx(shown(v)); const d = Math.hypot(t.x - p.x, t.y - p.y); if (d < bd) { bd = d; best = v; } });
    if (best) { drag = best; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const p = pointer(e);
    let x = (p.x - O.x) / S, y = (O.y - p.y) / S;
    const l = Math.hypot(x, y);
    if (l < 0.15) { x *= 0.15 / (l || 1); y *= 0.15 / (l || 1); }
    drag.x = Math.max(-2.5, Math.min(2.5, x)); drag.y = Math.max(-2.5, Math.min(2.5, y));
    draw();
  });
  const end = () => { drag = null; canvas.style.cursor = ''; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  $('sim-normalise').addEventListener('change', draw);
  draw();
}

// ---------------------------------------------------------------------------
// File opening
// ---------------------------------------------------------------------------
function parseCSV(text) {
  const rows = [];
  let row = [], field = '', inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; } else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((x) => x.trim()));
}

function openFile(file) {
  const msg = $('file-msg');
  msg.className = 'msg';
  if (file.size > MAX_BYTES) {
    msg.className = 'msg err';
    msg.textContent = `“${file.name}” is ${(file.size / 1048576).toFixed(1)} MB. The limit is 2 MB so everything stays fast in the browser. Try a smaller file or split it.`;
    return;
  }
  const reader = new FileReader();
  reader.onerror = () => { msg.className = 'msg err'; msg.textContent = `Could not read “${file.name}”.`; };
  reader.onload = () => {
    let text = String(reader.result || '');
    let split = state.split;
    if (/\.csv$/i.test(file.name) || file.type === 'text/csv') {
      const rows = parseCSV(text);
      const header = rows[0]?.map((h) => h.trim().toLowerCase()) || [];
      const col = header.findIndex((h) => ['text', 'content', 'body', 'document', 'doc'].includes(h));
      const body = col >= 0 ? rows.slice(1).map((r) => r[col] || '') : rows.map((r) => r.join(' '));
      text = body.map((d) => d.replace(/\s*\r?\n\s*/g, ' ').trim()).filter(Boolean).join('\n');
      split = 'line';
    }
    let docs = splitDocs(text, split, state.separator);
    let note = '';
    if (docs.length > MAX_DOCS) {
      note = ` Kept the first ${MAX_DOCS.toLocaleString()} of ${docs.length.toLocaleString()} documents (the limit).`;
      docs = docs.slice(0, MAX_DOCS);
      text = docs.join(split === 'line' ? '\n' : split === 'custom' ? state.separator : '\n\n');
    }
    state.datasetId = 'custom';
    state.customText = text;
    state.split = split;
    state.selected = null;
    state.query = '';
    state.labels.custom = {}; // old labels point at document numbers of the previous text
    $('docs').value = text;
    msg.className = note ? 'msg warn' : 'msg';
    msg.textContent = `Opened “${file.name}” locally: ${plural(docs.length, 'document')}.${note}`;
    update();
  };
  reader.readAsText(file);
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
function selectTopBm25() {
  state.selected = results?.bm[0]?.idx ?? null;
}

function initEvents() {
  const dsSel = $('dataset');
  dsSel.innerHTML = DATASETS.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join('') + '<option value="custom">My documents</option>';
  dsSel.addEventListener('change', () => {
    if (state.datasetId !== 'custom' && dsSel.value === 'custom' && !state.customText) state.customText = '';
    state.datasetId = dsSel.value;
    const ds = currentDataset();
    state.query = ds ? ds.queries[0].q : '';
    state.selected = null;
    $('docs').value = docsText();
    $('file-msg').textContent = '';
    update();
    selectTopBm25();
    renderAll();
  });

  const onDocsInput = debounce(() => {
    const text = $('docs').value;
    if (state.datasetId !== 'custom') {
      if (text === docsText()) return;
      state.datasetId = 'custom';
      state.split = 'blank';
    }
    state.customText = text;
    update();
  }, 250);
  $('docs').addEventListener('input', onDocsInput);

  $('split').addEventListener('change', () => {
    if (state.datasetId !== 'custom') { state.customText = docsText(); state.datasetId = 'custom'; }
    state.split = $('split').value;
    state.selected = null;
    update();
  });
  $('separator').addEventListener('input', debounce(() => { state.separator = $('separator').value; update(); }, 250));

  $('open-file').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (e) => { const f = e.target.files?.[0]; if (f) openFile(f); e.target.value = ''; });

  const onQuery = debounce(() => { state.query = $('query').value; update(); }, 150);
  $('query').addEventListener('input', onQuery);

  $('examples').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-q]');
    if (!b) return;
    state.query = b.dataset.q;
    $('query').value = state.query;
    update();
    selectTopBm25();
    renderAll();
  });

  $('topk').addEventListener('change', () => { state.k = +$('topk').value; update(); });

  // results: select / relevance
  document.addEventListener('change', (e) => {
    const cb = e.target.closest?.('input[data-rel]');
    if (!cb) return;
    setLabel(state.query, +cb.dataset.rel + 1, cb.checked);
    renderAll();
  });
  document.querySelector('.columns').addEventListener('click', (e) => {
    if (e.target.closest('.rel-check')) return;
    const li = e.target.closest('.result');
    if (!li) return;
    state.selected = +li.dataset.idx;
    showTab('detail');
    renderResults();
    renderDetail();
  });
  document.querySelector('.columns').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const li = e.target.closest('.result');
    if (!li || e.target.closest('.rel-check')) return;
    e.preventDefault();
    state.selected = +li.dataset.idx;
    showTab('detail');
    renderResults();
    renderDetail();
  });

  // actions (load model, cancel, resume)
  document.addEventListener('click', (e) => {
    const a = e.target.closest('[data-action]');
    if (!a) return;
    if (a.dataset.action === 'load-model') loadModel();
    if (a.dataset.action === 'cancel-embed') { cancelDocJob(true); update(); }
    if (a.dataset.action === 'resume-embed') { vec.docJobCancelled = false; vec.lastCancelledKey = null; ensureDocEmbeddings(true); update(); }
  });

  // tabs
  document.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

  // settings
  const slider = (id, apply) => $(id).addEventListener('input', () => { apply(+$(id).value); syncSettingsControls(); update(); });
  slider('k1', (v) => { state.bm25.k1 = v; });
  slider('b', (v) => { state.bm25.b = v; });
  slider('rrfk', (v) => { state.fusion.rrfK = v; });
  slider('alpha', (v) => { state.fusion.alpha = v; });
  $('bm25-reset').addEventListener('click', () => { state.bm25 = { ...BM.DEFAULT_BM25 }; syncSettingsControls(); update(); });
  const toggle = (id, key) => $(id).addEventListener('change', () => { state.text[key] = $(id).checked; update(); });
  toggle('opt-lowercase', 'lowercase');
  toggle('opt-punct', 'removePunct');
  toggle('opt-stop', 'stopwords');
  toggle('opt-stem', 'stem');
  $('model').innerHTML = Object.entries(MODELS).map(([id, m]) => `<option value="${id}">${esc(m.label)}</option>`).join('');
  $('model').setAttribute('data-tip', 'Which embedding model vector search uses. Both run in your browser.\nChanging the model needs a new one-time download, and all documents are embedded again.');
  $('model').addEventListener('change', () => {
    cancelDocJob();
    state.modelId = $('model').value;
    syncSettingsControls();
    update();
  });
  document.querySelectorAll('input[name=fusion]').forEach((r) => r.addEventListener('change', () => {
    state.fusion.method = r.value; syncSettingsControls(); update();
  }));

  $('clear-labels').addEventListener('click', () => {
    if (!state.query.trim()) return;
    (state.labels[dsKey()] ||= {})[normQ(state.query)] = [];
    save();
    renderAll();
  });

  // export
  document.querySelectorAll('[data-etab]').forEach((b) => b.addEventListener('click', () => {
    exportTab = b.dataset.etab; delete $('export-msg').dataset.sticky; renderExport();
  }));
  const flash = (text, cls = 'msg') => {
    const m = $('export-msg');
    m.className = cls; m.textContent = text; m.dataset.sticky = '1';
    setTimeout(() => { delete m.dataset.sticky; }, 2500);
  };
  $('export-copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(exportContent().text); flash('Copied to the clipboard.'); } catch { flash('Copy failed. Select the text and copy it by hand.', 'msg err'); }
  });
  $('export-download').addEventListener('click', () => {
    const { text, name, type } = exportContent();
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    flash(`Saved ${name}.`);
  });
  $('workspace-open').addEventListener('click', () => $('workspace-input').click());
  $('workspace-input').addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (!f) return;
    if (f.size > MAX_BYTES * 2) { flash('That file is too large for a workspace (limit 4 MB).', 'msg err'); return; }
    const r = new FileReader();
    r.onload = () => {
      try {
        applyWorkspace(JSON.parse(String(r.result)));
        $('docs').value = docsText();
        syncSettingsControls();
        update();
        flash(`Opened workspace “${f.name}”.`);
      } catch (err) { flash(`Could not open: ${err.message}`, 'msg err'); }
    };
    r.readAsText(f);
  });
  $('workspace-reset').addEventListener('click', () => {
    if (!confirm('Forget the auto-saved workspace (your documents, labels and settings) in this browser?')) return;
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
    state = defaultState();
    $('docs').value = docsText();
    syncSettingsControls();
    update();
    selectTopBm25();
    renderAll();
    flash('Reset to the defaults.');
  });
}

function showTab(name) {
  document.querySelectorAll('[data-tab]').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $('tab-detail').hidden = name !== 'detail';
  $('tab-settings').hidden = name !== 'settings';
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) applyWorkspace(JSON.parse(saved));
  } catch { state = defaultState(); }
  initEvents();
  initTooltips();
  syncSettingsControls();
  $('docs').value = docsText();
  update();
  if (state.selected === null) { selectTopBm25(); renderAll(); }
  initSimilarityCanvas();
}

boot();
