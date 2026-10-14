// Hybrid fusion: Reciprocal Rank Fusion, min-max normalisation and weighted score fusion.
// Pure functions only. Lists are arrays ordered best-first.

export const DEFAULT_FUSION = Object.freeze({ method: 'rrf', rrfK: 60, alpha: 0.5, depth: 100 });

/**
 * Reciprocal Rank Fusion.
 * RRF(d) = Σ over lists 1 / (k + rank(d)), ranks start at 1.
 * @param {Array<Array<string|number>>} lists  ids in rank order, one array per method
 * @param {number} k      smoothing constant (60 is the value from the original paper)
 * @param {number} depth  only the top `depth` of each list take part
 * @returns {Array<{id, score, ranks: Array<number|null>}>} best first
 */
export function rrf(lists, k = 60, depth = 100) {
  const byId = new Map();
  lists.forEach((list, li) => {
    list.slice(0, depth).forEach((id, i) => {
      let e = byId.get(id);
      if (!e) byId.set(id, (e = { id, score: 0, ranks: lists.map(() => null), firstSeen: byId.size }));
      if (e.ranks[li] !== null) return; // ignore duplicates within one list
      e.ranks[li] = i + 1;
      e.score += 1 / (k + i + 1);
    });
  });
  return sortFused([...byId.values()]);
}

/**
 * Min-max normalise scores to 0..1. If every score is the same, positive scores map to 1 and
 * zero/negative to 0 (there is no spread to stretch).
 * @param {Array<{id, score}>} items
 * @returns {Map<id, number>}
 */
export function minMaxNormalize(items) {
  const out = new Map();
  if (!items.length) return out;
  let min = Infinity, max = -Infinity;
  for (const { score } of items) { if (score < min) min = score; if (score > max) max = score; }
  const range = max - min;
  for (const { id, score } of items) out.set(id, range > 0 ? (score - min) / range : (max > 0 ? 1 : 0));
  return out;
}

/**
 * Weighted score fusion: α · normalised(vector) + (1 − α) · normalised(BM25).
 * A document missing from a list gets 0 from that list.
 * @param {Array<{id, score}>} bm25List  best first
 * @param {Array<{id, score}>} vectorList best first
 * @returns {Array<{id, score, ranks:[bm25Rank|null, vectorRank|null], parts:{bm25, vector}}>}
 */
export function weightedFusion(bm25List, vectorList, alpha = 0.5, depth = 100) {
  const a = bm25List.slice(0, depth), b = vectorList.slice(0, depth);
  const na = minMaxNormalize(a), nb = minMaxNormalize(b);
  const byId = new Map();
  const touch = (id) => {
    let e = byId.get(id);
    if (!e) byId.set(id, (e = { id, score: 0, ranks: [null, null], parts: { bm25: 0, vector: 0 }, firstSeen: byId.size }));
    return e;
  };
  a.forEach(({ id }, i) => { const e = touch(id); if (e.ranks[0] === null) { e.ranks[0] = i + 1; e.parts.bm25 = na.get(id); } });
  b.forEach(({ id }, i) => { const e = touch(id); if (e.ranks[1] === null) { e.ranks[1] = i + 1; e.parts.vector = nb.get(id); } });
  for (const e of byId.values()) e.score = alpha * e.parts.vector + (1 - alpha) * e.parts.bm25;
  return sortFused([...byId.values()]);
}

// Ties: higher score first, then the better (smaller) best rank, then first appearance. Deterministic.
function sortFused(entries) {
  const best = (e) => Math.min(...e.ranks.map((r) => (r === null ? Infinity : r)));
  entries.sort((x, y) => y.score - x.score || best(x) - best(y) || x.firstSeen - y.firstSeen);
  return entries.map(({ firstSeen, ...rest }) => rest);
}
