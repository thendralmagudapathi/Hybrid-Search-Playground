// Ranking quality metrics with binary relevance. Pure functions; never divide by zero.
// `ranked` is an array of ids best-first; `relevant` is a Set (or array) of relevant ids.

const asSet = (r) => (r instanceof Set ? r : new Set(r));

/** Precision@k = (relevant docs in the top k) / k */
export function precisionAtK(ranked, relevant, k) {
  if (k <= 0) return 0;
  const rel = asSet(relevant);
  return ranked.slice(0, k).filter((id) => rel.has(id)).length / k;
}

/** Recall@k = (relevant docs in the top k) / (all relevant docs). 0 when nothing is relevant. */
export function recallAtK(ranked, relevant, k) {
  const rel = asSet(relevant);
  if (rel.size === 0) return 0;
  return ranked.slice(0, k).filter((id) => rel.has(id)).length / rel.size;
}

/** Reciprocal rank (within the top k): 1 / position of the first relevant doc, or 0. */
export function reciprocalRank(ranked, relevant, k = Infinity) {
  const rel = asSet(relevant);
  const n = Math.min(k, ranked.length);
  for (let i = 0; i < n; i++) if (rel.has(ranked[i])) return 1 / (i + 1);
  return 0;
}

/** MRR = average reciprocal rank over several queries. runs: [{ ranked, relevant }] */
export function meanReciprocalRank(runs, k = Infinity) {
  if (!runs.length) return 0;
  return runs.reduce((s, r) => s + reciprocalRank(r.ranked, r.relevant, k), 0) / runs.length;
}

/** DCG@k with binary gains and log2 discount: Σ rel_i / log2(i + 1), positions i from 1. */
export function dcgAtK(ranked, relevant, k) {
  const rel = asSet(relevant);
  let dcg = 0;
  ranked.slice(0, k).forEach((id, i) => { if (rel.has(id)) dcg += 1 / Math.log2(i + 2); });
  return dcg;
}

/** nDCG@k = DCG@k / ideal DCG@k. 0 when nothing is relevant. */
export function ndcgAtK(ranked, relevant, k) {
  const rel = asSet(relevant);
  const ideal = Math.min(rel.size, k);
  if (ideal === 0) return 0;
  let idcg = 0;
  for (let i = 0; i < ideal; i++) idcg += 1 / Math.log2(i + 2);
  return dcgAtK(ranked, rel, k) / idcg;
}

/** All four metrics for one ranked list. */
export function evaluate(ranked, relevant, k) {
  return {
    precision: precisionAtK(ranked, relevant, k),
    recall: recallAtK(ranked, relevant, k),
    mrr: reciprocalRank(ranked, relevant, k),
    ndcg: ndcgAtK(ranked, relevant, k),
  };
}
