import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  precisionAtK, recallAtK, reciprocalRank, meanReciprocalRank, ndcgAtK, dcgAtK, evaluate,
} from '../public/metrics.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

// Ranked list: d3 d1 d7 d2 d9 ; relevant = {d1, d2, d5}
const ranked = ['d3', 'd1', 'd7', 'd2', 'd9'];
const rel = new Set(['d1', 'd2', 'd5']);

test('precision@k', () => {
  close(precisionAtK(ranked, rel, 3), 1 / 3); // only d1 in the top 3
  close(precisionAtK(ranked, rel, 5), 2 / 5); // d1 and d2
  close(precisionAtK(['a'], rel, 5), 0);      // short list still divides by k
});

test('recall@k', () => {
  close(recallAtK(ranked, rel, 3), 1 / 3); // 1 of 3 relevant found
  close(recallAtK(ranked, rel, 5), 2 / 3); // 2 of 3 relevant found
});

test('MRR', () => {
  close(reciprocalRank(ranked, rel), 1 / 2); // first relevant at position 2
  close(reciprocalRank(ranked, rel, 1), 0);  // not inside the top 1
  // Two queries: RR 1/2 and 1/1 → MRR (0.5 + 1) / 2 = 0.75
  close(meanReciprocalRank([{ ranked, relevant: rel }, { ranked: ['d5'], relevant: rel }]), 0.75);
});

test('nDCG@k', () => {
  // DCG@5 = 1/log2(2+1) + 1/log2(4+1) = 0.630930 + 0.430677 = 1.061606
  // IDCG@5 (3 relevant) = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.630930 + 0.5 = 2.130930
  // nDCG@5 = 1.061606 / 2.130930 = 0.498189
  close(dcgAtK(ranked, rel, 5), 1.061606, 1e-6);
  close(ndcgAtK(ranked, rel, 5), 0.498189, 1e-6);
  // k = 1: ideal has min(3, 1) = 1 relevant; position 1 is not relevant → 0
  close(ndcgAtK(ranked, rel, 1), 0);
  // Perfect ranking → 1
  close(ndcgAtK(['d1', 'd2', 'd5'], rel, 3), 1);
});

test('no relevant documents: everything is 0, no division by zero', () => {
  assert.deepEqual(evaluate(ranked, new Set(), 5), { precision: 0, recall: 0, mrr: 0, ndcg: 0 });
  assert.deepEqual(evaluate([], rel, 5), { precision: 0, recall: 0, mrr: 0, ndcg: 0 });
  assert.equal(precisionAtK(ranked, rel, 0), 0);
  assert.equal(meanReciprocalRank([]), 0);
});
