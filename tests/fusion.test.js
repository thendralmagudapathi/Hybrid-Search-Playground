import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rrf, minMaxNormalize, weightedFusion } from '../public/fusion.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('RRF with documents in one or both lists', () => {
  // BM25:   A B C       Vector: B D A
  // A: 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = 0.032266
  // B: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
  // C: 1/(60+3)            = 0.015873   (only BM25)
  // D: 1/(60+2)            = 0.016129   (only vector)
  const out = rrf([['A', 'B', 'C'], ['B', 'D', 'A']], 60);
  assert.deepEqual(out.map((e) => e.id), ['B', 'A', 'D', 'C']);
  close(out[0].score, 0.032522, 1e-6);
  close(out[1].score, 0.032266, 1e-6);
  assert.deepEqual(out.find((e) => e.id === 'C').ranks, [3, null]);
  assert.deepEqual(out.find((e) => e.id === 'D').ranks, [null, 2]);
});

test('RRF ties break by best single rank, then first appearance', () => {
  // X is #1 then #2; Y is #2 then #1 → identical scores, identical best rank → first seen (X).
  const out = rrf([['X', 'Y'], ['Y', 'X']], 60);
  close(out[0].score, out[1].score);
  assert.deepEqual(out.map((e) => e.id), ['X', 'Y']);
  // Two docs each found once at rank 1 with k = 10: both 1/11.
  const out2 = rrf([['P'], ['Q']], 10);
  close(out2[0].score, 1 / 11);
  close(out2[1].score, 1 / 11);
  assert.deepEqual(out2.map((e) => e.id), ['P', 'Q']);
});

test('RRF respects depth and k', () => {
  const out = rrf([['A', 'B', 'C']], 1, 2); // depth 2 → C ignored; k = 1 → A = 1/2
  assert.deepEqual(out.map((e) => e.id), ['A', 'B']);
  close(out[0].score, 1 / 2);
  assert.deepEqual(rrf([], 60), []);
});

test('min-max normalisation', () => {
  const m = minMaxNormalize([{ id: 'a', score: 10 }, { id: 'b', score: 5 }, { id: 'c', score: 0 }]);
  assert.deepEqual([...m.values()], [1, 0.5, 0]);
  const same = minMaxNormalize([{ id: 'a', score: 3 }, { id: 'b', score: 3 }]);
  assert.deepEqual([...same.values()], [1, 1]); // no spread: positive scores → 1
  assert.equal(minMaxNormalize([]).size, 0);
});

test('weighted fusion', () => {
  // BM25 raw:   A 9, B 3, C 0      → norm A 1, B 1/3, C 0
  // Vector raw: B 0.8, D 0.6, A 0.4 → norm B 1, D 0.5, A 0
  // α = 0.5: A = 0.5·0 + 0.5·1 = 0.5 ; B = 0.5·1 + 0.5·(1/3) = 0.666667 ; D = 0.25 ; C = 0
  const bm = [{ id: 'A', score: 9 }, { id: 'B', score: 3 }, { id: 'C', score: 0 }];
  const vec = [{ id: 'B', score: 0.8 }, { id: 'D', score: 0.6 }, { id: 'A', score: 0.4 }];
  const out = weightedFusion(bm, vec, 0.5);
  assert.deepEqual(out.map((e) => e.id), ['B', 'A', 'D', 'C']);
  close(out[0].score, 2 / 3);
  close(out.find((e) => e.id === 'D').score, 0.25);
  assert.deepEqual(out.find((e) => e.id === 'D').ranks, [null, 2]);
  // α = 0 → BM25 order first; α = 1 → vector order.
  assert.deepEqual(weightedFusion(bm, vec, 0).slice(0, 2).map((e) => e.id), ['A', 'B']);
  assert.deepEqual(weightedFusion(bm, vec, 1).slice(0, 3).map((e) => e.id), ['B', 'D', 'A']);
});

test('weighted fusion ties are deterministic', () => {
  // A only in BM25 (norm 1), B only in vector (norm 1); α = 0.5 → both 0.5, both best rank 1.
  const out = weightedFusion([{ id: 'A', score: 2 }], [{ id: 'B', score: 0.9 }], 0.5);
  close(out[0].score, 0.5);
  close(out[1].score, 0.5);
  assert.deepEqual(out.map((e) => e.id), ['A', 'B']);
});
