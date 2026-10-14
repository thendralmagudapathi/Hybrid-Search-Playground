import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  porterStem, tokenize, queryTerms, buildIndex, idf, rank, scoreAll, explain, DEFAULT_TEXT_OPTIONS,
} from '../public/bm25.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('Porter stemmer matches known word → stem pairs', () => {
  // Pairs from Porter's 1980 paper and the reference vocabulary.
  const pairs = {
    caresses: 'caress', ponies: 'poni', ties: 'ti', caress: 'caress', cats: 'cat',
    feed: 'feed', agreed: 'agre', plastered: 'plaster', bled: 'bled', motoring: 'motor', sing: 'sing',
    conflated: 'conflat', troubled: 'troubl', sized: 'size', hopping: 'hop', tanned: 'tan',
    falling: 'fall', hissing: 'hiss', fizzed: 'fizz', failing: 'fail', filing: 'file',
    happy: 'happi', sky: 'sky', running: 'run', relational: 'relat', conditional: 'condit',
    rational: 'ration', valenci: 'valenc', digitizer: 'digit',
    radicalli: 'radic', differentli: 'differ', vileli: 'vile', analogousli: 'analog',
    vietnamization: 'vietnam', predication: 'predic', operator: 'oper', feudalism: 'feudal',
    decisiveness: 'decis', hopefulness: 'hope', callousness: 'callous', formaliti: 'formal',
    sensitiviti: 'sensit', sensibiliti: 'sensibl', triplicate: 'triplic', formative: 'form',
    formalize: 'formal', electriciti: 'electr', electrical: 'electr', hopeful: 'hope',
    goodness: 'good', revival: 'reviv', allowance: 'allow', inference: 'infer', airliner: 'airlin',
    gyroscopic: 'gyroscop', adjustable: 'adjust', defensible: 'defens', irritant: 'irrit',
    replacement: 'replac', adjustment: 'adjust', dependent: 'depend', adoption: 'adopt',
    homologou: 'homolog', communism: 'commun', activate: 'activ', angulariti: 'angular',
    homologous: 'homolog', effective: 'effect', bowdlerize: 'bowdler', probate: 'probat',
    rate: 'rate', cease: 'ceas', controll: 'control', roll: 'roll', generalization: 'gener',
    generalizations: 'gener', connection: 'connect', connected: 'connect', connecting: 'connect',
  };
  for (const [w, s] of Object.entries(pairs)) assert.equal(porterStem(w), s, `${w} → ${s}`);
});

test('tokenizer keeps identifiers whole and adds their parts', () => {
  const t = tokenize('Got ERR-504 after upgrading to v2.3.1!', DEFAULT_TEXT_OPTIONS);
  // "after" and "to" are stopwords; parts shorter than 2 characters ("3", "1") are dropped.
  assert.deepEqual(t, ['got', 'err-504', 'err', '504', 'upgrad', 'v2.3.1', 'v2']);
});

test('text toggles change the tokens', () => {
  const opts = { lowercase: false, removePunct: true, stopwords: false, stem: false };
  assert.deepEqual(tokenize('The Running dogs', opts), ['The', 'Running', 'dogs']);
  assert.deepEqual(tokenize('The running dogs', { ...opts, lowercase: true, stopwords: true, stem: true }), ['run', 'dog']);
  assert.deepEqual(tokenize('timeout. timeout', { ...opts, removePunct: false }), ['timeout.', 'timeout']);
  assert.deepEqual(queryTerms('dog dogs dog'), ['dog']);
});

// Hand-calculated example. Three documents (already tokenised):
//   d0 = [cat, sat]                |d0| = 2
//   d1 = [cat, cat, dog]           |d1| = 3
//   d2 = [dog, bird, fish, fish]   |d2| = 4
// N = 3, avgdl = (2 + 3 + 4) / 3 = 3, k1 = 1.2, b = 0.75
const DOCS = [['cat', 'sat'], ['cat', 'cat', 'dog'], ['dog', 'bird', 'fish', 'fish']];

test('BM25 hand-calculated scores', () => {
  const index = buildIndex(DOCS);
  assert.equal(index.N, 3);
  close(index.avgdl, 3);

  // IDF(cat): n = 2 → ln(1 + (3 − 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.470004
  close(idf(3, 2), 0.470004);
  // IDF(fish): n = 1 → ln(1 + 2.5 / 1.5) = ln(2.666667) = 0.980829
  close(idf(3, 1), 0.980829);

  // Query "cat"
  // d0: L = 1 − 0.75 + 0.75·2/3 = 0.75 ; tf part = 1·2.2 / (1 + 1.2·0.75) = 2.2 / 1.9 = 1.157895
  //     score = 0.470004 · 1.157895 = 0.544215
  // d1: L = 1 ; tf part = 2·2.2 / (2 + 1.2) = 4.4 / 3.2 = 1.375 ; score = 0.470004 · 1.375 = 0.646255
  // d2: no "cat" → 0
  const s = scoreAll(index, ['cat'], { k1: 1.2, b: 0.75 });
  close(s[0], 0.544215);
  close(s[1], 0.646255);
  assert.equal(s[2], 0);
  assert.deepEqual(rank(index, ['cat']).map((r) => r.idx), [1, 0]);

  // Query "fish dog" on d2: L = 0.25 + 0.75·4/3 = 1.25
  //   fish: tf 2 → 4.4 / (2 + 1.2·1.25) = 4.4 / 3.5 = 1.257143 → 0.980829 · 1.257143 = 1.233042
  //   dog : tf 1 → 2.2 / (1 + 1.5) = 0.88                      → 0.470004 · 0.88     = 0.413603
  //   total = 1.646646
  const ex = explain(index, 2, ['fish', 'dog']);
  close(ex.rows[0].lengthFactor, 1.25);
  close(ex.rows[0].contribution, 1.233042);
  close(ex.rows[1].contribution, 0.413603);
  close(ex.total, 1.646646);
});

test('IDF is never negative, even for a term in every document', () => {
  for (let N = 1; N <= 50; N++) for (let df = 0; df <= N; df++) assert.ok(idf(N, df) > 0);
  const index = buildIndex([['a'], ['a'], ['a', 'b']]);
  assert.ok(scoreAll(index, ['a'])[0] > 0);
});

test('b = 0 removes the length effect', () => {
  // Same tf for "cat", very different lengths.
  const index = buildIndex([['cat'], ['cat', 'x', 'y', 'z', 'w', 'v', 'u', 't']]);
  const s0 = scoreAll(index, ['cat'], { k1: 1.2, b: 0 });
  close(s0[0], s0[1]);
  const s1 = scoreAll(index, ['cat'], { k1: 1.2, b: 0.75 });
  assert.ok(s1[0] > s1[1]);
});

test('k1 = 0 makes repeats count the same as one occurrence', () => {
  const index = buildIndex([['cat', 'x'], ['cat', 'cat'], ['y', 'z']]);
  const s = scoreAll(index, ['cat'], { k1: 0, b: 0.75 });
  close(s[0], s[1]);
  close(s[0], idf(3, 2)); // tf·(0+1)/(tf+0) = 1, so the score is just the IDF
});

test('empty inputs do not crash', () => {
  assert.equal(rank(buildIndex([]), ['a']).length, 0);
  assert.equal(rank(buildIndex([[], []]), ['a']).length, 0);
});
