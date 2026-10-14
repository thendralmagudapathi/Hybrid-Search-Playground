// BM25 keyword search: tokenizer, Porter stemmer, stopwords, index and scoring.
// Pure functions only (no DOM), so everything here can be unit-tested in Node.

// ---------------------------------------------------------------------------
// Stopwords (a compact English list, similar in spirit to Lucene/NLTK lists)
// ---------------------------------------------------------------------------
export const STOPWORDS = new Set(`a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does doesn doing don down during each few
for from further had hadn has hasn have haven having he her here hers herself him himself his how i if in into is isn it
its itself just let ll me more most mustn my myself no nor not now of off on once only or other ought our ours ourselves
out over own re same shan she should shouldn so some such t than that the their theirs them themselves then there these
they this those through to too under until up ve very was wasn we were weren what when where which while who whom why
will with won would wouldn you your yours yourself yourselves s d m o y`.split(/\s+/).filter(Boolean));

export const DEFAULT_TEXT_OPTIONS = Object.freeze({
  lowercase: true,
  removePunct: true,
  stopwords: true,
  stem: true,
});

export const DEFAULT_BM25 = Object.freeze({ k1: 1.2, b: 0.75 });

// ---------------------------------------------------------------------------
// Porter stemmer (M.F. Porter, 1980, "An algorithm for suffix stripping").
// Written from the published algorithm description.
// ---------------------------------------------------------------------------
const step2list = {
  ational: 'ate', tional: 'tion', enci: 'ence', anci: 'ance', izer: 'ize', bli: 'ble',
  alli: 'al', entli: 'ent', eli: 'e', ousli: 'ous', ization: 'ize', ation: 'ate',
  ator: 'ate', alism: 'al', iveness: 'ive', fulness: 'ful', ousness: 'ous', aliti: 'al',
  iviti: 'ive', biliti: 'ble', logi: 'log',
};
const step3list = { icate: 'ic', ative: '', alize: 'al', iciti: 'ic', ical: 'ic', ful: '', ness: '' };

const c = '[^aeiou]';          // consonant
const v = '[aeiouy]';          // vowel
const C = c + '[^aeiouy]*';    // consonant sequence
const V = v + '[aeiou]*';      // vowel sequence
const mgr0 = new RegExp('^(' + C + ')?' + V + C);                       // m > 0
const meq1 = new RegExp('^(' + C + ')?' + V + C + '(' + V + ')?$');     // m = 1
const mgr1 = new RegExp('^(' + C + ')?' + V + C + V + C);               // m > 1
const sV = new RegExp('^(' + C + ')?' + v);                             // stem contains a vowel
const cvc = new RegExp('^' + C + v + '[^aeiouwxy]$');                  // *o condition

export function porterStem(word) {
  if (word.length < 3) return word;
  let w = word;
  const firstY = w[0] === 'y';
  if (firstY) w = 'Y' + w.slice(1);

  // Step 1a: plurals
  let m;
  if ((m = /^(.+?)(ss|i)es$/.exec(w))) w = m[1] + m[2];
  else if ((m = /^(.+?)([^s])s$/.exec(w))) w = m[1] + m[2];

  // Step 1b: -eed, -ed, -ing
  if ((m = /^(.+?)eed$/.exec(w))) {
    if (mgr0.test(m[1])) w = w.slice(0, -1);
  } else if ((m = /^(.+?)(ed|ing)$/.exec(w))) {
    const stem = m[1];
    if (sV.test(stem)) {
      w = stem;
      if (/(at|bl|iz)$/.test(w)) w += 'e';
      else if (/([^aeiouylsz])\1$/.test(w)) w = w.slice(0, -1);
      else if (cvc.test(w)) w += 'e';
    }
  }

  // Step 1c: y -> i
  if ((m = /^(.+?)y$/.exec(w)) && sV.test(m[1])) w = m[1] + 'i';

  // Step 2
  if ((m = /^(.+?)(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti|logi)$/.exec(w))) {
    if (mgr0.test(m[1])) w = m[1] + step2list[m[2]];
  }

  // Step 3
  if ((m = /^(.+?)(icate|ative|alize|iciti|ical|ful|ness)$/.exec(w))) {
    if (mgr0.test(m[1])) w = m[1] + step3list[m[2]];
  }

  // Step 4
  if ((m = /^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/.exec(w))) {
    if (mgr1.test(m[1])) w = m[1];
  } else if ((m = /^(.+?)(s|t)(ion)$/.exec(w))) {
    const stem = m[1] + m[2];
    if (mgr1.test(stem)) w = stem;
  }

  // Step 5a
  if ((m = /^(.+?)e$/.exec(w))) {
    const stem = m[1];
    if (mgr1.test(stem) || (meq1.test(stem) && !cvc.test(stem))) w = stem;
  }
  // Step 5b
  if (/ll$/.test(w) && mgr1.test(w)) w = w.slice(0, -1);

  if (firstY) w = 'y' + w.slice(1);
  return w;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
// A "word" is letters/digits, optionally joined by - . _ (so ERR-504, v2.3.1, file_name stay whole).
const WORD_RE = /[\p{L}\p{N}]+(?:[-._][\p{L}\p{N}]+)*/gu;
const NON_SPACE_RE = /\S+/gu;
const JOINER_RE = /[-._]/;

// Vocabularies are small compared with token counts, so memoising the stemmer makes re-indexing much faster.
const stemCache = new Map();
function stemCached(w) {
  let s = stemCache.get(w);
  if (s === undefined) {
    if (stemCache.size > 200000) stemCache.clear();
    stemCache.set(w, (s = porterStem(w)));
  }
  return s;
}

/**
 * Turn one raw word into its processed terms.
 * Compound identifiers (ERR-504, v2.3.1) produce the whole token plus its parts (2+ chars).
 */
export function processWord(raw, opts = DEFAULT_TEXT_OPTIONS) {
  let word = opts.lowercase ? raw.toLowerCase() : raw;
  const pieces = [word];
  if (opts.removePunct && JOINER_RE.test(word)) {
    for (const part of word.split(/[-._]+/)) if (part.length >= 2) pieces.push(part);
  }
  const out = [];
  for (let p of pieces) {
    if (opts.stopwords && STOPWORDS.has(p.toLowerCase())) continue;
    if (opts.stem && /^[a-z]+$/.test(p)) p = stemCached(p);
    if (p) out.push(p);
  }
  return out;
}

/** Raw word spans in the text, each with its processed terms. Used for highlighting. */
export function analyzeWithSpans(text, opts = DEFAULT_TEXT_OPTIONS) {
  const re = opts.removePunct ? WORD_RE : NON_SPACE_RE;
  re.lastIndex = 0;
  const spans = [];
  let m;
  while ((m = re.exec(text))) {
    spans.push({ start: m.index, end: m.index + m[0].length, raw: m[0], terms: processWord(m[0], opts) });
  }
  return spans;
}

/** Text -> list of processed tokens (terms), in order, with repeats. */
export function tokenize(text, opts = DEFAULT_TEXT_OPTIONS) {
  const re = opts.removePunct ? WORD_RE : NON_SPACE_RE;
  const out = [];
  for (const m of String(text).matchAll(re)) {
    for (const t of processWord(m[0], opts)) out.push(t);
  }
  return out;
}

/** Distinct query terms (each distinct word in the query counts once). */
export function queryTerms(query, opts = DEFAULT_TEXT_OPTIONS) {
  return [...new Set(tokenize(query, opts))];
}

// ---------------------------------------------------------------------------
// Index + scoring (Lucene / Elasticsearch BM25 variant)
// ---------------------------------------------------------------------------

/** Build an inverted index from an array of token arrays (one per document). */
export function buildIndex(docTokens) {
  const N = docTokens.length;
  const docLen = new Float64Array(N);
  const docTf = new Array(N);
  const postings = new Map(); // term -> { docs: number[], tfs: number[] }
  let total = 0;
  for (let i = 0; i < N; i++) {
    const tf = new Map();
    for (const t of docTokens[i]) tf.set(t, (tf.get(t) || 0) + 1);
    docTf[i] = tf;
    docLen[i] = docTokens[i].length;
    total += docLen[i];
    for (const [t, n] of tf) {
      let p = postings.get(t);
      if (!p) postings.set(t, (p = { docs: [], tfs: [] }));
      p.docs.push(i);
      p.tfs.push(n);
    }
  }
  return { N, docLen, docTf, postings, avgdl: N ? total / N : 0 };
}

/** Lucene BM25 IDF. Never negative: the argument of ln is always > 1. */
export function idf(N, df) {
  return Math.log(1 + (N - df + 0.5) / (df + 0.5));
}

export function docFreq(index, term) {
  const p = index.postings.get(term);
  return p ? p.docs.length : 0;
}

/** 1 − b + b·|d|/avgdl : how much the document's length stretches the tf saturation. */
export function lengthFactor(docLen, avgdl, b) {
  return 1 - b + b * (avgdl > 0 ? docLen / avgdl : 1);
}

/** One term's contribution: IDF · tf·(k1+1) / (tf + k1·lengthFactor). */
export function termScore(idfValue, tf, k1, lenFactor) {
  if (tf <= 0) return 0;
  const denom = tf + k1 * lenFactor;
  if (denom === 0) return idfValue; // k1 = 0 and tf > 0 never hits this; kept for safety
  return idfValue * (tf * (k1 + 1)) / denom;
}

/** Scores for every document (Float64Array, 0 for docs that share no terms). */
export function scoreAll(index, terms, { k1 = DEFAULT_BM25.k1, b = DEFAULT_BM25.b } = {}) {
  const scores = new Float64Array(index.N);
  for (const t of new Set(terms)) {
    const p = index.postings.get(t);
    if (!p) continue;
    const w = idf(index.N, p.docs.length);
    for (let j = 0; j < p.docs.length; j++) {
      const d = p.docs[j];
      scores[d] += termScore(w, p.tfs[j], k1, lengthFactor(index.docLen[d], index.avgdl, b));
    }
  }
  return scores;
}

/** Ranked results [{ idx, score }] for docs with score > 0. Ties keep document order. */
export function rank(index, terms, params) {
  const scores = scoreAll(index, terms, params);
  const out = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > 0) out.push({ idx: i, score: scores[i] });
  out.sort((a, b) => b.score - a.score || a.idx - b.idx);
  return out;
}

/** Per-term breakdown of one document's score, with the numbers needed to explain it. */
export function explain(index, docIdx, terms, { k1 = DEFAULT_BM25.k1, b = DEFAULT_BM25.b } = {}) {
  const dl = index.docLen[docIdx];
  const lf = lengthFactor(dl, index.avgdl, b);
  const rows = [...new Set(terms)].map((term) => {
    const df = docFreq(index, term);
    const tf = index.docTf[docIdx].get(term) || 0;
    const w = idf(index.N, df);
    return { term, tf, df, idf: w, lengthFactor: lf, contribution: termScore(w, tf, k1, lf) };
  });
  const total = rows.reduce((s, r) => s + r.contribution, 0);
  return { rows, total, docLen: dl, avgdl: index.avgdl, N: index.N, k1, b };
}
