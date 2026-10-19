# Hybrid Search Playground
Access at: https://thendralmagudapathi.com/tools/hybrid-search-playground

A browser tool for AI engineers that shows how **BM25 keyword search**, **vector (embedding) search** and **hybrid fusion** rank the same documents, and explains *why* each document scored the way it did.

- Static site. No backend, no build step, no API keys. Everything runs in the visitor's browser.
- Documents you paste or open are processed locally and never leave the device.
- The only external dependency is [Transformers.js](https://huggingface.co/docs/transformers.js) v3 (pinned to `3.7.6`). It is loaded lazily from jsDelivr inside a Web Worker, and only when the user clicks **Load embedding model**.

## Run locally

```bash
npm start          # serves public/ at http://localhost:4174 (bound to 127.0.0.1)
npm test           # node --test, no dependencies
```

Node 18+ is enough. There is nothing to install.

## Deploy

Upload the `public/` folder to any static host (Cloudflare Pages, GitHub Pages, Netlify, S3…). No build command; output directory = `public`.

## Features

- **BM25** with the Lucene / Elasticsearch formula (`IDF = ln(1 + (N − n + 0.5)/(n + 0.5))`, never negative), `k1` / `b` sliders, and live text-processing toggles: lowercase, punctuation removal (identifiers like `ERR-504` and `v2.3.1` kept whole *and* split into parts), stopwords, and a hand-written Porter stemmer.
- **Score breakdown** per query term (tf, df, IDF, length factor, contribution), a stacked bar, plain-English sentences, and **"Why didn't this match?"** (missing words, vocabulary mismatch found with the embedding model, stemming/stopword/case effects, length penalty).
- **Vector search** with `Xenova/all-MiniLM-L6-v2` (default, q8, ~23 MB) or `Xenova/bge-small-en-v1.5` (q8, ~34 MB, CLS pooling + query instruction). WebGPU when available, otherwise WASM. Batched embedding with progress and Cancel; embeddings cached by model + exact text.
- **Hybrid fusion**: Reciprocal Rank Fusion (k slider) or weighted min-max score fusion (α slider), with each result's origin ("BM25 #2 + Vector #5") and "Only BM25 / vector found this" badges.
- **Evaluation**: mark relevant documents; Precision@k, Recall@k, MRR and nDCG@k for each method, for the current query and averaged over all labelled queries, plus generated insights.
- **Learn panel**: drag 2-D vectors to compare cosine, dot product and Euclidean distance, with and without normalisation.
- **Export**: Python script (Lucene BM25 + sentence-transformers + RRF/weighted), Elasticsearch / OpenSearch index settings, workspace JSON (also auto-saved to `localStorage`), and results CSV.
- Three original sample datasets with example queries and relevance labels: IT support tickets (BM25 wins on exact codes), Product FAQ (vector wins on paraphrases), Mixed technical docs (hybrid wins; doc 30 is long and shows the effect of `b`).

## Layout

```
server.js            tiny static server (port 4174, 127.0.0.1)
public/
  index.html styles.css
  app.js             UI, state, rendering, tooltips
  bm25.js            tokenizer, Porter stemmer, stopwords, BM25 index + scoring (pure)
  fusion.js          RRF, weighted fusion, min-max normalisation (pure)
  metrics.js         precision@k, recall@k, MRR, nDCG (pure)
  embed-worker.js    Web Worker running Transformers.js
  datasets.js        built-in sample datasets
tests/               node --test
```

## Notes

- `rank_bm25`'s `BM25Okapi` uses a different IDF (`ln((N − n + 0.5)/(n + 0.5))` with an epsilon floor), so its absolute scores differ from Elasticsearch and from this tool. The exported Python uses a hand-written Lucene-style BM25 instead.
- Elasticsearch's `standard` tokenizer splits `ERR-504` into `err` + `504` but does not also keep `err-504` whole, so scores for such identifiers can differ slightly from the playground.
- The browser runs 8-bit quantised models, so cosine scores differ slightly from full-precision `sentence-transformers`.
