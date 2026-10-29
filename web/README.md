# Tank Trouble AI Arena

The browser demo. The Rust engine, Laika and our AI (`engine/wasm`, built into
`lib/wasm/`) run at 25 Hz in a Web Worker; Canvas rendering and controls stay
on the main thread. Up to seven helper workers (`workers/search-helper.worker.js`)
keep copies of the arena in step and share the search's rollouts
(`lib/search-pool.js`): with seven or more workers in all the search is
4 candidates × 4 rollouts, otherwise 2 × 2; a copy that drifts drops the
search back to one worker.

- Players: our AI (network + real-time search; against Laika it tracks Laika's hidden state), Laika, Random, or you.
- Modes: watch against Laika, play against an AI, AI vs AI.

```bash
npm install
npm run dev          # local arena
npm test             # build and run the tests
npm run build:pages  # build for Cloudflare Pages (dist/pages)
```

Rebuild the WebAssembly after changing `engine/`: `npm run build:wasm` (needs
`wasm-pack`).

Method and results: [`../docs/REPORT.md`](../docs/REPORT.md).
