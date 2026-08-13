# Tank Trouble AI Arena

Large-screen browser arena for the Tank Trouble controllers in this repository.

## Browser product path

- `Tank Trouble Tactical` is the default: frozen KillField H36 attack search
  plus verified two-stage visible-bullet evasion and settlement survival.
- Browser-native physics, Laika and search run at 25 Hz in a dedicated Web
  Worker; Canvas rendering and controls stay on the main thread.
- `KillField JS` remains selectable as an attributed MIT comparison baseline.
- Watch, human-vs-AI and AI-vs-AI modes run without Python.

The Python runtime switch preserves the separate P27b / Hybrid / Exact Shield
research stack and requires the local bridge.

## Local commands

```bash
npm install
npm run dev
npm test
```

Useful evaluation commands:

```bash
npm run benchmark:runtime -- 5000 watch p27-js-tactical
npm run compare:runtime -- 990000 100 p27-js-tactical
```

From the repository root, `./run_js_arena.sh` starts the browser-only arena;
`./run_web_arena.sh` also starts the Python bridge.

The current result and comparison protocol are documented in
`../docs/TACTICAL_VS_KILLFIELD_2026-08-12.md`.
