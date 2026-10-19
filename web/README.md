# Tank Trouble AI Arena

The browser demo. Physics, Laika and the AIs run at 25 Hz in a Web Worker;
Canvas rendering and controls stay on the main thread.

- Agents: League PPO (default) and Tactical, against Laika, Hunter, Dodger or Random.
- Modes: watch, play against an AI, AI vs AI.

```bash
npm install
npm run dev          # local arena
npm test             # build and run the tests
npm run build:pages  # build for Cloudflare Pages (dist/pages)
```

Grade an agent against Laika under the evaluation protocol:

```bash
node rl/eval-agent.mjs ppo-league 970000 1000
```

Method and results: [`../docs/REPORT.md`](../docs/REPORT.md).
