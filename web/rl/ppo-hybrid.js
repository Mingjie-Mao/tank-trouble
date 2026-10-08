/**
 * Two ways to combine the league-PPO network with search.
 *
 * `PpoPriorTacticalAgent` — Tactical decides, PPO prunes. The network's
 * action distribution replaces the small MLP prior in front of Tactical's
 * rollouts; exact physics with the L2 Laika model still scores every
 * retained plan and makes the final call.
 *
 * `PpoDeepShieldAgent` — PPO decides, search vetoes. The policy's sampled
 * action is rolled forward against the scripted opponent (L2), so new shots
 * Laika would take are modelled, not just bullets already in flight. Only
 * when that rollout dies is the action replaced, by the most likely
 * alternative among the policy's top choices that survives.
 */

import { applyAction, makeSandbox } from "../lib/killfield-runtime/src/killfield/sandbox.js";
import { CANDIDATES } from "../lib/killfield-runtime/src/killfield/score.js";
import { LearnedPriorTacticalAgent } from "../lib/learned-prior-tactical-agent.js";
import { SEARCH_ACTIONS } from "../lib/learned-search-features.js";
import { PpoLeagueAgent } from "./ppo-agent.js";

/** Tank 0's buttons as a Discrete(18) index. */
function foldButtons(tank) {
  const throttle = tank.forward ? 2 : tank.backup ? 0 : 1;
  const turn = tank.turnLeft ? 0 : tank.turnRight ? 2 : 1;
  return 6 * throttle + 2 * turn + (tank.fire ? 1 : 0);
}

export class PpoPriorTacticalAgent extends LearnedPriorTacticalAgent {
  constructor(options = {}) {
    super(options);
    // The policy is consulted as a prior, so its own shield stays off:
    // Tactical's rollouts are the safety authority here.
    this.ppo = new PpoLeagueAgent({ seed: options.seed ?? 0, shield: false });
  }

  priorFor(game) {
    const probs = this.ppo.policy(game);
    if (!probs) return { prior: SEARCH_ACTIONS.map(() => 1), gate: 0 };
    // Tactical's ten classes: nine no-fire moves plus stationary fire. Every
    // fire action's mass goes to the fire class; Tactical never fires moving.
    let fire = 0;
    for (let a = 1; a < probs.length; a += 2) fire += probs[a];
    const prior = SEARCH_ACTIONS.map(([throttle, turn, f]) => (
      f ? fire : probs[6 * throttle + 2 * turn]));
    return { prior, gate: 0 };
  }

  drive(game) {
    if (this.ppo.round !== game.roundNumber) this.ppo._startRound(game);
    super.drive(game);
    // Keep the network's history equal to what the tank actually did.
    this.ppo.commit(game, foldButtons(game.tanks[0]), false);
  }
}

export class PpoDeepShieldAgent extends PpoLeagueAgent {
  constructor({ horizon = 24, alternatives = 4, vetoWithin = null, ...options } = {}) {
    super(options);
    this.horizon = horizon;
    // Only a death inside the first `vetoWithin` frames triggers a veto; a
    // later one is left for the policy to react to. Holding one action for
    // the whole horizon is pessimistic, and vetoing every distant death
    // trades losses for stalls.
    this.vetoWithin = vetoWithin ?? horizon;
    this.alternatives = alternatives;
    this.vetoes = 0;
  }

  /** Hold `action` for the horizon against L2 Laika; fire only on the first frame. */
  survival(game, action, seed) {
    const sb = makeSandbox(game, "L2", seed);
    applyAction(sb, CANDIDATES[action]);
    const me = sb.tanks[0];
    for (let frame = 0; frame < this.horizon; frame++) {
      if (frame === 1) me.fire = false;
      sb.step();
      if (!me.alive) return frame;
      if (sb.frozen) break;
    }
    return this.horizon;
  }

  choose(game, probs) {
    const action = this.sample(probs);
    if (!game.tanks[0].alive) return action;
    const seed = this.rng.randrange(1 << 30);
    const lasted = this.survival(game, action, seed);
    if (lasted >= this.vetoWithin) return action;
    const ranked = Array.from(probs.keys())
      .filter((a) => a !== action && probs[a] > 0)
      .sort((x, y) => probs[y] - probs[x])
      .slice(0, this.alternatives);
    let best = action;
    let bestLasted = lasted;
    for (const alternative of ranked) {
      const value = this.survival(game, alternative, seed);
      if (value >= this.horizon) {
        this.vetoes += 1;
        return alternative;
      }
      if (value > bestLasted) {
        best = alternative;
        bestLasted = value;
      }
    }
    if (best !== action) this.vetoes += 1;
    return best;
  }

  telemetry() {
    return { ...super.telemetry(), deepShieldVetoes: this.vetoes };
  }
}
