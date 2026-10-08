import {
  actionIndex, CANDIDATES, densityRollout, LIVE_ACTION_INDICES,
  maskMovingFireScores, NO_EFFECT_REPEAT_PENALTY, ROLLOUT_PLANS,
  STATIONARY_FIRE_ACTION,
} from "./killfield-runtime/src/killfield/score.js";
import { inferLearnedSearch } from "./learned-search-inference.js";
import {
  SEARCH_ACTIONS, searchActionClass, searchPriorFeatures,
} from "./learned-search-features.js";
import { TacticalCandidateAgent } from "./tactical-candidate-agent.js";

const NEUTRAL_CLASS = searchActionClass([1, 1, 0]);
const FIRE_CLASS = searchActionClass(STATIONARY_FIRE_ACTION);

// Precomputed so the per-frame loop never runs findIndex over the action table.
const PLAN_CLASSES = ROLLOUT_PLANS.map((plan) => ({
  first: searchActionClass(plan.firstAction),
  continuation: plan.continuationAction === null
    ? -1 : searchActionClass(plan.continuationAction),
}));

/**
 * The champion with a learned first-action prior in front of the rollout. The
 * network only narrows the plan list; exact physics still scores every retained
 * plan and remains the final decision maker. The learned gate is deliberately
 * shadow-only until it earns safe coverage.
 *
 * Pruning is a pure function of the visible state, so unlike the wall-clock
 * `evasionBudgetMs` narrowing this stays deterministic and can be evaluated on
 * paired seeds.
 */
export class LearnedPriorTacticalAgent extends TacticalCandidateAgent {
  constructor(options = {}) {
    super(options);
    this.priorTopK = Math.max(1, Math.min(10, Number(options.priorTopK ?? 7)));
    this.priorCalls = 0;
    this.priorCandidates = 0;
    this.priorRollouts = 0;
    this.priorGateSuggestions = 0;
    this.lastPriorCandidateCount = 0;
  }

  /** Top-K classes by prior, plus the invariants that are never pruned. */
  selectedClasses(prior) {
    const ranking = prior.map((probability, actionClass) => ({
      probability, actionClass,
    })).sort((left, right) => right.probability - left.probability);
    const selected = new Set(
      ranking.slice(0, this.priorTopK).map((entry) => entry.actionClass),
    );
    // Cheap safety/continuity invariants are never allowed to be pruned.
    selected.add(NEUTRAL_CLASS);
    selected.add(FIRE_CLASS);
    selected.add(searchActionClass(this.lastMotionAction ?? [1, 1, 0]));
    selected.delete(-1);
    return selected;
  }

  /** `{prior, gate}` over SEARCH_ACTIONS classes. Subclasses swap the network. */
  priorFor(game) {
    return inferLearnedSearch(searchPriorFeatures(game, this));
  }

  scores(game) {
    const field = this.ensureField(game);
    const prediction = this.priorFor(game);
    const selected = this.selectedClasses(prediction.prior);

    const seed = this.rng.randrange(1 << 30);
    const values = new Float64Array(CANDIDATES.length);
    values.fill(-1e9);
    let rollouts = 0;
    this.bestFireContinuation = null;

    if (this.fireContinuation) {
      // K1 collapses eighteen plans onto ten first actions: nine of them share
      // the stationary-fire first action and differ only in what follows. So
      // pruning by first action alone would retain all nine fire plans and cut
      // at most one rollout. The retained motion classes therefore prune the
      // continuation column as well as the standalone move plans.
      const me = game.tanks[0];
      const canFire = me.triggerReleased && game.weaponReady(me);
      const fireSelected = canFire && selected.has(FIRE_CLASS);
      for (let index = 0; index < ROLLOUT_PLANS.length; index += 1) {
        const plan = ROLLOUT_PLANS[index];
        const classes = PLAN_CLASSES[index];
        if (plan.kind === "fire_then_move") {
          if (!fireSelected) continue;
          if (!selected.has(classes.continuation)) continue;
        } else if (!selected.has(classes.first)) continue;
        const value = densityRollout(game, plan.firstAction, field, seed, {
          boxes: this.boxes,
          chainState: this.chain,
          horizon: this.horizon,
          hold: this.hold,
          oppModel: this.oppModel,
          continuationAction: plan.continuationAction,
        });
        rollouts += 1;
        const slot = actionIndex(plan.firstAction);
        if (value > values[slot]) {
          values[slot] = value;
          if (plan.firstAction === STATIONARY_FIRE_ACTION) {
            this.bestFireContinuation = plan.continuationAction;
          }
        }
      }
    } else {
      for (const actionClass of selected) {
        const index = LIVE_ACTION_INDICES[actionClass];
        values[index] = densityRollout(game, SEARCH_ACTIONS[actionClass], field, seed, {
          boxes: this.boxes,
          chainState: this.chain,
          horizon: this.horizon,
          hold: this.hold,
          oppModel: this.oppModel,
        });
        rollouts += 1;
      }
    }

    maskMovingFireScores(values);
    if (this.actionNoEffect && this.observedPreviousAction !== null) {
      const failed = this.observedPreviousAction;
      for (let i = 0; i < CANDIDATES.length; i += 1) {
        if (CANDIDATES[i][0] === failed[0] && CANDIDATES[i][1] === failed[1]) {
          values[i] -= NO_EFFECT_REPEAT_PENALTY;
        }
      }
    }

    this.priorCalls += 1;
    this.priorCandidates += selected.size;
    this.priorRollouts += rollouts;
    this.lastPriorCandidateCount = selected.size;
    if (prediction.gate >= 1) this.priorGateSuggestions += 1;
    return values;
  }

  telemetry() {
    return {
      ...super.telemetry(),
      priorTopK: this.priorTopK,
      priorCalls: this.priorCalls,
      priorCandidates: this.priorCandidates,
      meanPriorCandidates: this.priorCandidates / Math.max(1, this.priorCalls),
      priorRollouts: this.priorRollouts,
      meanPriorRollouts: this.priorRollouts / Math.max(1, this.priorCalls),
      priorGateSuggestions: this.priorGateSuggestions,
    };
  }
}
