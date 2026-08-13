import {
  CANDIDATES, densityRollout, LIVE_ACTION_INDICES,
  maskMovingFireScores, NO_EFFECT_REPEAT_PENALTY,
} from "./killfield-runtime/src/killfield/score.js";
import { inferLearnedSearch } from "./learned-search-inference.js";
import {
  SEARCH_ACTIONS, searchActionClass, searchPriorFeatures,
} from "./learned-search-features.js";
import { TacticalV2Agent } from "./tactical-v2-agent.js";

const NEUTRAL_CLASS = searchActionClass([1, 1, 0]);
const FIRE_CLASS = searchActionClass([1, 1, 1]);

/**
 * Hidden candidate: the network only narrows the action list. Exact physics
 * still scores every retained action and remains the final decision maker.
 * The learned gate is deliberately shadow-only until it earns safe coverage.
 */
export class LearnedPriorTacticalAgent extends TacticalV2Agent {
  constructor(options = {}) {
    super(options);
    this.priorTopK = Math.max(1, Math.min(10, Number(options.priorTopK ?? 7)));
    this.priorCalls = 0;
    this.priorCandidates = 0;
    this.priorGateSuggestions = 0;
    this.lastPriorCandidateCount = 0;
  }

  scores(game) {
    const field = this.ensureField(game);
    const prediction = inferLearnedSearch(searchPriorFeatures(game, this));
    const ranking = prediction.prior.map((probability, actionClass) => ({
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

    const seed = this.rng.randrange(1 << 30);
    const values = new Float64Array(CANDIDATES.length);
    values.fill(-1e9);
    for (const actionClass of selected) {
      const index = LIVE_ACTION_INDICES[actionClass];
      values[index] = densityRollout(game, SEARCH_ACTIONS[actionClass], field, seed, {
        boxes: this.boxes,
        chainState: this.chain,
        horizon: this.horizon,
        hold: this.hold,
        oppModel: this.oppModel,
      });
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
      priorGateSuggestions: this.priorGateSuggestions,
    };
  }
}
