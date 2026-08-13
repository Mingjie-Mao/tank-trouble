import {
  actionIndex, argmax, CANDIDATES,
} from "./killfield-runtime/src/killfield/score.js";
import { searchActionClass, searchFeatures } from "./learned-search-features.js";
import { TacticalV2Agent } from "./tactical-v2-agent.js";

export class SearchTeacherRecorder extends TacticalV2Agent {
  constructor(options = {}) {
    super(options);
    this.searchSamples = [];
  }

  scores(game) {
    const values = super.scores(game);
    const teacherIndex = argmax(values);
    const teacherAction = CANDIDATES[teacherIndex];
    const actionClass = searchActionClass(teacherAction);
    const fallbackClass = searchActionClass(this.lastMotionAction ?? [1, 1, 0]);
    if (actionClass >= 0 && game.bullets.length === 0) {
      const fallbackAction = this.lastMotionAction ?? [1, 1, 0];
      const fallbackScore = values[actionIndex(fallbackAction)];
      const teacherScore = values[teacherIndex];
      const teacherRegret = Math.max(0, teacherScore - fallbackScore);
      const ranked = Array.from(values)
        .filter(Number.isFinite)
        .sort((left, right) => right - left);
      this.searchSamples.push({
        features: searchFeatures(game, this),
        actionClass,
        fallbackClass,
        teacherScore,
        fallbackScore,
        teacherRegret,
        skipTarget: teacherRegret <= 5 ? 1 : 0,
        teacherMargin: (ranked[0] ?? 0) - (ranked[1] ?? 0),
        frame: game.frame,
        round: game.roundNumber,
      });
    }
    return values;
  }
}
