export const DODGER_TIMEOUT_REGRESSION = Object.freeze([
  Object.freeze({ seed: 1300000, candidateSide: 1 }),
  Object.freeze({ seed: 1300001, candidateSide: 0 }),
  Object.freeze({ seed: 1300001, candidateSide: 1 }),
  Object.freeze({ seed: 1300002, candidateSide: 0 }),
  Object.freeze({ seed: 1300002, candidateSide: 1 }),
]);

export const DODGER_INTERCEPT_REGRESSION = Object.freeze([
  Object.freeze({ seed: 1400004, candidateSide: 0 }),
  Object.freeze({ seed: 1400004, candidateSide: 1, expected: "non_loss" }),
]);
