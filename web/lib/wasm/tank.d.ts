/* tslint:disable */
/* eslint-disable */

/**
 * The page's whole match: two drivers, scores, streaks and telemetry.
 */
export class Arena {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply a UI command (the BrowserArena protocol) and return the snapshot.
     */
    command(payload: string): string;
    /**
     * Conclude `plan`'s searches from their values (one candidate-major
     * array per search, null where unplayed), step, and return the snapshot.
     * `ms` is how long the searches took.
     */
    finish(values: string, ms: number): string;
    /**
     * As a copy: take the frame the real arena just stepped; false if the
     * two have drifted apart.
     */
    follow(frame: string): boolean;
    /**
     * The last frame's moves, for copies kept in step (`follow`).
     */
    frame(): string;
    constructor(seed: number);
    /**
     * `step` in three parts, so searches can run on copies in other
     * workers: decide every seat and return the searches still to play
     * out, as JSON [{seat, sandbox_seed, candidates, seeds}]; play their
     * rollouts (`rollouts`, here or on a copy); then `finish`.
     */
    plan(): string;
    /**
     * Play a planned search's `jobs` ([candidate, seed index] pairs) within
     * `budget_ms`; the values in job order, null where unplayed.
     */
    rollouts(request: string): string;
    /**
     * The snapshot as JSON.
     */
    state(): string;
    /**
     * Advance one frame and return the snapshot as JSON.
     */
    step(): string;
}

/**
 * A lockstep copy for matches against controllers that live outside Rust:
 * our controller drives `seat`, the caller presses the other tank's buttons
 * and checks that both engines still agree.
 */
export class Duel {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Our action for this frame, as [throttle, turn, fire].
     */
    decide(): Uint8Array;
    constructor(seed: number, seat: number, search: boolean);
    /**
     * [x, y, rotation] of both tanks, for drift checks.
     */
    poses(): Float64Array;
    searched(): boolean;
    /**
     * Step with the other tank's buttons; returns the events, comma-joined per line.
     */
    step(forward: boolean, backup: boolean, left: boolean, right: boolean, fire: boolean): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_arena_free: (a: number, b: number) => void;
    readonly __wbg_duel_free: (a: number, b: number) => void;
    readonly arena_command: (a: number, b: number, c: number) => [number, number, number, number];
    readonly arena_finish: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly arena_follow: (a: number, b: number, c: number) => [number, number, number];
    readonly arena_frame: (a: number) => [number, number];
    readonly arena_new: (a: number) => number;
    readonly arena_plan: (a: number) => [number, number];
    readonly arena_rollouts: (a: number, b: number, c: number) => [number, number, number, number];
    readonly arena_state: (a: number) => [number, number];
    readonly arena_step: (a: number) => [number, number];
    readonly duel_decide: (a: number) => [number, number];
    readonly duel_new: (a: number, b: number, c: number) => number;
    readonly duel_poses: (a: number) => [number, number];
    readonly duel_searched: (a: number) => number;
    readonly duel_step: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
