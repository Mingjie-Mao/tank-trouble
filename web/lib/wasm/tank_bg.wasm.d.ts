/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_arena_free: (a: number, b: number) => void;
export const __wbg_duel_free: (a: number, b: number) => void;
export const arena_command: (a: number, b: number, c: number) => [number, number, number, number];
export const arena_finish: (a: number, b: number, c: number, d: number) => [number, number, number, number];
export const arena_follow: (a: number, b: number, c: number) => [number, number, number];
export const arena_frame: (a: number) => [number, number];
export const arena_new: (a: number) => number;
export const arena_plan: (a: number) => [number, number];
export const arena_rollouts: (a: number, b: number, c: number) => [number, number, number, number];
export const arena_state: (a: number) => [number, number];
export const arena_step: (a: number) => [number, number];
export const duel_decide: (a: number) => [number, number];
export const duel_new: (a: number, b: number, c: number) => number;
export const duel_poses: (a: number) => [number, number];
export const duel_searched: (a: number) => number;
export const duel_step: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
