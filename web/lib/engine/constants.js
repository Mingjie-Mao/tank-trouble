/**
 * Game constants, taken from `tank_trouble_original/constants.py`, which
 * extracts them from the decompiled Flash source (frame_53 DoAction and the
 * tank sprite). Every physical quantity is expressed at SCALE = 50 and
 * multiplied by (scale / 50) at runtime, because the cell size is re-derived
 * from the maze dimensions every round.
 */

export const FPS = 25;
export const DEG = Math.PI / 180;

// ---- playfield layout (frame_53:2152-2154) ----
export const MOVIEWIDTH = 692;
export const MOVIEHEIGHT = 480;
export const HEIGHTTOBOTTOM = 80;

// ---- bullets (frame_53:2155-2212) ----
export const STARTWEAPON = "bullet";
export const BULLETSPEED = 4.5;
export const BULLETLIFETIME = 250;
export const BULLETHITCHECKINTERVALS = 7;
export const BULLETDEADLY = 0;
// Read by Laika's dodge thresholds; these weapons never spawn against Laika.
export const FRAGSPEED = 4.5;
export const GATLINGSPEED = 5.5;
export const GATLINGLIFETIME = 125;

// ---- crates (frame_53:2173-2175); the timer runs and draws from the RNG ----
export const CRATESPAWNTIMEBASE = 350;
export const CRATESPAWNTIMERANDOM = 200;
export const CRATESPAWNMAZESIZESCALE = 2000;

// ---- round lifecycle (frame_53:2197-2199) ----
export const NUMBEROFFRAMESBEFOREEND = 125;
export const NUMBEROFFRAMESFROZEN = 50;
export const NUMBEROFFRAMESBEFORERESET = 5;
// Frames after a kill in which a bullet already in flight can still turn the
// apparent winner's round into a double KO.
export const SETTLEMENT_FRAMES = NUMBEROFFRAMESBEFOREEND - NUMBEROFFRAMESFROZEN;

export const MAXSHAKE = 8;
export const MAXDEADENDPENALTY = 5;

// ---- default settings (frame_57) ----
export const SETTINGS_MAX_BULLETS = 5;
export const SETTINGS_MAX_CRATES = 3;
export const SETTINGS_CRATE_SPAWN_MODIFIER = 1;

// ---- tank physics (tank sprite :42-44, :322) ----
export const TANK_FORWARD_SPEED_BASE = 4.0;
export const TANK_BACKUP_SPEED_BASE = 2.5;
export const TANK_TURN_SPEED = 10;
export const TANK_MOVE_STEPS = 5;

// ---- tank geometry, local sprite units, from the SWF vector bounds ----
export const TANK_BASE_WIDTH = 61.0;
export const TANK_BASE_HEIGHT = 81.0;
export const TANK_TURRET_WIDTH = 45.0;
export const TANK_TURRET_HEIGHT = 77.5;
export const TANK_DISPLAY_SCALE_FACTOR = 0.55 / 100.0;
export const TANK_BOUNDS_LOCAL = [-30.5, -55.0, 30.5, 40.5];
export const TANK_BARREL_HALF_WIDTH = TANK_TURRET_WIDTH / 6.0;
export const TANK_BARREL_TIP_Y = (-TANK_TURRET_HEIGHT / 16.0) * 11.0;
// Bullet-vs-tank hit shape: hull rectangle union barrel rectangle.
export const TANK_SHAPE_BARREL_HALF_WIDTH = 8.5;
export const TANK_SHAPE_BARREL_TIP_Y = -55.0;

export const BULLET_VISUAL_RADIUS = 3.5;
