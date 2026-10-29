//! Game constants from `tank_trouble_original/constants.py` (decompiled Flash
//! source). Physical quantities are given at SCALE = 50 and multiplied by
//! scale / 50 at runtime: the cell size is re-derived from every new maze.

pub const FPS: u32 = 25;
pub const DEG: f64 = std::f64::consts::PI / 180.0;

pub const MOVIEWIDTH: f64 = 692.0;
pub const MOVIEHEIGHT: f64 = 480.0;

pub const BULLETSPEED: f64 = 4.5;
pub const BULLETLIFETIME: i32 = 250;
pub const BULLETHITCHECKINTERVALS: i32 = 7;
pub const BULLETDEADLY: i32 = 0;
/// Read by Laika's dodge thresholds; these weapons never spawn against Laika.
pub const FRAGSPEED: f64 = 4.5;
pub const GATLINGSPEED: f64 = 5.5;
pub const GATLINGLIFETIME: i32 = 125;

pub const CRATESPAWNTIMEBASE: f64 = 350.0;
pub const CRATESPAWNTIMERANDOM: u32 = 200;
pub const CRATESPAWNMAZESIZESCALE: f64 = 2000.0;

pub const NUMBEROFFRAMESBEFOREEND: i32 = 125;
pub const NUMBEROFFRAMESFROZEN: i32 = 50;
pub const NUMBEROFFRAMESBEFORERESET: i32 = 5;

pub const MAXSHAKE: f64 = 8.0;
pub const MAXDEADENDPENALTY: i32 = 5;

pub const SETTINGS_MAX_BULLETS: i32 = 5;
pub const SETTINGS_CRATE_SPAWN_MODIFIER: f64 = 1.0;

pub const TANK_FORWARD_SPEED_BASE: f64 = 4.0;
pub const TANK_BACKUP_SPEED_BASE: f64 = 2.5;
pub const TANK_TURN_SPEED: f64 = 10.0;
pub const TANK_MOVE_STEPS: i32 = 5;

pub const TANK_BASE_WIDTH: f64 = 61.0;
pub const TANK_BASE_HEIGHT: f64 = 81.0;
pub const TANK_TURRET_WIDTH: f64 = 45.0;
pub const TANK_TURRET_HEIGHT: f64 = 77.5;
pub const TANK_DISPLAY_SCALE_FACTOR: f64 = 0.55 / 100.0;
pub const TANK_BOUNDS_LOCAL: [f64; 4] = [-30.5, -55.0, 30.5, 40.5];
pub const TANK_BARREL_HALF_WIDTH: f64 = TANK_TURRET_WIDTH / 6.0;
pub const TANK_BARREL_TIP_Y: f64 = (-TANK_TURRET_HEIGHT / 16.0) * 11.0;
/// Bullet-vs-tank hit shape: hull rectangle union barrel rectangle; the barrel
/// is x in [-7.5, 7.5], y in [-55, -21.25] (SWF shapes 80/82, turret sprite 134).
pub const TANK_SHAPE_BARREL_HALF_WIDTH: f64 = 7.5;
pub const TANK_SHAPE_BARREL_TIP_Y: f64 = -55.0;
