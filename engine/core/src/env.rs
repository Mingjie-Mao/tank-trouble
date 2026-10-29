//! One duel round as an RL episode, paid once by the round's true result: a
//! kill followed by death to a bullet already in flight is a double KO, and a
//! round both tanks survive for 30 s is a draw.

use crate::constants as c;
use crate::game::{Event, Game};
use crate::obs::{self, candidate, Context, SeatHistory, DUEL_FRAMES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Running,
    Win,
    Loss,
    DoubleKo,
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpponentKind {
    /// The game's own AI on tank 1.
    Laika,
    /// Tank 1 is driven by actions passed to `step`.
    Policy,
}

/// Terminal reward: a win is +1; the rest are parameters.
#[derive(Clone, Copy, Debug)]
pub struct Reward {
    pub double_ko: f64,
    pub loss: f64,
    pub draw: f64,
}

impl Default for Reward {
    fn default() -> Reward {
        Reward { double_ko: -0.5, loss: -1.0, draw: -1.0 }
    }
}

impl Reward {
    pub fn of(&self, outcome: Outcome) -> f64 {
        match outcome {
            Outcome::Win => 1.0,
            Outcome::Loss => self.loss,
            Outcome::DoubleKo => self.double_ko,
            Outcome::Draw => self.draw,
            Outcome::Running => 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Episode {
    pub game: Game,
    pub opponent: OpponentKind,
    pub reward: Reward,
    pub frames: u32,
    pub outcome: Outcome,
    pub prev_pose: [[f64; 3]; 2],
    pub history: [SeatHistory; 2],
    /// Every bullet-tank hit as (owner, victim, frame).
    pub kills: Vec<(usize, usize, u32)>,
    /// Frame of each seat's most recent shot this round.
    pub last_fire: [Option<u32>; 2],
}

fn poses(g: &Game) -> [[f64; 3]; 2] {
    [[g.tanks[0].x, g.tanks[0].y, g.tanks[0].rotation], [g.tanks[1].x, g.tanks[1].y, g.tanks[1].rotation]]
}

fn write_action(g: &mut Game, tank: usize, action: usize) {
    let (throttle, turn, fire) = candidate(action);
    g.tanks[tank].set_buttons(throttle == 2, throttle == 0, turn == 0, turn == 2, fire);
}

impl Episode {
    pub fn new(seed: u64, opponent: OpponentKind, reward: Reward) -> Episode {
        let mut game = Game::new(seed, opponent == OpponentKind::Laika);
        game.events.clear();
        let prev_pose = poses(&game);
        Episode {
            game,
            opponent,
            reward,
            frames: 0,
            outcome: Outcome::Running,
            prev_pose,
            history: [SeatHistory::default(), SeatHistory::default()],
            kills: Vec::new(),
            last_fire: [None, None],
        }
    }

    /// The observation for `seat` and its bullet mask.
    pub fn observe(&self, seat: usize, out: &mut [f32], mask: &mut [u8]) {
        let ctx = Context { prev_pose: &self.prev_pose, history: &self.history[seat], frames: self.frames, last_fire: self.last_fire };
        obs::encode(&self.game, seat, &ctx, out, mask);
    }

    /// Advance one frame; returns the reward (non-zero only at the terminal).
    pub fn step(&mut self, action: usize, opponent_action: Option<usize>) -> f64 {
        let g = &mut self.game;
        self.history[0].record(action);
        if g.tanks[0].alive && !g.frozen {
            write_action(g, 0, action);
        }
        self.prev_pose = poses(g);
        if self.opponent == OpponentKind::Policy {
            if let Some(a) = opponent_action {
                self.history[1].record(a);
                if g.tanks[1].alive && !g.frozen {
                    write_action(g, 1, a);
                }
            }
        }
        g.step();
        self.frames += 1;
        let mut winner = None;
        let mut ended = false;
        for e in &self.game.events {
            match e {
                Event::Hit { owner, victim } => self.kills.push((*owner, *victim, self.frames)),
                Event::Fire(t) => self.last_fire[*t] = Some(self.frames),
                Event::RoundEnd(w) => {
                    ended = true;
                    winner = *w;
                }
                _ => {}
            }
        }
        self.outcome = if ended {
            match winner {
                Some(0) => Outcome::Win,
                None => Outcome::DoubleKo,
                _ => Outcome::Loss,
            }
        } else if (self.frames >= DUEL_FRAMES && self.game.end_count < 0)
            || self.frames >= DUEL_FRAMES + c::NUMBEROFFRAMESBEFOREEND as u32
        {
            Outcome::Draw
        } else {
            Outcome::Running
        };
        self.reward.of(self.outcome)
    }
}
