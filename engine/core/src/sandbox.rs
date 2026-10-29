//! A copy of the game to roll forward without touching the real one.
//!
//! The copy is seat-relative — tank 0 is the seat asking — and fair: the
//! random stream is replaced by an independent one, and the opponent either
//! holds the buttons it is pressing now (`Opponent::Hold`) or is a Laika
//! rebuilt from scratch (`Opponent::FreshLaika`), so no goal stack, queued
//! action or future coin flip of the real game leaks into the rollout.

use crate::game::Game;
use crate::laika::Laika;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opponent {
    Hold,
    FreshLaika,
}

/// Swap the two seats so tank 0 is `seat`; bullet owners follow their tanks.
fn mirror(g: &mut Game) {
    g.tanks.swap(0, 1);
    g.tank_fields.swap(0, 1);
    g.scores.swap(0, 1);
    for (i, t) in g.tanks.iter_mut().enumerate() {
        t.number = i;
    }
    for b in &mut g.bullets {
        b.owner = 1 - b.owner;
    }
}

pub fn sandbox(game: &Game, seat: usize, opponent: Opponent, rng_seed: u64) -> Game {
    let mut g = game.clone();
    if seat == 1 {
        mirror(&mut g);
    }
    g.rng = Rng::new(rng_seed);
    g.laika_on_tank1 = false;
    g.laika_on_tank0 = false;
    g.events.clear();
    g.hit_tests.clear();
    g.ai = vec![None, None];
    if opponent == Opponent::FreshLaika {
        g.ai[1] = Some(Box::new(Laika::new(g.round.scale)));
    }
    g
}
