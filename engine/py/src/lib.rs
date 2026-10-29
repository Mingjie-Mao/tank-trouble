//! `tank_rl.VecEnv`: many duel rounds stepped in parallel, observations
//! written straight into NumPy arrays. A round that ends is replaced by a new
//! one on the next seed; its outcome is reported on the step that ended it.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use tank_core::belief::Belief;
use tank_core::policy::Policy;
use tank_core::rng::Rng;
use tank_core::teacher::{self, TeacherConfig};
use pyo3::prelude::*;
use rayon::prelude::*;
use tank_core::env::{Episode, OpponentKind, Outcome, Reward};
use tank_core::obs;

struct Slot {
    episode: Episode,
    /// Seeds still to play when grading a fixed list; None in training.
    queue: Option<std::vec::IntoIter<u64>>,
    next_seed: u64,
    stride: u64,
    coin: u64,
    done: bool,
}

fn outcome_code(o: Outcome) -> i8 {
    match o {
        Outcome::Running => 0,
        Outcome::Win => 1,
        Outcome::Loss => 2,
        Outcome::DoubleKo => 3,
        Outcome::Draw => 4,
    }
}

impl Slot {
    fn draw_opponent(&mut self, laika_share: f64) -> OpponentKind {
        // xorshift64*: a per-slot stream, reproducible from the base seed.
        self.coin ^= self.coin >> 12;
        self.coin ^= self.coin << 25;
        self.coin ^= self.coin >> 27;
        let u = (self.coin.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64;
        if u < laika_share { OpponentKind::Laika } else { OpponentKind::Policy }
    }
}

#[pyclass]
struct VecEnv {
    slots: Vec<Slot>,
    laika_share: f64,
    reward: Reward,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl VecEnv {
    /// `n` rounds in parallel. Training: slot i plays seeds seed + i, seed + i + n, ...
    /// with Laika as the opponent with probability `laika_share`, else a policy
    /// driven through `step`. Grading: pass `seeds`; slots play them once each
    /// against Laika and then stay done.
    #[new]
    #[pyo3(signature = (n, seed, laika_share=1.0, threads=8, seeds=None, double_ko=-0.5, draw=-1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, seed: u64, laika_share: f64, threads: usize, seeds: Option<Vec<u64>>,
           double_ko: f64, draw: f64) -> PyResult<Self> {
        let reward = Reward { double_ko, draw, ..Reward::default() };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut slots = Vec::with_capacity(n);
        for i in 0..n {
            let (queue, first, opponent_share) = match &seeds {
                Some(list) => {
                    let mine: Vec<u64> = list.iter().skip(i).step_by(n).copied().collect();
                    let mut it = mine.into_iter();
                    let first = it.next();
                    (Some(it), first, 1.0)
                }
                None => (None, Some(seed + i as u64), laika_share),
            };
            let mut slot = Slot {
                episode: Episode::new(first.unwrap_or(0), OpponentKind::Laika, reward),
                queue,
                next_seed: seed + (i + n) as u64,
                stride: n as u64,
                coin: (seed ^ 0x9e37_79b9_7f4a_7c15).wrapping_add(i as u64 * 0x632b_e59b_d9b4_e019) | 1,
                done: first.is_none(),
            };
            let kind = slot.draw_opponent(opponent_share);
            slot.episode = Episode::new(first.unwrap_or(0), kind, reward);
            slots.push(slot);
        }
        Ok(VecEnv { slots, laika_share: if seeds.is_some() { 1.0 } else { laika_share }, reward, pool })
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        obs::OBS_DIM
    }

    /// Where the network's input branches live in the observation:
    /// (map_start, map_len, bullet_start, bullet_slots, bullet_dim,
    ///  dodge_start, self_hit_index). Everything outside map and bullets is scalar.
    #[getter]
    fn layout(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let l = obs::LAYOUT;
        (l[0], l[1], l[2], l[3], l[4], l[5], l[6])
    }

    /// Per slot: 1 if tank 1 is driven by `step`'s opponent actions, 0 for Laika.
    fn policy_slots<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let v: Vec<u8> = self.slots.iter().map(|s| (s.episode.opponent == OpponentKind::Policy) as u8).collect();
        PyArray1::from_vec(py, v)
    }

    /// Observations of seat 0 for every slot and of seat 1 for policy slots:
    /// (obs0, mask0, obs1, mask1).
    #[allow(clippy::type_complexity)]
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u8>>,
                                                         Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u8>>)> {
        let (n, d, b) = (self.slots.len(), obs::OBS_DIM, obs::BULLET_SLOTS);
        let obs0 = PyArray2::<f32>::zeros(py, [n, d], false);
        let mask0 = PyArray2::<u8>::zeros(py, [n, b], false);
        let obs1 = PyArray2::<f32>::zeros(py, [n, d], false);
        let mask1 = PyArray2::<u8>::zeros(py, [n, b], false);
        {
            let mut w0 = obs0.readwrite();
            let mut m0 = mask0.readwrite();
            let mut w1 = obs1.readwrite();
            let mut m1 = mask1.readwrite();
            let (o0, k0, o1, k1) = (w0.as_slice_mut()?, m0.as_slice_mut()?, w1.as_slice_mut()?, m1.as_slice_mut()?);
            let slots = &self.slots;
            let pool = &self.pool;
            py.allow_threads(|| {
                pool.install(|| {
                    o0.par_chunks_mut(d).zip(k0.par_chunks_mut(b)).zip(o1.par_chunks_mut(d)).zip(k1.par_chunks_mut(b))
                        .zip(slots.par_iter())
                        .for_each(|((((o0, k0), o1), k1), slot)| {
                            if slot.done {
                                return;
                            }
                            let ep = &slot.episode;
                            ep.observe(0, o0, k0);
                            if ep.opponent == OpponentKind::Policy {
                                ep.observe(1, o1, k1);
                            }
                        });
                })
            });
        }
        Ok((obs0, mask0, obs1, mask1))
    }

    /// Step every live slot. Returns (reward, outcome, frames, seed): outcome
    /// is 0 while running, else 1 win / 2 loss / 3 double KO / 4 draw for the
    /// round that ended on this step (the slot already holds the next round).
    #[allow(clippy::type_complexity)]
    fn step<'py>(&mut self, py: Python<'py>, actions: PyReadonlyArray1<'py, i64>, opponent_actions: PyReadonlyArray1<'py, i64>)
        -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
        let n = self.slots.len();
        let actions = actions.as_slice()?.to_vec();
        let opp = opponent_actions.as_slice()?.to_vec();
        if actions.len() != n || opp.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err("one action per slot"));
        }
        let mut reward = vec![0f32; n];
        let mut outcome = vec![0i8; n];
        let mut frames = vec![0u32; n];
        let mut seeds = vec![0u64; n];
        let (share, rw) = (self.laika_share, self.reward);
        let slots = &mut self.slots;
        let pool = &self.pool;
        py.allow_threads(|| {
            pool.install(|| {
                slots.par_iter_mut().zip(actions.par_iter()).zip(opp.par_iter())
                    .zip(reward.par_iter_mut()).zip(outcome.par_iter_mut()).zip(frames.par_iter_mut()).zip(seeds.par_iter_mut())
                    .for_each(|((((((slot, &a), &o), r), out), fr), sd)| {
                        if slot.done {
                            return;
                        }
                        let ep = &mut slot.episode;
                        let opponent_action = (ep.opponent == OpponentKind::Policy).then_some(o as usize);
                        *r = ep.step(a as usize, opponent_action) as f32;
                        if ep.outcome == Outcome::Running {
                            return;
                        }
                        *out = outcome_code(ep.outcome);
                        *fr = ep.frames;
                        *sd = ep.game.seed;
                        let next = match &mut slot.queue {
                            Some(q) => q.next(),
                            None => {
                                let s = slot.next_seed;
                                slot.next_seed += slot.stride;
                                Some(s)
                            }
                        };
                        match next {
                            Some(s) => {
                                let kind = if slot.queue.is_some() { OpponentKind::Laika } else { slot.draw_opponent(share) };
                                slot.episode = Episode::new(s, kind, rw);
                            }
                            None => slot.done = true,
                        }
                    });
            })
        });
        Ok((PyArray1::from_vec(py, reward), PyArray1::from_vec(py, outcome), PyArray1::from_vec(py, frames), PyArray1::from_vec(py, seeds)))
    }

    /// True once every slot has played its last graded seed.
    fn all_done(&self) -> bool {
        self.slots.iter().all(|s| s.done)
    }
}

fn load_policy(path: &str) -> PyResult<Policy> {
    let bytes = std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let policy = Policy::from_bytes(&bytes).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if policy.obs_dim != obs::OBS_DIM || policy.layout != obs::LAYOUT {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("{path} reads another observation")));
    }
    Ok(policy)
}

struct Labels {
    obs: Vec<f32>,
    mask: Vec<u8>,
    probs: Vec<f32>,
    scores: Vec<f32>,
    danger: Vec<u8>,
    outcome: i8,
}

/// One student round against Laika: the student's own sampled actions are
/// played; danger frames get teacher scores, a `calm` share of other frames
/// is kept unscored.
fn label_round(policy: &Policy, seed: u64, cfg: &TeacherConfig, calm: f64) -> Labels {
    let mut ep = Episode::new(seed, OpponentKind::Laika, Reward::default());
    let mut rng = Rng::new(seed.wrapping_mul(0x9e37_79b9) ^ 0x5151);
    let mut obs = vec![0f32; policy.obs_dim];
    let mut mask = vec![0u8; policy.layout[3]];
    let mut out = Labels { obs: Vec::new(), mask: Vec::new(), probs: Vec::new(), scores: Vec::new(), danger: Vec::new(), outcome: 0 };
    while ep.outcome == Outcome::Running {
        let mut action = 0;
        if !ep.game.frozen {
            ep.observe(0, &mut obs, &mut mask);
            let probs = policy.shielded_probs(&obs, &mask);
            action = teacher::sample(&probs, &mut rng);
            let danger = teacher::danger(&ep.game, policy, &obs);
            if danger || rng.random() < calm {
                let scores = if danger { teacher::scores(&ep, policy, &probs, action, cfg, &mut rng, &[]) } else { [f32::NAN; 18] };
                out.obs.extend_from_slice(&obs);
                out.mask.extend_from_slice(&mask);
                out.probs.extend_from_slice(&probs);
                out.scores.extend_from_slice(&scores);
                out.danger.push(danger as u8);
            }
        }
        ep.step(action, None);
    }
    out.outcome = outcome_code(ep.outcome);
    out
}

/// Teacher labels for distillation over `seeds` (one round each), in parallel.
/// Returns (obs, mask, probs, scores, danger, outcomes).
#[pyfunction]
#[pyo3(signature = (policy_path, seeds, candidates=4, rollouts=4, horizon=24, calm=0.25, threads=9))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn teacher_labels<'py>(py: Python<'py>, policy_path: &str, seeds: Vec<u64>, candidates: usize, rollouts: usize,
                       horizon: usize, calm: f64, threads: usize)
    -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>,
                 Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i8>>)> {
    let policy = load_policy(policy_path)?;
    let cfg = TeacherConfig { candidates, rollouts, horizon, ..TeacherConfig::default() };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let rounds: Vec<Labels> = py.allow_threads(|| pool.install(|| seeds.par_iter().map(|&s| label_round(&policy, s, &cfg, calm)).collect()));
    let (d, b) = (policy.obs_dim, policy.layout[3]);
    let n: usize = rounds.iter().map(|r| r.danger.len()).sum();
    let cat = |f: &dyn Fn(&Labels) -> Vec<f32>| rounds.iter().flat_map(f).collect::<Vec<f32>>();
    let obs = cat(&|r: &Labels| r.obs.clone());
    let probs = cat(&|r: &Labels| r.probs.clone());
    let scores = cat(&|r: &Labels| r.scores.clone());
    let mask: Vec<u8> = rounds.iter().flat_map(|r| r.mask.clone()).collect();
    let danger: Vec<u8> = rounds.iter().flat_map(|r| r.danger.clone()).collect();
    let outcomes: Vec<i8> = rounds.iter().map(|r| r.outcome).collect();
    let shape2 = |v: Vec<f32>, w: usize| PyArray1::from_vec(py, v).reshape([n, w]);
    Ok((shape2(obs, d)?, PyArray1::from_vec(py, mask).reshape([n, b])?, shape2(probs, 18)?, shape2(scores, 18)?,
        PyArray1::from_vec(py, danger), PyArray1::from_vec(py, outcomes)))
}

/// Grade a policy against Laika, one round per seed, entirely in Rust. With
/// `mpc`, danger frames are decided by the teacher's best-scoring action
/// (the policy's own choice kept unless another is strictly better); with
/// `particles`, the search plays against copies of Laika kept in step with
/// the real one's visible moves instead of a fresh one.
/// Returns (outcome codes, every decision's milliseconds, negated when no search ran).
#[pyfunction]
#[pyo3(signature = (policy_path, seeds, mpc=false, candidates=2, rollouts=2, horizon=24, threads=9, particles=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn play<'py>(py: Python<'py>, policy_path: &str, seeds: Vec<u64>, mpc: bool, candidates: usize, rollouts: usize,
             horizon: usize, threads: usize, particles: usize) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<f32>>)> {
    let policy = load_policy(policy_path)?;
    let cfg = TeacherConfig { candidates, rollouts, horizon, ..TeacherConfig::default() };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let results: Vec<(i8, Vec<f32>)> = py.allow_threads(|| pool.install(|| seeds.par_iter().map(|&seed| {
        let mut ep = Episode::new(seed, OpponentKind::Laika, Reward::default());
        let mut rng = Rng::new(seed + 101);
        let mut belief = (particles > 0).then(|| Belief::new(&ep.game, 1, particles, seed + 202));
        let (mut obs, mut mask) = (vec![0f32; policy.obs_dim], vec![0u8; policy.layout[3]]);
        let mut ms = Vec::new();
        while ep.outcome == Outcome::Running {
            let mut action = 0;
            let t0 = std::time::Instant::now();
            let (decided, mut searched) = (!ep.game.frozen, false);
            if decided {
                ep.observe(0, &mut obs, &mut mask);
                let probs = policy.shielded_probs(&obs, &mask);
                action = teacher::sample(&probs, &mut rng);
                searched = mpc && teacher::danger(&ep.game, &policy, &obs);
                if searched {
                    let copies = belief.as_ref().map_or(&[][..], |b| b.particles());
                    let s = teacher::scores(&ep, &policy, &probs, action, &cfg, &mut rng, copies);
                    let best = (0..18).filter(|&a| !s[a].is_nan()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
                    if s[best] > s[action] {
                        action = best;
                    }
                }
            }
            let mut elapsed = t0.elapsed().as_secs_f32() * 1000.0;
            let t1 = std::time::Instant::now();
            let before = belief.as_ref().map(|_| ep.game.clone());
            elapsed += t1.elapsed().as_secs_f32() * 1000.0;
            ep.step(action, None);
            if let (Some(b), Some(before)) = (belief.as_mut(), before) {
                let t1 = std::time::Instant::now();
                b.update(&before, action, &ep.game, &ep.game.events);
                elapsed += t1.elapsed().as_secs_f32() * 1000.0;
            }
            // Negative marks a frame decided without search.
            if decided {
                ms.push(if searched { elapsed } else { -elapsed });
            }
        }
        (outcome_code(ep.outcome), ms)
    }).collect()));
    let ms: Vec<f32> = results.iter().flat_map(|r| r.1.iter().copied()).collect();
    Ok((PyArray1::from_vec(py, results.iter().map(|r| r.0).collect()), PyArray1::from_vec(py, ms)))
}

#[pymodule]
fn tank_rl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_function(wrap_pyfunction!(teacher_labels, m)?)?;
    m.add_function(wrap_pyfunction!(play, m)?)?;
    Ok(())
}
