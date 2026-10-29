//! The actor network of `training/ppo.py` (and its critic, when exported),
//! evaluated in plain Rust.
//!
//! Weights come from `ppo.py export`: a JSON header (observation layout and
//! each tensor's name, shape and offset) followed by little-endian f32 data.
//! Branches: a shared per-bullet MLP pooled by masked mean and max, a scalar
//! MLP, a trunk and the actor (and critic) head.

use std::collections::HashMap;

/// Weights kept transposed (column-major), so every row accumulates at once
/// in its own lane, each still summing in column order.
struct Linear {
    wt: Vec<f32>,
    b: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Linear {
    fn new(w: &[f32], b: &[f32], rows: usize, cols: usize) -> Linear {
        let wt = (0..cols * rows).map(|i| w[(i % rows) * cols + i / rows]).collect();
        Linear { wt, b: b.to_vec(), rows, cols }
    }

    fn apply(&self, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.extend_from_slice(&self.b);
        for (col, &xc) in self.wt.chunks_exact(self.rows).zip(&x[..self.cols]) {
            for (o, &w) in out.iter_mut().zip(col) {
                *o += w * xc;
            }
        }
    }
}

fn relu(v: &mut [f32]) {
    for x in v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// (map_start, map_len, bullet_start, bullet_slots, bullet_dim, dodge_start, self_hit)
pub type Layout = [usize; 7];

pub struct Policy {
    pub layout: Layout,
    pub obs_dim: usize,
    bullets: (Linear, Linear),
    bullet_out: Linear,
    scalars: (Linear, Linear),
    trunk: Linear,
    actor: Linear,
    critic: Option<Linear>,
    scalar_idx: Vec<usize>,
}

impl Policy {
    /// Parse `ppo.py export` output.
    pub fn from_bytes(bytes: &[u8]) -> Result<Policy, String> {
        if bytes.len() < 4 {
            return Err("truncated policy file".into());
        }
        let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[4..4 + n]).map_err(|e| e.to_string())?;
        let data = &bytes[4 + n..];
        let floats: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut tensors: HashMap<String, (Vec<usize>, usize)> = HashMap::new();
        for t in header["tensors"].as_array().ok_or("no tensors")? {
            let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            tensors.insert(t["name"].as_str().unwrap().to_string(), (shape, t["offset"].as_u64().unwrap() as usize));
        }
        let linear = |name: &str| -> Result<Linear, String> {
            let (ws, wo) = tensors.get(&format!("{name}.weight")).ok_or(format!("missing {name}.weight"))?;
            let (bs, bo) = tensors.get(&format!("{name}.bias")).ok_or(format!("missing {name}.bias"))?;
            let (rows, cols) = (ws[0], ws[1]);
            Ok(Linear::new(&floats[*wo..wo + rows * cols], &floats[*bo..bo + bs[0]], rows, cols))
        };
        let layout_v: Vec<usize> = header["layout"].as_array().ok_or("no layout")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let layout: Layout = layout_v.try_into().map_err(|_| "layout needs 7 values")?;
        let obs_dim = header["obs_dim"].as_u64().ok_or("no obs_dim")? as usize;
        if layout[1] > 0 {
            return Err("observations with a map block are no longer supported".into());
        }
        let mut keep = vec![true; obs_dim];
        keep[layout[2]..layout[2] + layout[3] * layout[4]].fill(false);
        Ok(Policy {
            layout,
            obs_dim,
            bullets: (linear("bullets.0")?, linear("bullets.2")?),
            bullet_out: linear("bullet_out.0")?,
            scalars: (linear("scalars.0")?, linear("scalars.2")?),
            trunk: linear("trunk.0")?,
            actor: linear("actor")?,
            critic: linear("critic").ok(),
            scalar_idx: (0..obs_dim).filter(|&i| keep[i]).collect(),
        })
    }

    /// Action logits (18) for one observation and its bullet mask.
    pub fn logits(&self, obs: &[f32], mask: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(18);
        self.actor.apply(&self.features(obs, mask), &mut out);
        out
    }

    /// The critic's value of the observation, if the file carries the critic.
    pub fn value(&self, obs: &[f32], mask: &[u8]) -> Option<f32> {
        let critic = self.critic.as_ref()?;
        let mut out = Vec::with_capacity(1);
        critic.apply(&self.features(obs, mask), &mut out);
        Some(out[0])
    }

    /// The shared trunk's output.
    fn features(&self, obs: &[f32], mask: &[u8]) -> Vec<f32> {
        let [_, _, b_start, slots, b_dim, ..] = self.layout;
        let (mut a, mut b) = (Vec::with_capacity(256), Vec::with_capacity(256));
        let mut joined = Vec::with_capacity(320);
        let width = self.bullets.1.rows;
        let mut sum = vec![0f32; width];
        let mut max = vec![0f32; width];
        let mut count = 0;
        for s in 0..slots {
            if mask[s] == 0 {
                continue;
            }
            self.bullets.0.apply(&obs[b_start + s * b_dim..b_start + (s + 1) * b_dim], &mut a);
            relu(&mut a);
            self.bullets.1.apply(&a, &mut b);
            relu(&mut b);
            for i in 0..width {
                sum[i] += b[i];
                max[i] = if count == 0 { b[i] } else { max[i].max(b[i]) };
            }
            count += 1;
        }
        let mut pooled: Vec<f32> = sum.iter().map(|v| if count > 0 { v / count as f32 } else { 0.0 }).collect();
        pooled.extend_from_slice(&max);
        self.bullet_out.apply(&pooled, &mut a);
        relu(&mut a);
        joined.extend_from_slice(&a);
        let scalars: Vec<f32> = self.scalar_idx.iter().map(|&i| obs[i]).collect();
        self.scalars.0.apply(&scalars, &mut a);
        for v in a.iter_mut() {
            *v = v.tanh();
        }
        self.scalars.1.apply(&a, &mut b);
        relu(&mut b);
        joined.extend_from_slice(&b);
        self.trunk.apply(&joined, &mut a);
        relu(&mut a);
        a
    }

    /// Softmax of the logits, then the safety shield: drop moves the nine-move
    /// survival lookahead says die (if any survives; otherwise keep the
    /// longest-lasting) and fire while the current shot would hit the shooter.
    pub fn shielded_probs(&self, obs: &[f32], mask: &[u8]) -> [f32; 18] {
        self.shield(&self.logits(obs, mask), obs)
    }

    /// `shielded_probs` from logits already computed.
    pub fn shield(&self, logits: &[f32], obs: &[f32]) -> [f32; 18] {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = [0f32; 18];
        let mut total = 0.0;
        for i in 0..18 {
            p[i] = (logits[i] - max).exp();
            total += p[i];
        }
        let dodge = &obs[self.layout[5]..self.layout[5] + 9];
        let any_safe = dodge.iter().any(|&d| d >= 0.0);
        let longest = dodge.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let self_hit = obs[self.layout[6]] > 0.5;
        let mut kept = 0.0;
        let mut keep = [false; 18];
        for a in 0..18 {
            let d = dodge[a / 2];
            keep[a] = (if any_safe { d >= 0.0 } else { d == longest }) && !(self_hit && a % 2 == 1);
            if keep[a] {
                kept += p[a];
            }
        }
        for a in 0..18 {
            p[a] = if kept > 0.0 { if keep[a] { p[a] / kept } else { 0.0 } } else { p[a] / total };
        }
        p
    }
}
