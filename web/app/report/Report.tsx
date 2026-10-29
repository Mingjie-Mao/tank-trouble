"use client";

import { useState } from "react";
import { setLanguage, useLanguage } from "../language";

type T = { zh: string; en: string };

// ---------------------------------------------------------------- data
// Every number below is from a pre-registered, seed-paired evaluation; see
// docs/REPORT.md.

const RESULTS: { name: T; sub: T; rate: string; detail: T; time: T }[] = [
  { name: { zh: "网络 + 4×4 搜索，跟踪 Laika", en: "Network + 4×4 search, tracking Laika" },
    sub: { zh: "网页演示，8 核及以上", en: "In the demo, 8+ cores" },
    rate: "98.92%",
    detail: { zh: "1 万局 · 胜 9892 · 双亡 32 · 负 27 · 平 49", en: "10,000 rounds · 9892 W · 32 double KO · 27 L · 49 draws" },
    time: { zh: "浏览器每帧 0.3 ms，危险帧 p50 20 ms（8 个线程）", en: "0.3 ms per frame in the browser, danger frames p50 20 ms (8 threads)" } },
  { name: { zh: "网络 + 2×2 搜索，跟踪 Laika", en: "Network + 2×2 search, tracking Laika" },
    sub: { zh: "网页演示，核数更少时", en: "In the demo, fewer cores" },
    rate: "98.53%",
    detail: { zh: "1 万局 · 胜 9853 · 双亡 45 · 负 40 · 平 62", en: "10,000 rounds · 9853 W · 45 double KO · 40 L · 62 draws" },
    time: { zh: "浏览器每帧 0.3 ms，危险帧 p50 23 ms", en: "0.3 ms per frame in the browser, danger frames p50 23 ms" } },
  { name: { zh: "网络 + 实时搜索", en: "Network + real-time search" },
    sub: { zh: "Laika 从零重建", en: "Laika rebuilt from scratch" },
    rate: "97.04%",
    detail: { zh: "1 万局 · 胜 9704 · 双亡 102 · 负 122 · 平 72", en: "10,000 rounds · 9704 W · 102 double KO · 122 L · 72 draws" },
    time: { zh: "浏览器每帧 0.3 ms，危险帧 p50 25 ms", en: "0.3 ms per frame in the browser, danger frames p50 25 ms" } },
  { name: { zh: "仅网络", en: "Network alone" },
    sub: { zh: "230 维观测", en: "230-value observation" },
    rate: "96.67%",
    detail: { zh: "1 万局 · 胜 9667 · 双亡 112 · 负 170 · 平 51", en: "10,000 rounds · 9667 W · 112 double KO · 170 L · 51 draws" },
    time: { zh: "浏览器每帧 0.3 ms", en: "0.3 ms per frame in the browser" } },
];

const PIPELINE: { title: T; stat: string; body: T }[] = [
  {
    title: { zh: "观测：230 个可见事实", en: "Observation: 230 visible facts" },
    stat: "230",
    body: {
      zh: "只用玩家能从屏幕上读出的事实：周围 5×5 格的墙、双方坦克、每颗子弹的弹道预测、Laika 自己的瞄准模型、每个移动方向的 24 帧生存前瞻（含 Laika 即将打出的那一枪），以及通往对手的路线和每一枪离对手最近能擦过多远。由消融实验逐组确定。",
      en: "Only facts a player could read off the screen: the walls of the 5×5 cells around, both tanks, every bullet's forecast, Laika's own aiming model, a 24-frame survival lookahead for each move (including the shot Laika is about to take), the route to the opponent and how near each shot passes. Settled group by group by ablation.",
    },
  },
  {
    title: { zh: "联赛 PPO", en: "League PPO" },
    stat: "96.0%",
    body: {
      zh: "Rust 环境里 512 局并行，约 1 万步/秒。对手 20% Laika、80% 自己最近 8 个历史版本。只在一局结束时给奖励：胜 +1，负或平 −1，双亡 −0.5。先训 3000 万步，再以学习率 1e-4、γ 0.9997 续训 6000 万步。推理时安全护盾去掉会被打死的移动和会打到自己的开火。",
      en: "512 rounds in parallel in Rust, about 10,000 steps/s. Opponents: 20% Laika, 80% its last eight frozen versions. Reward only at the end: win +1, loss or draw −1, double KO −0.5. 30M steps, then 60M more at learning rate 1e-4 and γ 0.9997. At inference a safety shield removes moves that die and shots that hit the shooter.",
    },
  },
  {
    title: { zh: "MPC 老师", en: "MPC teacher" },
    stat: "+0.67pp",
    body: {
      zh: "危险时（约 1/4 的帧），把网络最想做的几个动作放进公平沙盒（Laika 从头重建、独立随机数）各推演 24 帧。在 6000 局、37 万个危险帧上打分，目标是 π·exp(16·得分)；再训 2000 万步 PPO，每个小批次都加上向老师靠拢的 KL 损失。比同样步数的单纯续训高 0.67 个百分点。",
      en: "In danger (about a quarter of frames) the network's top actions are each played out 24 frames in a fair sandbox (Laika rebuilt from scratch, an independent random stream). Scored on 372,779 danger frames from 6,000 rounds; the target is π·exp(16·score); 20M more PPO steps add a KL pull toward it to every minibatch. +0.67pp over the same steps without it.",
    },
  },
  {
    title: { zh: "实时搜索", en: "Real-time search" },
    stat: "97.04%",
    body: {
      zh: "对战时，危险帧上用同一个搜索（2 个候选 × 2 次推演），30 ms 预算内先评网络采样的动作、再评其他候选，只有严格更好才替换；超时就用网络的动作。Rust 编译成 WebAssembly，在浏览器的 Web Worker 里跑。",
      en: "In play, danger frames run the same search (2 candidates × 2 rollouts) under a 30 ms budget: the network's sampled action first, then the others, replaced only by a strictly better one; out of time, the network's action stands. Rust compiled to WebAssembly, in a Web Worker.",
    },
  },
  {
    title: { zh: "跟踪 Laika", en: "Tracking Laika" },
    stat: "98.53%",
    body: {
      zh: "作弊诊断：让搜索里的 Laika 复制真 Laika 的记忆（目标、排队中的动作、计时器），开发集从 97.33% 升到 98.72%；再复制随机数也不再提高。记忆看不见，但动作看得见：16 个 Laika 副本逐帧重放，只留下坦克移动和开火与真 Laika 一致的，搜索就和它们对推。99.95% 的帧里至少有一个副本与真实状态完全一致，每帧多花约 0.1 ms。",
      en: "A cheating diagnostic: giving the search a copy of the real Laika's memory (goal, queued actions, timers) lifts development seeds from 97.33% to 98.72%; its random stream adds nothing more. Its memory is hidden but its moves are not: sixteen Laika copies replay every frame, those whose tank moved and fired as the real one did survive, and the search plays them. A copy holds the exact state on 99.95% of frames, for about 0.1 ms per frame.",
    },
  },
  {
    title: { zh: "多线程加宽搜索", en: "A wider search, in parallel" },
    stat: "98.92%",
    body: {
      zh: "Laika 对了，算力就有用：4 个候选 × 4 次推演在开发集上比 2×2 高 0.34 个百分点，留出集 98.92%，被打死从 40 局降到 27 局。最多 20 次推演一个线程 30 ms 算不完，于是分给最多 8 个 Web Worker：每个辅助线程持有一份棋局副本，重放每条命令和每帧动作并核对状态指纹；超时 5 ms 的算未完成，副本不同步就退回单线程 2×2。推演本身也提速到 2.9 ms，结果逐位不变。",
      en: "Once the search's Laika is right, compute pays: 4 candidates × 4 rollouts beat 2×2 by 0.34pp on development seeds and reach 98.92% held out, losses down from 40 to 27. Up to 20 rollouts do not fit one thread's 30 ms, so they are shared among up to eight Web Workers, each helper holding a copy of the arena that replays every command and every frame's moves and checks a fingerprint of the state; an answer 5 ms late counts as unplayed, and a copy that drifts drops the search to one worker and 2×2. The rollout itself got faster too, 2.9 ms, bit for bit the same.",
    },
  },
];

const PAIRED: { label: T; d: number; lo: number; hi: number; n: string; ok: boolean }[] = [
  { label: { zh: "长训练网络 − 上一版", en: "Long-trained − previous" }, d: 0.66, lo: 0.15, hi: 1.17, n: "10,000", ok: true },
  { label: { zh: "老师指导 − 单纯续训", en: "With teacher − plain PPO" }, d: 0.67, lo: 0.16, hi: 1.18, n: "10,000", ok: true },
  { label: { zh: "网络 − 上一版", en: "Network − previous" }, d: 0.73, lo: 0.21, hi: 1.25, n: "10,000", ok: true },
  { label: { zh: "网络 + 搜索 − 上一版", en: "Network + search − previous" }, d: 1.10, lo: 0.59, hi: 1.61, n: "10,000", ok: true },
  { label: { zh: "网络 + 搜索 − 网络", en: "Network + search − network" }, d: 0.37, lo: -0.11, hi: 0.85, n: "10,000", ok: false },
  { label: { zh: "跟踪 Laika − 从零重建", en: "Tracked Laika − fresh Laika" }, d: 1.49, lo: 1.18, hi: 1.80, n: "10,000", ok: true },
  { label: { zh: "4×4 − 2×2，都跟踪 Laika", en: "4×4 − 2×2, both tracking" }, d: 0.39, lo: 0.09, hi: 0.69, n: "10,000", ok: true },
];

const SEEDS: { label: T; runs: number[] }[] = [
  { label: { zh: "1030 维 · 3000 万步", en: "1030 values · 30M" }, runs: [92.7, 91.8, 79.5] },
  { label: { zh: "230 维 · 3000 万步", en: "230 values · 30M" }, runs: [90.1, 74.2, 86.2] },
  { label: { zh: "230 维 · 续训 6000 万步", en: "230 values · +60M" }, runs: [91.6, 96.0] },
];

const DEPTHS = [8, 16, 32, 64, 128];
const HEADROOM = {
  recoverable: { double: [31, 81, 97, 99, 99], loss: [16, 32, 79, 95, 95] },
  resample: { double: [11, 38, 68, 80, 84], loss: [2, 18, 47, 62, 72] },
};

const OBS: { name: T; dims: number; body: T }[] = [
  { name: { zh: "自身", en: "Self" }, dims: 8, body: { zh: "弹药、能否开火、朝向、自身坐标系里的速度、转速、是否贴墙", en: "ammo, ready to fire, heading, velocity in my frame, turn rate, wall contact" } },
  { name: { zh: "局部墙", en: "Local walls" }, dims: 62, body: { zh: "我周围 5×5 格的每条墙边，以及我在格内的位置", en: "every wall edge of the 5×5 cells around me, and my offset in my cell" } },
  { name: { zh: "对手", en: "Opponent" }, dims: 13, body: { zh: "我坐标系里的位置和距离、迷宫距离、朝它走的下一格、有无视线、它的炮口对我的角度、接近速度", en: "position and distance in my frame, maze distance, next cell toward it, line of sight, its barrel against my bearing, closing speed" } },
  { name: { zh: "炮管", en: "Barrels" }, dims: 6, body: { zh: "双方现在开火会打中对手还是自己、几帧后——用引擎自己的子弹子步计算", en: "would each barrel's shot now hit the opponent or the shooter, and when — with the engine's own bullet substeps" } },
  { name: { zh: "Laika 的意图", en: "Laika's intent" }, dims: 7, body: { zh: "按它自己的瞄准模型它是否马上开火；双方各要转几帧才能瞄中；是否在它 7 格追击、2 格必开火的范围内；它上次开火多久了；它的弹药", en: "its own aiming model says it is about to fire; frames each side must turn to hold a hit; inside its 7-cell hunting and 2-cell firing range; time since its last shot; its ammo" } },
  { name: { zh: "逐发子弹", en: "Bullets" }, dims: 100, body: { zh: "每颗子弹按紧迫程度排序：位置、方向、归属、是否反弹过、寿命、会不会打中我和何时、最近擦过距离", en: "each bullet, most urgent first: position, direction, owner, bounced, lifetime, forecast hit on me and when, closest pass" } },
  { name: { zh: "生存前瞻", en: "Survival" }, dims: 9, body: { zh: "9 个移动方向各保持 24 帧会不会被打中，包括 Laika 即将打出的那一枪——网页“AI 视角”里的 3×3 方块就是它", en: "for each of nine moves, does holding it 24 frames survive, the shot Laika is about to take included — the 3×3 grid in the arena's AI view" } },
  { name: { zh: "时间", en: "Time" }, dims: 6, body: { zh: "回合时钟、是否结算中、上一个动作、我上次开火多久了", en: "round clock, settlement, last action, time since my last shot" } },
  { name: { zh: "战术", en: "Tactics" }, dims: 19, body: { zh: "游戏自带的最短路线（下一个路口方向、还要转多少）、双方格子的出口数、每一枪离对手最近擦过多远、左右各转 3 帧后会不会更近、子弹威胁汇总、当前动作保持了多久", en: "the game's own shortest route (next waypoint, how far to turn), exits of both cells, how near each shot passes and whether turning three frames either way passes nearer, the bullets' summed threat, how long the current move has been held" } },
];

const NEGATIVE: { idea: T; result: T }[] = [
  { idea: { zh: "用单次 3000 万步训练比较观测", en: "Comparing observations on single 30M runs" }, result: { zh: "种子波动（79–93%）比差别还大", en: "the seed spread (79–93%) exceeds the differences" } },
  { idea: { zh: "平局惩罚从 −1 加到 −2", en: "Draw priced at −2 instead of −1" }, result: { zh: "平局变少、输变多，胜率不变（3 种子）", en: "fewer draws, more losses, same win rate (3 seeds)" } },
  { idea: { zh: "一半对局打 Laika（原为 1/5）", en: "Half the rounds against Laika (was a fifth)" }, result: { zh: "均值不变，波动更大（3 种子）", en: "same mean, wider spread (3 seeds)" } },
  { idea: { zh: "用价值网络给未分胜负的推演打分", en: "Scoring undecided rollouts by the critic" }, result: { zh: "学生一样，标注慢 2.5 倍", en: "same student, labelling 2.5× slower" } },
  { idea: { zh: "更宽的实时搜索（3×3），Laika 从零重建", en: "A wider real-time search (3×3), Laika rebuilt from scratch" }, result: { zh: "不比 2×2 强，慢一倍", en: "no stronger than 2×2, twice as slow" } },
  { idea: { zh: "更强的搜索（4×8、32 帧），Laika 从零重建", en: "A stronger search (4×8, 32 frames), Laika rebuilt from scratch" }, result: { zh: "+0.35，不显著（6000 局）——缺的是对 Laika 的了解，不是算力", en: "+0.35pp, not significant (6,000 rounds) — what was missing was knowledge of Laika, not compute" } },
  { idea: { zh: "第二轮专家迭代", en: "A second round of expert iteration" }, result: { zh: "持平", en: "no change" } },
  { idea: { zh: "网络加宽到 512", en: "A 512-unit network" }, result: { zh: "训练更稳，最终持平；加搜索后 −0.62", en: "steadier training, same end point; −0.62pp with search" } },
];

// ---------------------------------------------------------------- charts

function PairedChart({ zh }: { zh: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 720;
  const row = 36;
  const labelW = 250;
  const x = (v: number) => labelW + ((v + 1) / 7) * (w - labelW - 20);
  const h = PAIRED.length * row + 34;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={zh ? "配对检验" : "Paired tests"}>
        {[-1, 0, 1, 2, 3, 4, 5, 6].map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={4} y2={h - 26} className={v === 0 ? "axis-zero" : "grid"} />
            <text x={x(v)} y={h - 8} className="tick" textAnchor="middle">{v > 0 ? `+${v}` : v}</text>
          </g>
        ))}
        {PAIRED.map((p, i) => {
          const y = 22 + i * row;
          return (
            <g key={i} className={hover === i ? "row hover" : "row"}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={y - row / 2} width={w} height={row} className="row-bg" />
              <text x={labelW - 12} y={y + 4} textAnchor="end" className="label">{p.label[zh ? "zh" : "en"]}</text>
              <line x1={x(p.lo)} x2={x(p.hi)} y1={y} y2={y} className={p.ok ? "ci" : "ci muted"} />
              <circle cx={x(p.d)} cy={y} r={6} className={p.ok ? "dot" : "dot muted"} />
              {hover === i && (
                <text x={x(p.hi) + 10} y={y + 4} className="value">
                  {`${p.d > 0 ? "+" : ""}${p.d.toFixed(2)}pp [${p.lo.toFixed(2)}, +${p.hi.toFixed(2)}] · ${p.n}`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <p className="caption">{zh
        ? "按种子配对的胜率差（百分点）与 95% 置信区间，各 1 万局。区间不跨 0 即显著；灰色那条是实时搜索在网络之上的增益，这次没测出显著差别。"
        : "Seed-paired win-rate differences (points) with 95% CIs, 10,000 rounds each. An interval clear of 0 is significant; the grey one, the search's gain over the network alone, was not resolved."}</p>
    </div>
  );
}

function SeedChart({ zh }: { zh: boolean }) {
  const [hover, setHover] = useState<string | null>(null);
  const w = 560;
  const h = 260;
  const x = (i: number) => 110 + i * ((w - 230) / (SEEDS.length - 1));
  const y = (v: number) => 20 + ((100 - v) / 30) * (h - 60);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={zh ? "种子波动" : "Seed spread"}>
        {[70, 80, 90, 100].map((v) => (
          <g key={v}>
            <line x1={50} x2={w - 20} y1={y(v)} y2={y(v)} className="grid" />
            <text x={42} y={y(v) + 4} className="tick" textAnchor="end">{v}%</text>
          </g>
        ))}
        {SEEDS.map((g, i) => (
          <g key={i}>
            <text x={x(i)} y={h - 16} className="tick" textAnchor="middle">{g.label[zh ? "zh" : "en"]}</text>
            {g.runs.map((v, k) => {
              const id = `${i}-${k}`;
              return (
                <g key={id} onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)}>
                  <circle cx={x(i) + (k - (g.runs.length - 1) / 2) * 16} cy={y(v)} r={hover === id ? 8 : 6} className="dot" />
                  {hover === id && <text x={x(i) + 14} y={y(v) - 10} className="value">{v.toFixed(1)}%</text>}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <p className="caption">{zh
        ? "同一配置换训练种子，3000 万步时可能差 15 个百分点：有的种子困在自对弈的 30 秒平局里。所以只看长训练后、在从未用过的种子上配对测 1 万局的结果。"
        : "The same configuration on different training seeds can differ by 15 points at 30M steps: some seeds stay stuck in self-play 30 s draws. Hence long training, and only 10,000-round paired tests on never-used seeds."}</p>
    </div>
  );
}

function HeadroomChart({ zh }: { zh: boolean }) {
  const [metric, setMetric] = useState<"recoverable" | "resample">("recoverable");
  const w = 520;
  const h = 260;
  const x = (i: number) => 60 + i * ((w - 100) / (DEPTHS.length - 1));
  const y = (v: number) => 20 + ((100 - v) / 100) * (h - 60);
  const series = HEADROOM[metric];
  const lines: { key: "double" | "loss"; cls: string; name: T }[] = [
    { key: "double", cls: "line", name: { zh: "双亡局", en: "double KOs" } },
    { key: "loss", cls: "line alt", name: { zh: "被打死的局", en: "losses" } },
  ];
  return (
    <div className="chart">
      <div className="chart-tabs">
        <button className={metric === "recoverable" ? "active" : ""} onClick={() => setMetric("recoverable")}>
          {zh ? "存在赢法" : "A win exists"}
        </button>
        <button className={metric === "resample" ? "active" : ""} onClick={() => setMetric("resample")}>
          {zh ? "网络重走一遍就赢" : "The network wins on a resample"}
        </button>
      </div>
      <div className="legend">
        {lines.map((l) => (
          <span key={l.key}><i className={l.key === "double" ? "swatch" : "swatch alt"} />{l.name[zh ? "zh" : "en"]}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={zh ? "失败局能否救回" : "Recoverable failures"}>
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line x1={50} x2={w - 20} y1={y(v)} y2={y(v)} className="grid" />
            <text x={42} y={y(v) + 4} className="tick" textAnchor="end">{v}%</text>
          </g>
        ))}
        {DEPTHS.map((d, i) => (
          <text key={d} x={x(i)} y={h - 16} className="tick" textAnchor="middle">
            {zh ? `${d} 帧前` : `${d} frames`}
          </text>
        ))}
        {lines.map((l) => (
          <g key={l.key}>
            <polyline points={series[l.key].map((v, i) => `${x(i)},${y(v)}`).join(" ")} className={l.cls} />
            {series[l.key].map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r={5} className={l.key === "double" ? "dot" : "dot alt"}>
                <title>{`${l.name[zh ? "zh" : "en"]} · ${DEPTHS[i]} · ${v}%`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <p className="caption">{zh
        ? "把每局失败重放到失败前若干帧，再重试 24 次。死前 1.3 秒（32 帧）几乎都还有赢法，0.3 秒（8 帧）基本没救——失败是关键时刻的选择问题，这正是 MPC 老师和蒸馏要解决的。"
        : "Each failed round is replayed to k frames before the failure and retried 24 times. From 1.3 s (32 frames) before, nearly all can still be won; from 0.3 s (8 frames), almost none — failures are decisions at critical moments, which the MPC teacher and distillation address."}</p>
    </div>
  );
}

// ---------------------------------------------------------------- page

export function Report() {
  const language = useLanguage();
  const zh = language === "zh";
  const t = (text: T) => text[language];
  const [step, setStep] = useState(5);
  const [block, setBlock] = useState(6);
  const total = OBS.reduce((sum, b) => sum + b.dims, 0);

  return (
    <main className="site-shell report">
      <nav className="report-nav">
        {/* A plain link: vinext's <Link> prefetch throws on Cloudflare Pages. */}
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a className="secondary" href="/">{zh ? "← 对战" : "← Arena"}</a>
        <div className="language-switch" role="group" aria-label={zh ? "语言切换" : "Language switcher"}>
          <button type="button" className={zh ? "active" : ""} aria-pressed={zh} onClick={() => setLanguage("zh")}>中</button>
          <button type="button" className={!zh ? "active" : ""} aria-pressed={!zh} onClick={() => setLanguage("en")}>EN</button>
        </div>
      </nav>

      <header className="report-head">
        <h1>{zh ? "方法与结果" : "Method and results"}</h1>
        <p>{zh
          ? "我们的 AI 对战 Tank Trouble 原版 AI Laika。真胜率：先击杀、随后被仍在飞行的子弹打死算双亡，不算胜；30 秒未分胜负算平局。所有测试都预注册、在训练和调参都没用过的种子上按局配对。与公开项目 KillField 的规划器正面对战 1000 局：583 胜 201 负。"
          : "Our AI plays Tank Trouble's original AI, Laika. True win rate: a kill followed by death to a bullet still in flight is a double KO, not a win; a round undecided after 30 s is a draw. Every test is pre-registered and seed-paired on seeds never used for training or tuning. Head to head against KillField's planner, a public Tank Trouble AI: 583 wins to 201 over 1,000 rounds."}</p>
      </header>

      <section className="cards">
        {RESULTS.map((r) => (
          <div key={r.rate} className="card">
            <span>{t(r.sub)}</span>
            <strong>{t(r.name)}</strong>
            <b>{r.rate}</b>
            <small>{t(r.detail)}</small>
            <small>{t(r.time)}</small>
          </div>
        ))}
      </section>

      <section>
        <h2>{zh ? "怎么做的" : "How it works"}</h2>
        <div className="pipeline">
          {PIPELINE.map((p, i) => (
            <button key={i} className={step === i ? "step active" : "step"} onClick={() => setStep(i)}>
              <span>{i + 1}</span>
              <strong>{t(p.title)}</strong>
              <b>{p.stat}</b>
            </button>
          ))}
        </div>
        <p className="step-body">{t(PIPELINE[step].body)}</p>
      </section>

      <section>
        <h2>{zh ? "配对检验" : "Paired tests"}</h2>
        <PairedChart zh={zh} />
      </section>

      <section className="two-up">
        <div>
          <h2>{zh ? "失败局能否救回" : "Can a failed round be saved?"}</h2>
          <HeadroomChart zh={zh} />
        </div>
        <div>
          <h2>{zh ? "为什么要长训练" : "Why train long"}</h2>
          <SeedChart zh={zh} />
        </div>
      </section>

      <section>
        <h2>{zh ? `网络看到的 ${total} 个数` : `The ${total} numbers the network sees`}</h2>
        <div className="obs">
          {OBS.map((b, i) => (
            <button key={i} className={block === i ? "obs-block active" : "obs-block"}
              style={{ flexGrow: Math.max(1, Math.log2(b.dims)) }}
              onMouseEnter={() => setBlock(i)} onFocus={() => setBlock(i)} onClick={() => setBlock(i)}>
              <strong>{t(b.name)}</strong>
              <span>{b.dims}</span>
            </button>
          ))}
        </div>
        <p className="step-body">{t(OBS[block].body)}</p>
      </section>

      <section>
        <h2>{zh ? "试过但没用的" : "What did not help"}</h2>
        <table className="report-table">
          <tbody>
            {NEGATIVE.map((n, i) => (
              <tr key={i}><th>{t(n.idea)}</th><td>{t(n.result)}</td></tr>
            ))}
          </tbody>
        </table>
      </section>

      <footer className="report-foot">
        <a href="https://github.com/Mingjie-Mao/tank-trouble" target="_blank" rel="noreferrer">GitHub ↗</a>
      </footer>
    </main>
  );
}
