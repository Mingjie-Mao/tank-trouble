"use client";

import { useState } from "react";
import { setLanguage, useLanguage } from "../language";

type T = { zh: string; en: string };

// ---------------------------------------------------------------- data
// Every number below is from a pre-registered, seed-paired evaluation; see
// docs/REPORT.md.

const RESULTS = [
  { name: "League PPO", sub: { zh: "强化学习 · 网页演示", en: "Reinforcement learning · in the demo" },
    rate: "95.5%", detail: { zh: "1 万局 · 胜 9553 · 双亡 292 · 负 108 · 平 47", en: "10,000 rounds · 9553 W · 292 double KO · 108 L · 47 draws" },
    p95: "3.3 ms" },
  { name: "Tactical", sub: { zh: "实时 MPC 搜索", en: "Real-time MPC search" },
    rate: "91.0–91.7%", detail: { zh: "三组各 2000 局 · 主要失败是 30 秒僵持", en: "three sets of 2,000 · fails mainly by 30 s stalls" },
    p95: "9.2 ms" },
];

const PIPELINE: { title: T; stat: string; body: T }[] = [
  {
    title: { zh: "Tactical：实时 MPC 搜索", en: "Tactical: real-time MPC search" },
    stat: "91.5%",
    body: {
      zh: "每帧把候选动作复制到沙盒里前推 36 帧，Laika 用它自己的脚本从空状态重新跑（不读隐藏状态）。按接近可命中位置、弹道是否命中和暴露程度打分；子弹安全搜索否决会死的移动，结算审计否决会导致双亡的开火，僵持时拓扑追击。",
      en: "Each frame, candidate actions are rolled 36 frames forward in a sandbox with Laika re-run from a fresh state (no hidden state is read). Scored by progress toward firing positions, verified hits and exposure; a bullet-safety search vetoes moves that die, a settlement audit vetoes shots that end in a double KO, and a topological chase breaks stalls.",
    },
  },
  {
    title: { zh: "League PPO：联赛自博弈", en: "League PPO: league self-play" },
    stat: "93.9%",
    body: {
      zh: "网络每帧读取 1030 个可见事实，输出 18 个动作。只在一局结束时给奖励（胜 +1、双亡 −0.1、负或平 −1）。对手池：10% Laika、10% Tactical、80% 自己的历史版本。推理时加安全护盾：去掉会被打死的移动和会打到自己的开火。",
      en: "The network reads 1030 visible facts per frame and outputs 18 actions. Reward only at the end of a round (win +1, double KO −0.1, loss or draw −1). Opponents: 10% Laika, 10% Tactical, 80% its own past versions. At inference a safety shield removes moves that die and shots that hit the shooter.",
    },
  },
  {
    title: { zh: "MPC 老师", en: "MPC teacher" },
    stat: "+1.8pp",
    body: {
      zh: "失败多发生在死前约 1 秒的关键选择上。危险时刻（约 1/7 的帧），把网络最想做的几个动作交给网络自己在沙盒里各演 24 帧，选结局最好的。更强，但每次决策 0.2–0.6 秒，跑不了实时。",
      en: "Most failures are decided about a second before they happen. In danger (about one frame in seven) the network's top actions are each played out 24 frames by the network itself in a sandbox and the best is taken. Stronger, but 0.2–0.6 s per decision: too slow for real time.",
    },
  },
  {
    title: { zh: "蒸馏（专家迭代）", en: "Distillation (expert iteration)" },
    stat: "95.5%",
    body: {
      zh: "网络自己打 2000 局，在它遇到的 16.3 万个危险帧上请老师给候选动作打分；把网络原来的动作分布按 exp(16·得分) 向老师倾斜，再训练网络拟合。网络结构和每帧耗时不变，真胜率 +1.6pp。",
      en: "The network plays 2,000 rounds; on its 163,086 danger frames the teacher scores its candidate actions. The network's own distribution, tilted by exp(16·score), becomes the target. Same network and per-frame cost, +1.6pp true wins.",
    },
  },
];

const PAIRED: { label: T; d: number; lo: number; hi: number; n: string; ok: boolean }[] = [
  { label: { zh: "PPO − Tactical（第 1 组）", en: "PPO − Tactical (set 1)" }, d: 2.85, lo: 1.22, hi: 4.48, n: "2,000", ok: true },
  { label: { zh: "PPO − Tactical（第 2 组）", en: "PPO − Tactical (set 2)" }, d: 2.10, lo: 0.43, hi: 3.77, n: "2,000", ok: true },
  { label: { zh: "PPO − Tactical（第 3 组）", en: "PPO − Tactical (set 3)" }, d: 3.65, lo: 2.12, hi: 5.18, n: "2,000", ok: true },
  { label: { zh: "MPC 老师 − PPO", en: "MPC teacher − PPO" }, d: 1.80, lo: 0.49, hi: 3.11, n: "2,000", ok: true },
  { label: { zh: "蒸馏后 − 蒸馏前（2000 局）", en: "Distilled − PPO (2,000)" }, d: 0.65, lo: -0.55, hi: 1.85, n: "2,000", ok: false },
  { label: { zh: "蒸馏后 − 蒸馏前（1 万局）", en: "Distilled − PPO (10,000)" }, d: 1.59, lo: 1.02, hi: 2.16, n: "10,000", ok: true },
];

const BETA = [
  { x: "PPO", v: 93.55 }, { x: "β 4", v: 94.45 }, { x: "β 8", v: 95.25 },
  { x: "β 16", v: 96.70 }, { x: "β 32", v: 96.30 },
];
const SHUFFLED = 94.90;

const DEPTHS = [8, 16, 32, 64, 128];
const HEADROOM = {
  recoverable: { double: [31, 81, 97, 99, 99], loss: [16, 32, 79, 95, 95] },
  resample: { double: [11, 38, 68, 80, 84], loss: [2, 18, 47, 62, 72] },
};

const OBS: { name: T; dims: number; body: T }[] = [
  { name: { zh: "迷宫栅格", en: "Maze grid" }, dims: 840, body: { zh: "12×10 格 × {存在、4 面墙、我在此、对手在此}", en: "12×10 cells × {exists, 4 walls, me here, opponent here}" } },
  { name: { zh: "墙体射线", en: "Wall rays" }, dims: 16, body: { zh: "车体四周 16 个方向到最近墙的距离", en: "distance to the nearest wall in 16 directions" } },
  { name: { zh: "双方状态", en: "Both tanks" }, dims: 24, body: { zh: "位置、朝向、速度、转速、弹药、能否开火、是否贴墙；对手换算到我的坐标系", en: "pose, velocity, turn rate, ammo, readiness, wall contact; the opponent in my frame" } },
  { name: { zh: "导航", en: "Navigation" }, dims: 10, body: { zh: "BFS 路径长度、下一步方向、直线方位、死路深度", en: "BFS path length, next step direction, bearing, dead-end depth" } },
  { name: { zh: "双方瞄准", en: "Aim, both" }, dims: 10, body: { zh: "当前炮口这一枪：打中对手 / 打中自己 / 落空，命中时间，最近擦过距离", en: "the current barrel's shot: hit / self-hit / miss, time to hit, closest pass" } },
  { name: { zh: "逐发子弹", en: "Bullets" }, dims: 100, body: { zh: "每颗子弹的位置、速度、归属、是否反弹过、寿命，以及 75 帧内会不会打中谁、何时", en: "each bullet's position, velocity, owner, bounced, lifetime, and whether and when it reaches either tank within 75 frames" } },
  { name: { zh: "威胁与阶段", en: "Threat and phase" }, dims: 8, body: { zh: "最紧迫的来弹、瞄准我的子弹数，回合阶段与时间", en: "most urgent incoming bullet, bullets on course to me, round phase and clock" } },
  { name: { zh: "最近动作", en: "Recent actions" }, dims: 11, body: { zh: "最近 3 个动作、动作变更率、连续原地帧数", en: "last three actions, action-change rate, idle streak" } },
  { name: { zh: "躲避前瞻", en: "dodge_safety" }, dims: 9, body: { zh: "9 个移动方向各保持 24 帧会不会被打中——最重要的一组输入，网页“AI 视角”里的 3×3 方块就是它", en: "for each of nine moves, does holding it for 24 frames survive — the most important input; the 3×3 grid in the arena's AI view" } },
  { name: { zh: "先手", en: "First strike" }, dims: 2, body: { zh: "双方各需原地转几帧才能瞄准命中对方", en: "frames each side must turn before its shot hits the other" } },
];

const NEGATIVE: { idea: T; result: T }[] = [
  { idea: { zh: "训练从 3000 万步加到 1 亿步", en: "Training from 30M to 100M steps" }, result: { zh: "93.60% → 93.05%", en: "93.60% → 93.05%" } },
  { idea: { zh: "双亡惩罚从 −0.1 加到 −0.6", en: "Double-KO price −0.1 → −0.6" }, result: { zh: "无差别", en: "no difference" } },
  { idea: { zh: "深度护盾（沙盒预演后否决）", en: "Deep shield (veto after a sandbox rollout)" }, result: { zh: "不如普通护盾", en: "no gain over the plain shield" } },
  { idea: { zh: "对峙时交给 Tactical", en: "Hand face-offs to Tactical" }, result: { zh: "双亡变少、输变多，净胜不变", en: "fewer double KOs, more losses, no net gain" } },
  { idea: { zh: "先手特征（3 种子匹配消融）", en: "First-strike features (3-seed matched ablation)" }, result: { zh: "续训更稳，但不比起点强", en: "steadier fine-tuning, no gain" } },
  { idea: { zh: "第二轮蒸馏", en: "A second distillation round" }, result: { zh: "无增益：学生已追上老师", en: "no gain: the student caught up with the teacher" } },
];

// ---------------------------------------------------------------- charts

function PairedChart({ zh }: { zh: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 720;
  const row = 36;
  const labelW = 230;
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
        ? "按种子配对的胜率差（百分点）与 95% 置信区间。区间不跨 0 即显著；灰色那条说明 2000 局分辨不出 1.6pp，需要 1 万局。"
        : "Seed-paired win-rate differences (points) with 95% CIs. An interval clear of 0 is significant; the grey one shows 2,000 rounds cannot resolve 1.6pp — 10,000 can."}</p>
    </div>
  );
}

function BetaChart({ zh }: { zh: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 520;
  const h = 260;
  const x = (i: number) => 60 + i * ((w - 100) / (BETA.length - 1));
  const y = (v: number) => 20 + ((97.2 - v) / 4.2) * (h - 60);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={zh ? "蒸馏温度" : "Distillation beta"}>
        {[93, 94, 95, 96, 97].map((v) => (
          <g key={v}>
            <line x1={50} x2={w - 20} y1={y(v)} y2={y(v)} className="grid" />
            <text x={42} y={y(v) + 4} className="tick" textAnchor="end">{v}%</text>
          </g>
        ))}
        <polyline points={BETA.map((b, i) => `${x(i)},${y(b.v)}`).join(" ")} className="line" />
        {BETA.map((b, i) => (
          <g key={b.x} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <circle cx={x(i)} cy={y(b.v)} r={hover === i ? 8 : 6} className="dot" />
            <text x={x(i)} y={h - 16} className="tick" textAnchor="middle">{b.x}</text>
            {hover === i && <text x={x(i)} y={y(b.v) - 14} className="value" textAnchor="middle">{b.v.toFixed(2)}%</text>}
          </g>
        ))}
        <rect x={x(3) - 6} y={y(SHUFFLED) - 6} width={12} height={12} className="dot muted" transform={`rotate(45 ${x(3)} ${y(SHUFFLED)})`} />
        <text x={x(3) + 12} y={y(SHUFFLED) + 4} className="value">{zh ? `打乱老师打分 ${SHUFFLED}%` : `teacher scores shuffled ${SHUFFLED}%`}</text>
      </svg>
      <p className="caption">{zh
        ? "开发种子上不同 β 的胜率（各 2000 局）。打乱老师打分的对照只涨一部分：提升主要来自老师的选择，而不只是采样更集中。"
        : "Win rate on development seeds for each β (2,000 rounds each). The shuffled-teacher control gains only part of it: most of the gain is the teacher's choice, not just sharper sampling."}</p>
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
  const [step, setStep] = useState(3);
  const [block, setBlock] = useState(8);
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
          ? "两个 AI 对战 Tank Trouble 原版 AI Laika。真胜率：先击杀、随后被仍在飞行的子弹打死算双亡，不算胜；30 秒未分胜负算平局。所有测试都预注册、在从未用过的种子上按局配对。"
          : "Two AIs play Tank Trouble's original AI, Laika. True win rate: a kill followed by death to a bullet still in flight is a double KO, not a win; a round undecided after 30 s is a draw. Every test is pre-registered and seed-paired on never-used seeds."}</p>
      </header>

      <section className="cards">
        {RESULTS.map((r) => (
          <div key={r.name} className="card">
            <span>{t(r.sub)}</span>
            <strong>{r.name}</strong>
            <b>{r.rate}</b>
            <small>{t(r.detail)}</small>
            <small>{zh ? `每帧决策 p95 ${r.p95}` : `decision p95 ${r.p95}`}</small>
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
          <h2>{zh ? "蒸馏：向老师倾斜多少" : "Distillation: how far to tilt"}</h2>
          <BetaChart zh={zh} />
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
        <span>{zh ? "观测、奖励与规划器基础来自 KillField（MIT）" : "Observation, reward and planner build on KillField (MIT)"}</span>
      </footer>
    </main>
  );
}
