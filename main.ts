// gta6-trophy-rarity-forecast-01 — forecasting GTA VI trophy earn rates
// on Exophase one week after launch, from the house's page reads.
//
// The page Exophase shows is not a live counter. It freezes for hours and
// then jumps: the tracked-player count leaps, and almost every earn rate
// falls, because the players who just appeared have barely played. So a
// rate is a ratio of two counts that move for different reasons — the
// denominator grows as more people buy the game and sync a profile, the
// numerator grows as the players already counted keep playing — and this
// program forecasts the ratio by treating the two sides separately
// wherever the data allows it, and by fitting the ratio's own drift where
// it does not. Its bands are kept wide on purpose: a refresh can move a
// rate a long way in one step, and a band that pretends otherwise is
// wrong the moment the page jumps.
//
// The forecast is built in layers, one trophy at a time:
//   1. The read history is squeezed into "slices": runs of reads that
//      print identical numbers are one frozen page. A slice ends when a
//      read prints different numbers, which is when Exophase refreshed;
//      the refresh is placed halfway across the gap between the last read
//      of the old slice and the first read of the new one.
//   2. Earners are recovered per slice: earners = rate x tracked / 100,
//      rounded. Rates print to two decimals, so while tracked players
//      number in the hundreds this recovers the exact earner count.
//   3. The tracked count at the target is projected two ways — a power
//      law against log time and a saturating exponential — and the two
//      projections are averaged in log space. With a single slice (no
//      refresh ever seen) a launch-night prior takes over instead, tuned
//      to the one refresh the rules publish (638 tracked becoming 3,868).
//   4. Each trophy's rate is projected from its own history. With three
//      or more slices two trend shapes are fitted — a saturating curve
//      that rises toward a ceiling, and a straight line of the logit's
//      rate against log time — and blended three-to-one toward the curve,
//      because a flattening rate is the usual case by day three and a
//      line drawn from it overshoots. Both fits are weighted by binomial
//      precision and pulled toward the average pace of trophies of the
//      same kind. With two slices the trend is averaged with the
//      dilution arithmetic; with one slice the dilution arithmetic
//      stands alone.
//   5. Dilution is the arithmetic of a refresh: when tracked players
//      jump from N to N', the newcomers bring earners at a fraction
//      exp(-H / 0.75) of the old rate, H being the hours of play the
//      trophy usually needs — except the opening mission, which
//      everybody finishes first, so it stays at 100%.
//   6. Guardrails: earners never fall, and no trophy may pass its
//      kind's generous one-week ceiling.
//   7. The quantile spread on the logit scale combines the fit's own
//      residual (grown by how far the target sits past the data), a
//      random-walk term over the horizon, a refresh-jump term tied to
//      how much the tracked count is expected to grow, a small-count
//      term, and a floor. A trophy at 100% on every slice gets a hedged
//      row instead: the median stays at 100% while the lower quantiles
//      admit a flood of new players could pull it down.
//
// Deterministic throughout: no randomness, no clock, no network — the
// output depends only on the case file. Standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The five quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
/** A rate in percent, kept inside the range the scorer's logit can take. */
const clampPct = (v: number): number => clamp(v, 0.01, 99.99);
/** Logit of a percent. */
const logitOf = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** Inverse logit, back to a percent. */
const pctOfLogit = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -12, 12)));
/** Rounded to 4 decimals, as the starter rounds its rows. */
const round4 = (v: number): number => Math.round(v * 10_000) / 10_000;

/**
 * Inverse of the standard normal CDF, Acklam's rational approximation.
 * The coefficients are the published ones for that approximation; the
 * layout here is my own.
 */
function normalQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = 0.02425;
  const poly = (coeffs: number[], x: number): number => {
    let acc = coeffs[0];
    for (let i = 1; i < coeffs.length; i++) acc = acc * x + coeffs[i];
    return acc;
  };
  if (p < tail) {
    const q = Math.sqrt(-2 * Math.log(p));
    return poly(c, q) / (poly(d, q) * q + 1);
  }
  if (p > 1 - tail) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -poly(c, q) / (poly(d, q) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (poly(a, r) * q) / (poly(b, r) * r + 1);
}

/**
 * Weighted least squares of y on x. Returns [intercept, slope]; with a
 * degenerate design the slope is 0 and the intercept the mean of y.
 */
function weightedLine(xs: number[], ys: number[], ws: number[]): [number, number] {
  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = ws[i];
    sw += w;
    swx += w * xs[i];
    swy += w * ys[i];
    swxx += w * xs[i] * xs[i];
    swxy += w * xs[i] * ys[i];
  }
  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-12 || sw <= 0) return [sw > 0 ? swy / sw : 0, 0];
  const slope = (sw * swxy - swx * swy) / denom;
  return [(swy - slope * swx) / sw, slope];
}

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface PageRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: PageRead[];
  notes?: string;
}

/** One frozen page of Exophase history. */
interface Slice {
  /** Hours since release at the estimated refresh moment. */
  hours: number;
  /** Players tracked on that page. */
  players: number;
  rates: Array<number | null>;
}

/** What a trophy is like, as read off its own description and tier. */
interface TrophyKind {
  label: string;
  /** Hours of play earning it usually takes — a gameplay prior. */
  hours: number;
  /** Generous ceiling, in percent, for one week of play. */
  ceiling: number;
}

/**
 * Kind rules, in the order they are tried: the first match wins. The
 * numbers are gameplay priors, not rates — a chapter number, a stunt
 * count or a dollar amount in the description is the only evidence
 * used, and the same table applies to every trophy the case lists.
 */
const KIND_RULES: Array<{ test: RegExp; hours: (m: RegExpMatchArray) => number; ceiling: number; label: string }> = [
  { test: /complete chapter (\d+)/i, hours: (m) => 1.5 + Number(m[1]), ceiling: 90, label: 'chapter' },
  { test: /final mission/i, hours: () => 25, ceiling: 70, label: 'finale' },
  { test: /100% completion/i, hours: () => 200, ceiling: 8, label: 'completion' },
  { test: /complete (\d+) stunts/i, hours: (m) => 5 + 0.15 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /complete (\d+) challenges/i, hours: (m) => 5 + 0.15 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /(\d+) locations/i, hours: (m) => 5 + 0.15 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /discover (\d+) vehicles/i, hours: (m) => 5 + 0.15 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /register (\d+) vehicles/i, hours: (m) => 5 + 0.1 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /(\d+) bank vaults/i, hours: (m) => 5 + 0.2 * Number(m[1]), ceiling: 50, label: 'grind' },
  { test: /collectible card sets/i, hours: () => 20, ceiling: 40, label: 'grind' },
  { test: /meditate at every spot/i, hours: () => 15, ceiling: 40, label: 'grind' },
  { test: /gold medal in all/i, hours: () => 12, ceiling: 40, label: 'grind' },
  { test: /win .* at every location/i, hours: () => 10, ceiling: 40, label: 'grind' },
  { test: /complete (\d+) mission strands/i, hours: () => 10, ceiling: 50, label: 'grind' },
  { test: /earn \$[\d,]+/i, hours: () => 25, ceiling: 40, label: 'grind' },
  { test: /spend \$[\d,]+/i, hours: () => 15, ceiling: 40, label: 'grind' },
  { test: /purchase all/i, hours: () => 12, ceiling: 40, label: 'grind' },
  { test: /try each type of activity/i, hours: () => 2, ceiling: 70, label: 'activity' },
  { test: /complete an arcade game/i, hours: () => 0.5, ceiling: 70, label: 'activity' },
  { test: /fishing challenge/i, hours: () => 1, ceiling: 70, label: 'activity' },
  { test: /take down \d+ enemies/i, hours: () => 1, ceiling: 70, label: 'activity' },
  { test: /enjoy a cocktail/i, hours: () => 0.5, ceiling: 70, label: 'activity' },
  { test: /bowling alley/i, hours: () => 1, ceiling: 70, label: 'activity' },
  { test: /bowl 'em over/i, hours: () => 1, ceiling: 70, label: 'activity' },
  { test: /unscheduled departure/i, hours: () => 2, ceiling: 70, label: 'activity' },
  { test: /smash your way out/i, hours: () => 2, ceiling: 70, label: 'activity' },
  { test: /^rob /i, hours: () => 3, ceiling: 70, label: 'activity' },
  { test: /win (\d+) different races/i, hours: () => 4, ceiling: 70, label: 'activity' },
  { test: /lose a \d+ star wanted/i, hours: () => 2, ceiling: 70, label: 'activity' },
  { test: /hunt a unique animal/i, hours: () => 3, ceiling: 70, label: 'activity' },
  { test: /explore all of/i, hours: () => 8, ceiling: 70, label: 'activity' },
  { test: /^complete [A-Z]/i, hours: () => 8, ceiling: 80, label: 'mission' },
];

function kindOf(t: Trophy): TrophyKind {
  const d = t.description ?? '';
  // The opening mission outranks even the platinum check: a trophy whose
  // description is the opening mission is the opening mission, whatever
  // its tier. (In the real list the two never overlap; this only settles
  // the precedence.)
  if (/gone fishing/i.test(d)) return { label: 'opening', hours: 0.1, ceiling: 99.99 };
  if (t.tier === 'platinum') return { label: 'platinum', hours: 200, ceiling: 3 };
  for (const rule of KIND_RULES) {
    const m = d.match(rule.test);
    if (m !== null) return { label: rule.label, hours: rule.hours(m), ceiling: rule.ceiling };
  }
  return { label: 'other', hours: 5, ceiling: 85 };
}

/**
 * Squeeze the read history into slices: consecutive reads printing the
 * same tracked count and the same rates are one frozen page. A slice
 * ends at a refresh, dated at the midpoint between the last read of the
 * old slice and the first read of the new one; the first slice is dated
 * at its first read.
 */
function collapseHistory(c: CaseFile, releaseMs: number): Slice[] {
  const slices: Slice[] = [];
  let gapEndMs = 0;
  for (const read of c.history) {
    const ms = Date.parse(read.at);
    if (Number.isNaN(ms)) continue;
    const prev = slices[slices.length - 1];
    const frozen = prev !== undefined &&
      prev.players === read.players_tracked &&
      prev.rates.length === read.rates.length &&
      prev.rates.every((v, j) => v === read.rates[j]);
    if (frozen) {
      gapEndMs = ms;
      continue;
    }
    const refreshMs = slices.length === 0 ? ms : (gapEndMs + ms) / 2;
    slices.push({
      hours: Math.max(0.1, (refreshMs - releaseMs) / 3_600_000),
      players: read.players_tracked,
      rates: read.rates.slice(),
    });
    gapEndMs = ms;
  }
  return slices;
}

/**
 * Project the tracked-player count at the target, in hours since release.
 * With at least two distinct counts, two growth shapes are fitted to the
 * slice counts — a power law in log time and a saturating exponential —
 * and averaged in log space, so neither an exploding nor a flattening
 * curve decides alone. With one count, the launch prior takes over:
 * tracked players multiply fast in the first hours after a record
 * launch and the pace decays as the launch matures, calibrated to the
 * launch-night refresh the rules publish (638 tracked becoming 3,868,
 * about a sixfold jump over the first five hours).
 */
function projectTracked(slices: Slice[], tTarget: number): number {
  const last = slices[slices.length - 1];
  const distinct = slices.filter((s, i) => i === 0 || s.players !== slices[i - 1].players);
  if (distinct.length < 2) {
    const pace = 0.55 / (1 + tTarget / 8);
    const span = Math.max(0, tTarget - slices[0].hours);
    return last.players * Math.exp(pace * span);
  }
  // Power law: log players against log(1 + hours), slope floored at 0.
  const [aPow, bPow] = weightedLine(
    distinct.map((s) => Math.log1p(s.hours)),
    distinct.map((s) => Math.log(Math.max(1, s.players))),
    distinct.map(() => 1),
  );
  const nPow = Math.exp(aPow + Math.max(0, bPow) * Math.log1p(tTarget));
  // Saturating exponential: players = Ninf * (1 - exp(-k * hours)).
  // Scan k on a log grid; at each step Ninf comes from relative least
  // squares, and the k with the smallest relative error wins.
  let best = { sse: Infinity, ninf: last.players, k: 0.05 };
  for (let k = 0.002; k <= 1.5; k *= 1.25) {
    const u = distinct.map((s) => 1 - Math.exp(-k * s.hours));
    let suu = 0;
    let sun = 0;
    for (let j = 0; j < u.length; j++) {
      suu += u[j] * u[j];
      sun += u[j] * distinct[j].players;
    }
    const ninf = sun / Math.max(suu, 1e-12);
    if (!(ninf > 0)) continue;
    let sse = 0;
    for (let j = 0; j < u.length; j++) {
      const rel = (distinct[j].players - ninf * u[j]) / distinct[j].players;
      sse += rel * rel;
    }
    if (sse < best.sse) best = { sse, ninf, k };
  }
  const nSat = best.ninf * (1 - Math.exp(-best.k * tTarget));
  const n = Math.exp((Math.log(Math.max(1, nPow)) + Math.log(Math.max(1, nSat))) / 2);
  return clamp(n, last.players, last.players * 30);
}

/** One observation for the trend fits: a slice's rate for one trophy. */
interface Obs {
  /** Hours since release at the slice's refresh. */
  t: number;
  /** Rate in percent on that slice. */
  r: number;
  /** Binomial precision weight, clamped. */
  w: number;
}

/** A saturating-curve fit: rate rises toward a ceiling like C(1 - e^-t/H). */
interface SatFit {
  ceiling: number;
  pace: number;
  /** Residual spread of the logit rates around the fitted curve. */
  resid: number;
  n: number;
}

/** A logit-line fit: logit(rate) linear in log(1 + hours). */
interface LineFit {
  intercept: number;
  slope: number;
  /** Residual spread around the fitted line, on the logit scale. */
  resid: number;
  n: number;
  meanX: number;
  ssx: number;
}

/** Binomial precision of a rate reading: earners x (tracked - earners) / tracked. */
function precision(players: number, ratePct: number): number {
  const earners = Math.round((clampPct(ratePct) / 100) * players);
  return clamp((earners * Math.max(1, players - earners)) / Math.max(1, players), 0.5, 500);
}

function observations(slices: Slice[], idx: number): Obs[] {
  const out: Obs[] = [];
  for (const s of slices) {
    const r = s.rates[idx];
    if (r === null || r === undefined) continue;
    out.push({ t: s.hours, r: clampPct(r), w: precision(s.players, r) });
  }
  return out;
}

/**
 * The saturating fit, unshrunk: rate = C x (1 - exp(-t / H)). Scan H on
 * a log grid, solve C by weighted least squares at each step, keep the H
 * with the smallest weighted squared error in rate space. Null below two
 * usable points.
 */
function fitSaturationRaw(obs: Obs[]): { ceiling: number; pace: number } | null {
  if (obs.length < 2) return null;
  let best: { ceiling: number; pace: number; sse: number } | null = null;
  for (let h = 1; h <= 3000; h *= 1.35) {
    const u = obs.map((o) => 1 - Math.exp(-o.t / h));
    let suu = 0;
    let sur = 0;
    for (let j = 0; j < obs.length; j++) {
      suu += obs[j].w * u[j] * u[j];
      sur += obs[j].w * u[j] * obs[j].r;
    }
    if (suu < 1e-12) continue;
    const ceiling = clamp(sur / suu, 0.01, 99.99);
    let sse = 0;
    for (let j = 0; j < obs.length; j++) {
      const resid = obs[j].r - ceiling * u[j];
      sse += obs[j].w * resid * resid;
    }
    if (best === null || sse < best.sse) best = { ceiling, pace: h, sse };
  }
  return best === null ? null : { ceiling: best.ceiling, pace: best.pace };
}

/**
 * The saturating fit with shrinkage: the pace H is pulled in log space
 * toward the mean pace of trophies of the same kind (groupLogPace), then
 * the ceiling is refitted at the shrunk pace so the curve still passes
 * through the data's level. The residual spread is measured on the logit
 * scale, so it feeds the same spread machinery as the line fit.
 */
function fitSaturation(obs: Obs[], groupLogPace: number): SatFit | null {
  const raw = fitSaturationRaw(obs);
  if (raw === null) return null;
  const n = obs.length;
  const pace = Math.exp((n * Math.log(raw.pace) + 3 * groupLogPace) / (n + 3));
  const u = obs.map((o) => 1 - Math.exp(-o.t / pace));
  let suu = 0;
  let sur = 0;
  for (let j = 0; j < obs.length; j++) {
    suu += obs[j].w * u[j] * u[j];
    sur += obs[j].w * u[j] * obs[j].r;
  }
  const ceiling = suu < 1e-12 ? raw.ceiling : clamp(sur / suu, 0.01, 99.99);
  let sse = 0;
  let wsum = 0;
  for (let j = 0; j < obs.length; j++) {
    const fitted = clampPct(ceiling * u[j]);
    const resid = logitOf(obs[j].r) - logitOf(fitted);
    sse += obs[j].w * resid * resid;
    wsum += obs[j].w;
  }
  return { ceiling, pace, resid: Math.sqrt(sse / Math.max(wsum, 1e-9)), n };
}

/**
 * Weighted least squares of logit(rate) on log(1 + hours), a read's
 * weight its binomial precision. The slope is shrunk toward the mean
 * slope of trophies of the same kind (groupSlope) so a noisy slope on
 * one trophy cannot run away; the intercept is refitted at the shrunk
 * slope. Null below two usable points.
 */
function fitLogitLine(obs: Obs[], groupSlope: number): LineFit | null {
  if (obs.length < 2) return null;
  const xs = obs.map((o) => Math.log1p(o.t));
  const ys = obs.map((o) => logitOf(o.r));
  const ws = obs.map((o) => o.w);
  const [, betaRaw] = weightedLine(xs, ys, ws);
  const n = obs.length;
  const slope = (n * betaRaw + 4 * groupSlope) / (n + 4);
  // Refit the intercept at the shrunk slope, so the line still passes
  // through the data's centre.
  let sw = 0;
  let swx = 0;
  let swy = 0;
  for (let j = 0; j < n; j++) {
    sw += ws[j];
    swx += ws[j] * xs[j];
    swy += ws[j] * ys[j];
  }
  const intercept = sw > 0 ? (swy - slope * swx) / sw : 0;
  let sse = 0;
  let wsum = 0;
  for (let j = 0; j < n; j++) {
    const resid = ys[j] - (intercept + slope * xs[j]);
    sse += ws[j] * resid * resid;
    wsum += ws[j];
  }
  const meanX = sw > 0 ? swx / sw : 0;
  let ssx = 0;
  for (let j = 0; j < n; j++) ssx += ws[j] * (xs[j] - meanX) * (xs[j] - meanX);
  return { intercept, slope, resid: Math.sqrt(sse / Math.max(wsum, 1e-9)), n, meanX, ssx };
}

/**
 * The dilution forecast: when tracked players jump from N to N', the new
 * players bring earners at a fraction of the old rate. A new player has
 * typically had about an hour of play, and earning a trophy that needs H
 * hours is roughly exp(-H / 0.75) as likely for them as for a player
 * counted earlier — the launch cohort's playtime is itself roughly
 * exponential, so the ratio of the two tails is an exponential in H.
 * The opening mission is the exception: everybody finishes it in their
 * first minutes, so new players bring it at the full rate.
 */
function dilutedRate(kind: TrophyKind, lastRate: number, lastEarners: number, nLast: number, nTarget: number): number {
  if (kind.label === 'opening' || lastRate >= 99.9) return 99.99;
  // Half an earner of smoothing: a trophy nobody has earned yet can
  // still be earned by a newcomer who tries the right thing first.
  const smooth = (lastEarners + 0.5) / (nLast + 1);
  const discount = Math.exp(-kind.hours / 0.75);
  const earners = lastEarners + smooth * discount * Math.max(0, nTarget - nLast);
  return (100 * earners) / Math.max(1, nTarget);
}

/** Mean fitted slope and mean fitted log pace per kind and per tier, for shrinkage. */
function kindAverages(c: CaseFile, slices: Slice[], kinds: TrophyKind[]): Map<string, { slope: number; logPace: number }> {
  const slopes = new Map<string, number[]>();
  const logPaces = new Map<string, number[]>();
  const push = (map: Map<string, number[]>, key: string, v: number): void => {
    const arr = map.get(key) ?? [];
    arr.push(v);
    map.set(key, arr);
  };
  for (let i = 0; i < c.trophies.length; i++) {
    const obs = observations(slices, i);
    if (obs.length < 2) continue;
    const keys = [kinds[i].label, c.trophies[i].tier];
    // Unshrunk fits, just to learn the kind's typical pace.
    const line = fitLogitLine(obs, 0);
    if (line !== null) {
      // Undo the shrinkage fitLogitLine applied with groupSlope 0, so
      // the raw slope is what gets averaged.
      const rawSlope = (line.slope * (obs.length + 4)) / obs.length;
      for (const key of keys) push(slopes, key, rawSlope);
    }
    const sat = fitSaturationRaw(obs);
    if (sat !== null) {
      for (const key of keys) push(logPaces, key, Math.log(sat.pace));
    }
  }
  const mean = (arr: number[]): number => arr.reduce((s, v) => s + v, 0) / arr.length;
  const out = new Map<string, { slope: number; logPace: number }>();
  for (const key of new Set([...slopes.keys(), ...logPaces.keys()])) {
    out.set(key, {
      slope: slopes.has(key) ? mean(slopes.get(key) as number[]) : 0,
      logPace: logPaces.has(key) ? mean(logPaces.get(key) as number[]) : Math.log(40),
    });
  }
  return out;
}

/**
 * Forecast one trophy. Returns its five quantiles and the median rate.
 * `groups` holds the kind's mean slope and log pace for shrinkage;
 * `nTarget` is the forecast tracked count at the target.
 */
function forecastRow(
  c: CaseFile,
  slices: Slice[],
  kinds: TrophyKind[],
  groups: Map<string, { slope: number; logPace: number }>,
  nTarget: number,
  tTarget: number,
  idx: number,
): { q: number[]; median: number } {
  const t = c.trophies[idx];
  const kind = kinds[idx];
  const last = slices[slices.length - 1];
  // The trophy's last observed rate, scanning back over slices that
  // lacked it.
  let lastRate: number | null = null;
  let lastEarners = 0;
  for (let j = slices.length - 1; j >= 0; j--) {
    const r = slices[j].rates[idx];
    if (r !== null && r !== undefined) {
      lastRate = r;
      lastEarners = Math.round((r / 100) * slices[j].players);
      break;
    }
  }
  if (lastRate === null) lastRate = 0.5;

  // A trophy at 100% on every slice gets a hedged row: the median stays
  // at 100%, the lower quantiles admit that a flood of new players could
  // pull it down, and the hedge tightens once the trophy has survived a
  // tenfold growth in tracked players without budging.
  const alwaysFull = slices.every((s) => {
    const r = s.rates[idx];
    return r !== null && r !== undefined && r >= 99.9;
  });
  if (lastRate >= 99.9 && alwaysFull) {
    const growth = last.players / Math.max(1, slices[0].players);
    const row = growth >= 10
      ? [99.9, 99.97, 99.99, 99.99, 99.99]
      : [99.5, 99.9, 99.99, 99.99, 99.99];
    return { q: row, median: 99.99 };
  }

  const obs = observations(slices, idx);
  const group = groups.get(kind.label) ?? groups.get(t.tier) ?? { slope: 0, logPace: Math.log(40) };
  const line = fitLogitLine(obs, group.slope);
  const sat = fitSaturation(obs, group.logPace);
  const xTarget = Math.log1p(tTarget);
  const dil = dilutedRate(kind, lastRate, lastEarners, last.players, nTarget);
  // The trend reading: the saturating curve, blended three-to-one with
  // the logit line in logit space, so a trophy still climbing steeply is
  // not held back entirely by the curve, and a flattening one is not
  // dragged onward by the line.
  const trendReading = (): number | null => {
    const lin = line === null ? null : pctOfLogit(line.intercept + line.slope * xTarget);
    const satMed = sat === null ? null : clampPct(sat.ceiling * (1 - Math.exp(-tTarget / sat.pace)));
    if (lin === null) return satMed;
    if (satMed === null) return lin;
    return pctOfLogit(0.75 * logitOf(satMed) + 0.25 * logitOf(lin));
  };

  let median: number;
  let sigmaFit = 0.5;
  if (slices.length >= 3) {
    const tm = trendReading();
    if (tm !== null) {
      median = tm;
      const resid = sat !== null ? sat.resid : line !== null ? line.resid : 0.5;
      if (line !== null && line.n >= 3) {
        const leverage = 1 + 1 / line.n + ((xTarget - line.meanX) ** 2) / Math.max(line.ssx, 1e-9);
        sigmaFit = clamp(resid * Math.min(2.5, Math.sqrt(Math.max(1, leverage))), 0.15, 1.5);
      } else {
        sigmaFit = clamp(resid, 0.15, 1.5);
      }
    } else {
      median = dil;
      sigmaFit = 0.5;
    }
  } else if (slices.length === 2) {
    // One refresh seen: average the trend's reading with the dilution
    // arithmetic, in logit space, so neither decides alone.
    const tm = trendReading();
    median = tm !== null ? pctOfLogit((logitOf(tm) + logitOf(dil)) / 2) : dil;
    sigmaFit = 0.45;
  } else {
    // No trend to fit: the dilution arithmetic is all there is.
    median = dil;
    sigmaFit = 0.5;
  }

  // Guardrails: earners cannot fall, and no trophy can pass its kind's
  // generous one-week ceiling.
  const floorRate = (100 * lastEarners) / Math.max(1, nTarget);
  median = clamp(Math.max(median, floorRate), 0.01, 99.99);
  median = Math.min(median, kind.ceiling);

  // Spread on the logit scale: the fit's own residual, a random walk
  // that widens with the square root of the horizon, a refresh jump
  // proportional to the expected growth in tracked players, a small-count
  // term for trophies with only a handful of earners, and a floor.
  const sigmaRw = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  const sigmaJump = Math.sqrt((0.8 * Math.abs(Math.log(nTarget / Math.max(1, last.players)))) ** 2 + 0.35 ** 2);
  const sigmaSmall = Math.min(0.5, 0.5 / Math.sqrt(lastEarners + 1));
  const sigma = Math.min(
    3,
    Math.max(0.45, Math.sqrt(sigmaFit ** 2 + sigmaRw ** 2 + sigmaJump ** 2 + sigmaSmall ** 2 + 0.3 ** 2)),
  );
  const medLogit = logitOf(median);
  const q = LEVELS.map((level) => round4(clampPct(pctOfLogit(medLogit + normalQuantile(level) * sigma))));
  // Rounding can only ever make neighbours equal, never invert them, but
  // enforce the ordering explicitly all the same.
  for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
  return { q, median };
}

function runCase(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  // Fall back to the first read's time when the release date is missing;
  // with no history at all the origin is never used, because
  // collapseHistory returns no slices and the empty-history rows apply.
  const firstReadMs = c.history.length > 0 ? Date.parse(c.history[0].at) : NaN;
  const originMs = Number.isNaN(releaseMs) ? firstReadMs : releaseMs;
  const slices = collapseHistory(c, originMs);
  if (slices.length === 0) {
    // No history at all: say almost nothing, but say it widely.
    return {
      trophies: c.trophies.map((t) => ({ id: t.id, q: [0.01, 0.1, 1, 10, 50] })),
    };
  }
  const tTarget = Math.max(0.1, (Date.parse(c.target) - originMs) / 3_600_000);
  const nTarget = projectTracked(slices, tTarget);
  const kinds = c.trophies.map(kindOf);
  const groups = kindAverages(c, slices, kinds);
  const trophies = c.trophies.map((t, i) => {
    const { q } = forecastRow(c, slices, kinds, groups, nTarget, tTarget, i);
    return { id: t.id, q };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (!Array.isArray(c.trophies) || !Array.isArray(c.history)) {
    throw new Error(`${file} is not a trophies case`);
  }
  const out = runCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
