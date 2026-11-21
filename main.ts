// gta6-trophy-rarity-forecast-01 — a refresh-aware earner-growth forecaster.
//
// What this program does, in one paragraph: Exophase does not update its
// trophy page continuously. It refreshes a game's counts on its own schedule,
// hours apart, so the page sits frozen between refreshes and then jumps: the
// players-tracked count leaps up and (almost) every earn rate falls, because
// the newly tracked players have played far less than the ones already
// counted. A rate is therefore not a smooth series but a ratio of two moving
// counts, earners over tracked players, and the two move for different
// reasons: tracked players grow as more people buy the game and sync a
// profile, while earners grow as the players already counted keep playing.
// This program forecasts the ratio by modelling the two counts separately
// where it can and by fitting the ratio's own trend where it must, and it
// keeps its quantile bands wide enough that a refresh can move a rate a long
// way without leaving the band.
//
// How it works, step by step:
//   1. Collapse the history into plateaus: consecutive reads that print the
//      same players-tracked count and the same rates are one frozen page.
//      A plateau ends when a read prints different numbers, which is when
//      Exophase refreshed. The refresh itself is dated at the midpoint of
//      the gap between the last read of the old plateau and the first read
//      of the new one.
//   2. Recover earners per trophy per plateau: earners = rate x tracked /
//      100, rounded. Rates are printed to two decimals, so with hundreds of
//      players tracked this recovers the exact earner count, and with
//      hundreds of thousands it is still within a handful.
//   3. Forecast the players-tracked count at the target. With at least two
//      plateaus it fits two growth curves to the plateau counts — a power
//      law in time since release and a saturating exponential — and averages
//      them in log space, so neither an exploding nor a flattening curve
//      decides alone. With a single plateau (no refresh ever seen) it falls
//      back to a launch-growth prior: tracked players multiply fast in the
//      first hours after a record launch and the pace decays as the launch
//      matures. The prior's pace is calibrated to the one refresh the rules
//      publish for launch night (638 tracked players becoming 3,868).
//   4. Forecast each trophy's rate. With three or more plateaus it fits two
//      trend models and blends them: a saturating hazard curve — the rate
//      rises toward a ceiling C like C x (1 - exp(-hours / H)), which is
//      what a trophy's earn rate does as the players who can earn it
//      gradually do, and which cannot run away the way a straight line can
//      — and a line on the logit of the rate against log(1 + hours), which
//      stays useful for a trophy still in its steep phase. Both are fitted
//      by weighted least squares (a read's weight is its binomial
//      precision), both have their pace shrunk toward the mean pace of
//      trophies of the same kind, and the blend leans three-to-one toward
//      the saturating curve, because a rate that is flattening is the
//      normal case by the third day and a line extrapolated from it
//      overshoots. With two plateaus it averages that blend with the
//      dilution forecast below; with one plateau it uses the dilution
//      forecast alone, because a flat history holds no trend at all.
//   5. The dilution forecast is the arithmetic of a refresh: when tracked
//      players jump from N to N', the new players bring earners at a
//      fraction of the old rate. A new player has typically had about an
//      hour of play, so the fraction is exp(-H / 0.75) where H is the hours
//      of play a trophy usually needs — read off the trophy's own
//      description (a chapter number, a count of stunts, a dollar amount),
//      never off a hardcoded rate. The opening mission is the exception:
//      everybody finishes it in their first minutes, so new players bring
//      it at the full rate and it stays at 100%.
//   6. Guardrails: earners cannot fall, so the forecast rate cannot fall
//      below last earners over forecast tracked players; and each kind of
//      trophy has a generous ceiling for one week of play, so an
//      extrapolated trend cannot claim that 95% of players finished a
//      200-hour platinum in seven days.
//   7. Quantiles are honest. The spread on the logit scale combines the
//      trend fit's own residual (grown by how far the target is past the
//      data), a random-walk term that widens with the square root of the
//      horizon, a refresh-jump term proportional to how much the tracked
//      count is expected to grow (a refresh dilutes every rate by about the
//      log of that growth), a small-count term for trophies with only a
//      handful of earners, and a floor. A trophy that has sat at 100% at
//      every plateau gets a hedged row instead: the median stays at 100%,
//      but the lower quantiles admit that a flood of new players could
//      pull it down, and the hedge tightens once the trophy has survived a
//      tenfold growth in tracked players without budging.
//
// Everything is deterministic: no randomness, no clock reads, no network,
// no input beyond the case file. Standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
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
  history: Read[];
  notes?: string;
}

interface Plateau {
  /** Hours since release at the estimated refresh moment. */
  tHours: number;
  tracked: number;
  rates: Array<number | null>;
}

interface Kind {
  /** Story chapter, grind, activity, named mission, and so on. */
  kind: string;
  /** Hours of play a trophy of this kind usually needs. */
  hours: number;
  /** Generous ceiling, in percent, for one week of play. */
  cap: number;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -12, 12)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Inverse standard normal CDF (Acklam's rational approximation). */
function normQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Ordinary least squares of y on x, returning [intercept, slope]. */
function linfit(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return [sy / n, 0];
  const slope = (n * sxy - sx * sy) / denom;
  return [(sy - slope * sx) / n, slope];
}

/**
 * What kind of trophy this is, read off its own description and tier, and
 * how many hours of play earning it usually takes. The hours are a prior
 * about gameplay, not about rates: a chapter number, a stunt count or a
 * dollar amount in the description is all the evidence used, and the same
 * rule applies to every trophy the case lists, whatever the game.
 */
function classify(t: Trophy): Kind {
  const d = t.description ?? '';
  if (/gone fishing/i.test(d)) return { kind: 'opening', hours: 0.1, cap: 99.99 };
  if (t.tier === 'platinum') return { kind: 'platinum', hours: 200, cap: 3 };
  const chapter = d.match(/complete chapter (\d+)/i);
  if (chapter) return { kind: 'chapter', hours: 1.5 + Number(chapter[1]), cap: 90 };
  if (/final mission/i.test(d)) return { kind: 'finale', hours: 25, cap: 70 };
  if (/100% completion/i.test(d)) return { kind: 'completion', hours: 200, cap: 8 };
  let m: RegExpMatchArray | null;
  if ((m = d.match(/complete (\d+) stunts/i))) return { kind: 'grind', hours: 5 + 0.15 * Number(m[1]), cap: 50 };
  if ((m = d.match(/complete (\d+) challenges/i))) return { kind: 'grind', hours: 5 + 0.15 * Number(m[1]), cap: 50 };
  if ((m = d.match(/(\d+) locations/i))) return { kind: 'grind', hours: 5 + 0.15 * Number(m[1]), cap: 50 };
  if ((m = d.match(/discover (\d+) vehicles/i))) return { kind: 'grind', hours: 5 + 0.15 * Number(m[1]), cap: 50 };
  if ((m = d.match(/register (\d+) vehicles/i))) return { kind: 'grind', hours: 5 + 0.1 * Number(m[1]), cap: 50 };
  if ((m = d.match(/(\d+) bank vaults/i))) return { kind: 'grind', hours: 5 + 0.2 * Number(m[1]), cap: 50 };
  if (/collectible card sets/i.test(d)) return { kind: 'grind', hours: 20, cap: 40 };
  if (/meditate at every spot/i.test(d)) return { kind: 'grind', hours: 15, cap: 40 };
  if (/gold medal in all/i.test(d)) return { kind: 'grind', hours: 12, cap: 40 };
  if (/win .* at every location/i.test(d)) return { kind: 'grind', hours: 10, cap: 40 };
  if ((m = d.match(/complete (\d+) mission strands/i))) return { kind: 'grind', hours: 10, cap: 50 };
  if (/earn \$[\d,]+/i.test(d)) return { kind: 'grind', hours: 25, cap: 40 };
  if (/spend \$[\d,]+/i.test(d)) return { kind: 'grind', hours: 15, cap: 40 };
  if (/purchase all/i.test(d)) return { kind: 'grind', hours: 12, cap: 40 };
  if (/try each type of activity/i.test(d)) return { kind: 'activity', hours: 2, cap: 70 };
  if (/complete an arcade game/i.test(d)) return { kind: 'activity', hours: 0.5, cap: 70 };
  if (/fishing challenge/i.test(d)) return { kind: 'activity', hours: 1, cap: 70 };
  if (/take down \d+ enemies/i.test(d)) return { kind: 'activity', hours: 1, cap: 70 };
  if (/enjoy a cocktail/i.test(d)) return { kind: 'activity', hours: 0.5, cap: 70 };
  if (/bowling alley/i.test(d)) return { kind: 'activity', hours: 1, cap: 70 };
  if (/bowl 'em over/i.test(d)) return { kind: 'activity', hours: 1, cap: 70 };
  if (/unscheduled departure/i.test(d)) return { kind: 'activity', hours: 2, cap: 70 };
  if (/smash your way out/i.test(d)) return { kind: 'activity', hours: 2, cap: 70 };
  if (/^rob /i.test(d)) return { kind: 'activity', hours: 3, cap: 70 };
  if ((m = d.match(/win (\d+) different races/i))) return { kind: 'activity', hours: 4, cap: 70 };
  if (/lose a \d+ star wanted/i.test(d)) return { kind: 'activity', hours: 2, cap: 70 };
  if (/hunt a unique animal/i.test(d)) return { kind: 'activity', hours: 3, cap: 70 };
  if (/explore all of/i.test(d)) return { kind: 'activity', hours: 8, cap: 70 };
  if (/^complete [A-Z]/i.test(d)) return { kind: 'mission', hours: 8, cap: 80 };
  return { kind: 'other', hours: 5, cap: 85 };
}

/**
 * Collapse the history into plateaus: runs of reads that print the same
 * players-tracked count and the same rates are one frozen page. A plateau
 * ends at a refresh, dated at the midpoint between the last read of the old
 * plateau and the first read of the new one (the refresh happened somewhere
 * in that gap); the first plateau is dated at its first read.
 */
function extractPlateaus(c: CaseFile, releaseMs: number): Plateau[] {
  const out: Plateau[] = [];
  let prevEndMs = 0;
  for (const r of c.history) {
    const ms = Date.parse(r.at);
    if (Number.isNaN(ms)) continue;
    const lastPlateau = out[out.length - 1];
    const sameAsLast = lastPlateau !== undefined &&
      lastPlateau.tracked === r.players_tracked &&
      lastPlateau.rates.length === r.rates.length &&
      lastPlateau.rates.every((v, j) => v === r.rates[j]);
    if (sameAsLast) {
      prevEndMs = ms;
      continue;
    }
    const refreshMs = out.length === 0 ? ms : (prevEndMs + ms) / 2;
    out.push({
      tHours: Math.max(0.1, (refreshMs - releaseMs) / 3_600_000),
      tracked: r.players_tracked,
      rates: r.rates.slice(),
    });
    prevEndMs = ms;
  }
  return out;
}

/**
 * Forecast the players-tracked count at the target, in hours since release.
 * With at least two distinct counts it fits a power law in log time and a
 * saturating exponential and averages the two in log space; with one count
 * it uses the launch-growth prior (fast multiplication in the first hours
 * after a record launch, decaying as the launch matures), calibrated to the
 * launch-night refresh the rules publish: 638 tracked players becoming
 * 3,868, a factor of about six over the first five hours.
 */
function forecastTracked(plateaus: Plateau[], tTarget: number): number {
  const last = plateaus[plateaus.length - 1];
  const distinct = plateaus.filter((p, i) => i === 0 || p.tracked !== plateaus[i - 1].tracked);
  if (distinct.length < 2) {
    const growthPerHour = 0.55 / (1 + tTarget / 8);
    const windowHours = Math.max(0, tTarget - plateaus[0].tHours);
    return last.tracked * Math.exp(growthPerHour * windowHours);
  }
  // Power law: log tracked against log(1 + hours).
  const xs = distinct.map((p) => Math.log1p(p.tHours));
  const ys = distinct.map((p) => Math.log(Math.max(1, p.tracked)));
  const [a, b] = linfit(xs, ys);
  const slope = Math.max(0, b);
  const nPower = Math.exp(a + slope * Math.log1p(tTarget));
  // Saturating exponential: tracked = Ninf * (1 - exp(-k * hours)).
  // Grid over k, fitting Ninf by relative least squares at each step.
  let bestSse = Infinity;
  let bestNinf = last.tracked;
  let bestK = 0.05;
  for (let k = 0.002; k <= 1.5; k *= 1.25) {
    const u = distinct.map((p) => 1 - Math.exp(-k * p.tHours));
    let suu = 0;
    let sun = 0;
    for (let j = 0; j < u.length; j++) {
      suu += u[j] * u[j];
      sun += u[j] * distinct[j].tracked;
    }
    const ninf = sun / Math.max(suu, 1e-12);
    if (!(ninf > 0)) continue;
    let sse = 0;
    for (let j = 0; j < u.length; j++) {
      const rel = (distinct[j].tracked - ninf * u[j]) / distinct[j].tracked;
      sse += rel * rel;
    }
    if (sse < bestSse) {
      bestSse = sse;
      bestNinf = ninf;
      bestK = k;
    }
  }
  const nSat = bestNinf * (1 - Math.exp(-bestK * tTarget));
  const n = Math.exp((Math.log(Math.max(1, nPower)) + Math.log(Math.max(1, nSat))) / 2);
  return clamp(n, last.tracked, last.tracked * 30);
}

interface SaturatingFit {
  /** Ceiling the rate rises toward, in percent. */
  ceiling: number;
  /** Hours of play that set the pace of the rise. */
  paceHours: number;
  /** Residual spread of the logit rates around the fitted curve. */
  residLogitSd: number;
  n: number;
}

interface SatPoint {
  tau: number;
  rate: number;
  w: number;
}

/**
 * The saturating fit, unshrunk: rate = C x (1 - exp(-tau / H)). Grid over
 * H on a log scale, solving C by weighted least squares at each step, and
 * keep the H with the smallest weighted squared error in rate space.
 * Returns null with fewer than two usable points.
 */
function saturatingFitRaw(points: SatPoint[]): { ceiling: number; paceHours: number } | null {
  if (points.length < 2) return null;
  let best: { ceiling: number; paceHours: number; sse: number } | null = null;
  for (let h = 1; h <= 3000; h *= 1.35) {
    const u = points.map((p) => 1 - Math.exp(-p.tau / h));
    let suu = 0;
    let sur = 0;
    for (let j = 0; j < points.length; j++) {
      suu += points[j].w * u[j] * u[j];
      sur += points[j].w * u[j] * points[j].rate;
    }
    if (suu < 1e-12) continue;
    const ceiling = clamp(sur / suu, 0.01, 99.99);
    let sse = 0;
    for (let j = 0; j < points.length; j++) {
      const resid = points[j].rate - ceiling * u[j];
      sse += points[j].w * resid * resid;
    }
    if (best === null || sse < best.sse) best = { ceiling, paceHours: h, sse };
  }
  return best === null ? null : { ceiling: best.ceiling, paceHours: best.paceHours };
}

/**
 * The saturating fit with shrinkage: the pace H is pulled in log space
 * toward the mean pace of trophies of the same kind (groupLogH), then the
 * ceiling is refit given the shrunk pace, so the curve still passes through
 * the data's level. The residual spread is measured on the logit scale, so
 * it can feed the same spread machinery as the line fit.
 */
function saturatingFit(points: SatPoint[], groupLogH: number): SaturatingFit | null {
  const raw = saturatingFitRaw(points);
  if (raw === null) return null;
  const n = points.length;
  const logH = (n * Math.log(raw.paceHours) + 3 * groupLogH) / (n + 3);
  const paceHours = Math.exp(logH);
  const u = points.map((p) => 1 - Math.exp(-p.tau / paceHours));
  let suu = 0;
  let sur = 0;
  for (let j = 0; j < points.length; j++) {
    suu += points[j].w * u[j] * u[j];
    sur += points[j].w * u[j] * points[j].rate;
  }
  const ceiling = suu < 1e-12 ? raw.ceiling : clamp(sur / suu, 0.01, 99.99);
  let sse = 0;
  let wsum = 0;
  for (let j = 0; j < points.length; j++) {
    const fitted = clampRate(ceiling * u[j]);
    const resid = logit(points[j].rate) - logit(fitted);
    sse += points[j].w * resid * resid;
    wsum += points[j].w;
  }
  return { ceiling, paceHours, residLogitSd: Math.sqrt(sse / Math.max(wsum, 1e-9)), n };
}

interface TrendFit {
  /** Intercept of logit(rate) on log(1 + hours). */
  alpha: number;
  /** Slope, after shrinkage toward the kind's mean slope. */
  beta: number;
  /** Residual spread of the logit rates around the fitted line. */
  residSd: number;
  n: number;
  xMean: number;
  sxx: number;
}

/**
 * Weighted least squares of logit(rate) on log(1 + hours), with a read's
 * weight its binomial precision — earners x (tracked - earners) / tracked —
 * so a plateau with many players tracked counts more than one with a few.
 * The slope is then shrunk toward the mean slope of trophies of the same
 * kind (groupSlope), which keeps a noisy slope on one trophy from running
 * away; the intercept is refit given the shrunk slope. Returns null when
 * the trophy has fewer than two usable plateau points.
 */
function trendFit(
  points: Array<{ x: number; y: number; w: number }>,
  groupSlope: number,
): TrendFit | null {
  if (points.length < 2) return null;
  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;
  for (const p of points) {
    sw += p.w;
    swx += p.w * p.x;
    swy += p.w * p.y;
    swxx += p.w * p.x * p.x;
    swxy += p.w * p.x * p.y;
  }
  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-12) return null;
  const betaRaw = (sw * swxy - swx * swy) / denom;
  const alphaRaw = (swy - betaRaw * swx) / sw;
  // Shrink the slope toward the kind's mean; a trophy with few points
  // leans on its kind, one with many stands on its own.
  const n = points.length;
  const beta = (n * betaRaw + 4 * groupSlope) / (n + 4);
  // Refit the intercept given the shrunk slope, so the line still passes
  // through the data's centre.
  const alpha = (swy - beta * swx) / sw;
  let sse = 0;
  let wsum = 0;
  for (const p of points) {
    const resid = p.y - (alpha + beta * p.x);
    sse += p.w * resid * resid;
    wsum += p.w;
  }
  const residSd = Math.sqrt(sse / Math.max(wsum, 1e-9));
  const xMean = swx / sw;
  let sxx = 0;
  for (const p of points) sxx += p.w * (p.x - xMean) * (p.x - xMean);
  return { alpha, beta, residSd, n, xMean, sxx };
}

/**
 * The dilution forecast: when tracked players jump from N to N', the new
 * players bring earners at a fraction of the old rate. A new player has
 * typically had about an hour of play, and earning a trophy that needs H
 * hours of play is roughly exp(-H / 0.75) as likely for them as for a
 * player counted earlier — the launch cohort's playtime is itself roughly
 * exponential, so the ratio of the two tails is an exponential in H. The
 * opening mission is the exception: everybody finishes it in their first
 * minutes, so new players bring it at the full rate.
 */
function dilutionForecast(
  kind: Kind,
  lastRate: number,
  lastEarners: number,
  nLast: number,
  nTarget: number,
): number {
  if (kind.kind === 'opening' || lastRate >= 99.9) return 99.99;
  // Half an earner of smoothing: a trophy nobody has earned yet can still
  // be earned by a new player who tries the right thing first.
  const smoothRate = (lastEarners + 0.5) / (nLast + 1);
  const discount = Math.exp(-kind.hours / 0.75);
  const newEarners = lastEarners + smoothRate * discount * Math.max(0, nTarget - nLast);
  return (100 * newEarners) / Math.max(1, nTarget);
}

/**
 * Forecast one trophy. Returns its five quantiles and the median rate.
 * `groupSlope` is the mean fitted slope of the trophy's kind, used to
 * shrink its own slope; `nTarget` the forecast tracked count at the target.
 */
function forecastTrophy(
  c: CaseFile,
  trophies: Trophy[],
  plateaus: Plateau[],
  kinds: Kind[],
  groupStats: Map<string, { beta: number; logH: number }>,
  nTarget: number,
  tTarget: number,
  idx: number,
): { q: number[]; median: number } {
  const t = trophies[idx];
  const kind = kinds[idx];
  const last = plateaus[plateaus.length - 1];
  // The trophy's last observed rate, scanning back over plateaus that
  // lacked it.
  let lastRate: number | null = null;
  let lastEarners = 0;
  for (let j = plateaus.length - 1; j >= 0; j--) {
    const r = plateaus[j].rates[idx];
    if (r !== null && r !== undefined) {
      lastRate = r;
      lastEarners = Math.round((r / 100) * plateaus[j].tracked);
      break;
    }
  }
  if (lastRate === null) lastRate = 0.5;

  // A trophy that has sat at 100% at every plateau gets a hedged row: the
  // median stays at 100%, the lower quantiles admit that a flood of new
  // players could pull it down, and the hedge tightens once the trophy has
  // survived a tenfold growth in tracked players without budging.
  const alwaysFull = plateaus.every((p) => {
    const r = p.rates[idx];
    return r !== null && r !== undefined && r >= 99.9;
  });
  if (lastRate >= 99.9 && alwaysFull) {
    const growth = last.tracked / Math.max(1, plateaus[0].tracked);
    const row = growth >= 10
      ? [99.9, 99.97, 99.99, 99.99, 99.99]
      : [99.5, 99.9, 99.99, 99.99, 99.99];
    return { q: row, median: 99.99 };
  }

  // Points for the trend fits: one per plateau that lists the trophy.
  const points: Array<{ x: number; y: number; w: number }> = [];
  const satPoints: SatPoint[] = [];
  for (const p of plateaus) {
    const r = p.rates[idx];
    if (r === null || r === undefined) continue;
    const e = Math.round((clampRate(r) / 100) * p.tracked);
    const w = clamp((e * Math.max(1, p.tracked - e)) / Math.max(1, p.tracked), 0.5, 500);
    points.push({ x: Math.log1p(p.tHours), y: logit(r), w });
    satPoints.push({ tau: p.tHours, rate: clampRate(r), w });
  }
  const stats = groupStats.get(kind.kind) ?? groupStats.get(t.tier) ?? { beta: 0, logH: Math.log(40) };
  const fit = trendFit(points, stats.beta);
  const sat = saturatingFit(satPoints, stats.logH);
  const xTarget = Math.log1p(tTarget);
  const dil = dilutionForecast(kind, lastRate, lastEarners, last.tracked, nTarget);
  // The trend reading: the saturating curve, blended three-to-one with the
  // logit line in logit space, so a trophy still climbing steeply is not
  // held back entirely by the curve, and a flattening one is not dragged
  // onward by the line.
  const trendMedian = (): number | null => {
    const lin = fit === null ? null : invLogit(fit.alpha + fit.beta * xTarget);
    const satMed = sat === null ? null : clampRate(sat.ceiling * (1 - Math.exp(-tTarget / sat.paceHours)));
    if (lin === null) return satMed;
    if (satMed === null) return lin;
    return invLogit(0.75 * logit(satMed) + 0.25 * logit(lin));
  };

  let median: number;
  let sigmaFit = 0.5;
  if (plateaus.length >= 3) {
    const tm = trendMedian();
    if (tm !== null) {
      median = tm;
      const resid = sat !== null ? sat.residLogitSd : fit !== null ? fit.residSd : 0.5;
      if (fit !== null && fit.n >= 3) {
        const leverage = 1 + 1 / fit.n + ((xTarget - fit.xMean) ** 2) / Math.max(fit.sxx, 1e-9);
        sigmaFit = clamp(resid * Math.min(2.5, Math.sqrt(Math.max(1, leverage))), 0.15, 1.5);
      } else {
        sigmaFit = clamp(resid, 0.15, 1.5);
      }
    } else {
      median = dil;
      sigmaFit = 0.5;
    }
  } else if (plateaus.length === 2) {
    // One refresh seen: average the trend's reading with the dilution
    // arithmetic, in logit space, so neither decides alone.
    const tm = trendMedian();
    if (tm !== null) {
      median = invLogit((logit(tm) + logit(dil)) / 2);
    } else {
      median = dil;
    }
    sigmaFit = 0.45;
  } else {
    // No trend to fit: the dilution arithmetic is all there is.
    median = dil;
    sigmaFit = 0.5;
  }

  // Guardrails: earners cannot fall, and no trophy can exceed its kind's
  // generous one-week ceiling.
  const floorRate = (100 * lastEarners) / Math.max(1, nTarget);
  median = clamp(Math.max(median, floorRate), 0.01, 99.99);
  median = Math.min(median, kind.cap);

  // Spread on the logit scale: the fit's own residual, a random walk that
  // widens with the square root of the horizon, a refresh jump
  // proportional to the expected growth in tracked players, a small-count
  // term for trophies with only a handful of earners, and a floor.
  const sigmaRw = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  const sigmaJump = Math.sqrt((0.8 * Math.abs(Math.log(nTarget / Math.max(1, last.tracked)))) ** 2 + 0.35 ** 2);
  const sigmaSmall = Math.min(0.5, 0.5 / Math.sqrt(lastEarners + 1));
  const sigma = Math.min(
    3,
    Math.max(0.45, Math.sqrt(sigmaFit ** 2 + sigmaRw ** 2 + sigmaJump ** 2 + sigmaSmall ** 2 + 0.3 ** 2)),
  );
  const medLogit = logit(median);
  const q = LEVELS.map((level) => round4(clampRate(invLogit(medLogit + normQuantile(level) * sigma))));
  // Rounding can only ever make neighbours equal, never invert them, but
  // enforce the ordering explicitly all the same.
  for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
  return { q, median };
}

/** Mean fitted slope and mean fitted log pace per kind (and per tier), for shrinkage. */
function computeGroupStats(
  c: CaseFile,
  plateaus: Plateau[],
  kinds: Kind[],
): Map<string, { beta: number; logH: number }> {
  const rawSlopes = new Map<string, number[]>();
  const rawLogH = new Map<string, number[]>();
  for (let i = 0; i < c.trophies.length; i++) {
    const points: Array<{ x: number; y: number; w: number }> = [];
    const satPoints: SatPoint[] = [];
    for (const p of plateaus) {
      const r = p.rates[i];
      if (r === null || r === undefined) continue;
      const e = Math.round((clampRate(r) / 100) * p.tracked);
      const w = clamp((e * Math.max(1, p.tracked - e)) / Math.max(1, p.tracked), 0.5, 500);
      points.push({ x: Math.log1p(p.tHours), y: logit(r), w });
      satPoints.push({ tau: p.tHours, rate: clampRate(r), w });
    }
    if (points.length < 2) continue;
    const keys = [kinds[i].kind, c.trophies[i].tier];
    // Unshrunk fits, just to learn the kind's typical pace.
    const fit = trendFit(points, 0);
    if (fit !== null) {
      // Undo the shrinkage trendFit applied with groupSlope 0, so the raw
      // slope is what gets averaged: beta = (n*betaRaw + 0)/(n+4).
      const betaRaw = (fit.beta * (points.length + 4)) / points.length;
      for (const key of keys) {
        const arr = rawSlopes.get(key) ?? [];
        arr.push(betaRaw);
        rawSlopes.set(key, arr);
      }
    }
    const sat = saturatingFitRaw(satPoints);
    if (sat !== null) {
      for (const key of keys) {
        const arr = rawLogH.get(key) ?? [];
        arr.push(Math.log(sat.paceHours));
        rawLogH.set(key, arr);
      }
    }
  }
  const mean = (arr: number[]): number => arr.reduce((s, v) => s + v, 0) / arr.length;
  const out = new Map<string, { beta: number; logH: number }>();
  const keys = new Set([...rawSlopes.keys(), ...rawLogH.keys()]);
  for (const key of keys) {
    out.set(key, {
      beta: rawSlopes.has(key) ? mean(rawSlopes.get(key) as number[]) : 0,
      logH: rawLogH.has(key) ? mean(rawLogH.get(key) as number[]) : Math.log(40),
    });
  }
  return out;
}

function forecastCase(c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const originMs = Number.isNaN(releaseMs) ? Date.parse(c.history[0].at) : releaseMs;
  const plateaus = extractPlateaus(c, originMs);
  if (plateaus.length === 0) {
    // No history at all: say almost nothing, but say it widely.
    return {
      trophies: c.trophies.map((t) => ({ id: t.id, q: [0.01, 0.1, 1, 10, 50] })),
    };
  }
  const tTarget = Math.max(0.1, (Date.parse(c.target) - originMs) / 3_600_000);
  const nTarget = forecastTracked(plateaus, tTarget);
  const kinds = c.trophies.map(classify);
  const groupStats = computeGroupStats(c, plateaus, kinds);
  const trophies = c.trophies.map((t, i) => {
    const { q } = forecastTrophy(c, c.trophies, plateaus, kinds, groupStats, nTarget, tTarget, i);
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
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
