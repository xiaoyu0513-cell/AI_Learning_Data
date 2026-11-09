"use client";

export type Palette = {
  primary: string; // hex
  secondary: string; // hex
};

const STORAGE_KEY = "recruitment-accent-theme";

type HSL = { h: number; s: number; l: number };

/* ===== 颜色工具 ===== */
function hexToRgb(hex: string) {
  const m = hex.replace("#", "");
  if (m.length === 3) {
    const [r, g, b] = m.split("").map((c) => parseInt(c + c, 16));
    return { r, g, b };
  }
  const n = parseInt(m, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbToHsl(r: number, g: number, b: number): HSL {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return { h: h * 60, s, l };
}

function hslToHex(h: number, s: number, l: number): string {
  h = ((h % 360) + 360) % 360;
  s = Math.min(1, Math.max(0, s));
  l = Math.min(1, Math.max(0, l));
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const to = (v: number) =>
    Math.round((v + m) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/* ===== 色相直方图取色 ===== */
export function extractPalette(image: HTMLImageElement): Palette {
  const MAX = 64;
  const scale = Math.min(1, MAX / Math.max(image.naturalWidth, image.naturalHeight));
  const w = Math.max(1, Math.round(image.naturalWidth * scale));
  const h = Math.max(1, Math.round(image.naturalHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas not supported");
  ctx.drawImage(image, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  type Bucket = { weight: number; r: number; g: number; b: number; hue: number; sat: number };
  const buckets: Bucket[] = Array.from({ length: 36 }, (_, i) => ({
    weight: 0,
    r: 0,
    g: 0,
    b: 0,
    hue: i * 10,
    sat: 0,
  }));

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    if (a < 128) continue;
    const { h: hue, s, l } = rgbToHsl(r, g, b);
    if (s < 0.12 || l < 0.08 || l > 0.92) continue; // 跳过灰、近黑、近白
    const weight = s * (1 - Math.abs(l - 0.5) * 1.3);
    const bi = Math.floor(hue / 10) % 36;
    buckets[bi].weight += weight;
    buckets[bi].r += r * weight;
    buckets[bi].g += g * weight;
    buckets[bi].b += b * weight;
    buckets[bi].sat = Math.max(buckets[bi].sat, s);
  }

  const sorted = buckets.filter((b) => b.weight > 0).sort((a, b) => b.weight - a.weight);
  if (sorted.length === 0) {
    return { primary: "#6366f1", secondary: "#d946ef" }; // 兜底：默认 indigo/fuchsia
  }

  const toHex = (bk: Bucket) => {
    const hsl = rgbToHsl(bk.r / bk.weight, bk.g / bk.weight, bk.b / bk.weight);
    // 提饱和、把明度收敛到适合做按钮/强调色的区间，保证观感统一
    const s = clamp(hsl.s, 0.45, 0.8);
    const l = clamp(hsl.l, 0.42, 0.62);
    return hslToHex(hsl.h, s, l);
  };

  const primary = toHex(sorted[0]);
  const pHue = sorted[0].hue;

  let secondaryBucket = sorted.find(
    (b) => Math.min(Math.abs(b.hue - pHue), 360 - Math.abs(b.hue - pHue)) >= 50,
  );
  if (!secondaryBucket) {
    // 没有足够远的第二色 → 用主色偏移 45° 生成一个协调的辅助色
    const { h: hh, s, l } = rgbToHsl(
      sorted[0].r / sorted[0].weight,
      sorted[0].g / sorted[0].weight,
      sorted[0].b / sorted[0].weight,
    );
    const secondary = hslToHex(hh + 45, clamp(s, 0.4, 0.75), clamp(l, 0.45, 0.6));
    return { primary, secondary };
  }

  const secondary = toHex(secondaryBucket);
  return { primary, secondary };
}

/* ===== 应用到 CSS 变量 ===== */
export function applyPalette(palette: Palette) {
  const root = document.documentElement;
  const p1 = hexToRgb(palette.primary);
  const { h: h1, s: s1, l: l1 } = rgbToHsl(p1.r, p1.g, p1.b);
  const p2 = hexToRgb(palette.secondary);
  const { h: h2, s: s2, l: l2 } = rgbToHsl(p2.r, p2.g, p2.b);

  const L = clamp(l1, 0.42, 0.62);
  const s500 = clamp(s1, 0.45, 0.8);
  const s2c = clamp(s2, 0.45, 0.8);

  const st = {
    "--color-accent-300": hslToHex(h1, s500, Math.min(L + 0.2, 0.9)),
    "--color-accent-400": hslToHex(h1, s500, Math.min(L + 0.1, 0.82)),
    "--color-accent-500": hslToHex(h1, s500, L),
    "--color-accent-600": hslToHex(h1, s500, Math.max(L - 0.09, 0.22)),
    "--color-accent": hslToHex(h1, s500, L),
    "--color-accent-2": hslToHex(h2, s2c, clamp(l2, 0.45, 0.6)),
  };

  const ambient = {
    "--ambient-1": `hsla(${h1}, ${Math.round(s500 * 100)}%, ${Math.round(L * 100)}%, 0.18)`,
    "--ambient-2": `hsla(${h2}, ${Math.round(s2c * 100)}%, ${Math.round(clamp(l2, 0.45, 0.6) * 100)}%, 0.14)`,
    "--ambient-3": `hsla(${h1}, ${Math.round(s500 * 100)}%, ${Math.round(L * 100)}%, 0.1)`,
  };

  for (const [k, v] of Object.entries({ ...st, ...ambient })) {
    root.style.setProperty(k, v);
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(palette));
  } catch {
    // 忽略存储失败
  }
}

const THEME_VARS = [
  "--color-accent-300",
  "--color-accent-400",
  "--color-accent-500",
  "--color-accent-600",
  "--color-accent",
  "--color-accent-2",
  "--ambient-1",
  "--ambient-2",
  "--ambient-3",
];

export function resetTheme() {
  const root = document.documentElement;
  for (const v of THEME_VARS) root.style.removeProperty(v);
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // 忽略
  }
}

export function loadStoredTheme() {
  if (typeof window === "undefined") return;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) applyPalette(JSON.parse(raw) as Palette);
  } catch {
    // 忽略损坏缓存
  }
}