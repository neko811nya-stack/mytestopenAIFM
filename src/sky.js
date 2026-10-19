// Sky: single-scattering atmosphere (Rayleigh + Mie + ozone) baked into a sky-view LUT,
// a drifting volumetric cloud slab, and a low-res reflection environment for the ocean.
import { FRAME_WGSL, makeLayout, makeBindGroup, computePipeline, VIS } from './common.js';

export const SUN_ILLUMINANCE = 20.0;

// ---------------------------------------------------------------- WGSL
export const ATMOSPHERE_WGSL = /* wgsl */ `
const R_GROUND = 6360e3;
const R_TOP = 6460e3;
const RAY_SCAT = vec3f(5.802e-6, 13.558e-6, 33.1e-6);
const MIE_SCAT = 3.996e-6;
const MIE_EXT = 8.396e-6;
const OZONE_ABS = vec3f(0.650e-6, 1.881e-6, 0.085e-6);
const SUN_ILLUMINANCE = ${SUN_ILLUMINANCE.toFixed(2)};
const VIEW_HEIGHT = 30.0;

fn raySphere(ro: vec3f, rd: vec3f, r: f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - r * r;
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}
fn atmoDensity(h: f32) -> vec3f {
  return vec3f(exp(-h / 8000.0), exp(-h / 1200.0), max(0.0, 1.0 - abs(h - 25000.0) / 15000.0));
}
fn atmoExtinction(d: vec3f) -> vec3f { return RAY_SCAT * d.x + vec3f(MIE_EXT * d.y) + OZONE_ABS * d.z; }
fn sunTransmittanceFrom(p: vec3f, L: vec3f) -> vec3f {
  if (raySphere(p, L, R_GROUND).x > 0.0) { return vec3f(0.0); }
  let tTop = raySphere(p, L, R_TOP).y;
  let dt = tTop / 8.0;
  var od = vec3f(0.0);
  for (var i = 0; i < 8; i++) {
    let q = p + L * ((f32(i) + 0.5) * dt);
    od += atmoExtinction(atmoDensity(length(q) - R_GROUND)) * dt;
  }
  return exp(-od);
}
fn phaseRayleigh(mu: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + mu * mu); }
fn phaseMie(mu: f32, g: f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}
fn skyMarch(rdIn: vec3f, L: vec3f) -> vec3f {
  // Below the horizon we show the horizon itself (the ocean covers it).
  let rd = normalize(vec3f(rdIn.x, max(rdIn.y, 0.0), rdIn.z));
  let ro = vec3f(0.0, R_GROUND + VIEW_HEIGHT, 0.0);
  var tMax = raySphere(ro, rd, R_TOP).y;
  let gnd = raySphere(ro, rd, R_GROUND);
  if (gnd.x > 0.0) { tMax = gnd.x; }
  let mu = dot(rd, L);
  let pr = phaseRayleigh(mu);
  let pm = phaseMie(mu, 0.8);
  var T = vec3f(1.0);
  var acc = vec3f(0.0);
  let STEPS = 40;
  for (var i = 0; i < STEPS; i++) {
    let a = f32(i) / f32(STEPS);
    let b = f32(i + 1) / f32(STEPS);
    let t0 = tMax * a * a;
    let t1 = tMax * b * b;
    let dt = t1 - t0;
    let p = ro + rd * (0.5 * (t0 + t1));
    let d = atmoDensity(length(p) - R_GROUND);
    let ext = atmoExtinction(d);
    let sunT = sunTransmittanceFrom(p, L);
    let rs = RAY_SCAT * d.x;
    let ms = MIE_SCAT * d.y;
    // single scattering + a crude isotropic multiple-scattering term
    let scat = (rs * pr + vec3f(ms * pm)) * sunT + (rs + vec3f(ms)) * sunT * 0.025;
    let stepT = exp(-ext * dt);
    acc += T * (scat - scat * stepT) / max(ext, vec3f(1e-12));
    T *= stepT;
  }
  return acc * SUN_ILLUMINANCE;
}
`;

// Direction <-> sky-view LUT uv (full sphere, elevation squeezed toward the horizon).
export const SKY_SAMPLE_WGSL = /* wgsl */ `
fn skyUV(d: vec3f) -> vec2f {
  let el = asin(clamp(d.y, -1.0, 1.0));
  let x = sign(el) * sqrt(abs(el) / (0.5 * PI));
  return vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, 0.5 - 0.5 * x);
}
fn skyDir(uv: vec2f) -> vec3f {
  let x = 1.0 - 2.0 * uv.y;
  let el = sign(x) * x * x * 0.5 * PI;
  let az = (uv.x - 0.5) * 2.0 * PI;
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}
fn sampleSky(d: vec3f) -> vec3f { return textureSampleLevel(skyLut, skySampler, skyUV(d), 0.0).rgb; }
// Rough sky irradiance on an upward-facing surface.
fn skyIrradiance() -> vec3f {
  var s = sampleSky(vec3f(0.0, 1.0, 0.0));
  for (var i = 0; i < 6; i++) {
    let a = f32(i) * PI / 3.0;
    s += sampleSky(normalize(vec3f(cos(a), 0.45, sin(a)))) * 1.2;
  }
  return s / 8.2 * PI;
}
// Equirect environment over the upper hemisphere.
fn envUV(d: vec3f) -> vec2f {
  let el = asin(clamp(d.y, 0.0, 1.0));
  return vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, 1.0 - sqrt(el / (0.5 * PI)));
}
fn envDir(uv: vec2f) -> vec3f {
  let v = 1.0 - uv.y;
  let el = v * v * 0.5 * PI;
  let az = (uv.x - 0.5) * 2.0 * PI;
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}
`;

export const CLOUD_WGSL = /* wgsl */ `
const CLOUD_BOTTOM = 1400.0;
const CLOUD_TOP = 2700.0;
const CLOUD_SCALE = 1.0 / 16000.0;
const CLOUD_SIGMA = 0.03;
override LIGHT_STEPS: i32 = 4;

fn cloudShape(uv: vec2f, lod: f32) -> f32 {
  let n = textureSampleLevel(noiseTex, noiseSampler, uv, lod);
  return n.r * 0.65 + n.g * 0.35;
}
// lod: log2 of the 2D-noise texel footprint, so distant clouds are filtered instead of aliasing.
fn cloudDensity(p: vec3f, detail: bool, lod: f32) -> f32 {
  let h = saturate((p.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM));
  let wind = frame.windDir * frame.cloudTime * 15.0;
  let q = p.xz + wind;
  // the deck thins out toward the horizon, leaving a clear band for the setting sun
  let far = length(p.xz - frame.camPos.xz);
  let cov = frame.cloudCoverage * (1.0 - 0.75 * smoothstep(18000.0, 60000.0, far));
  // Dome-shaped threshold: only the densest part of the coverage map reaches the top of the slab.
  let base = cloudShape(q * CLOUD_SCALE, lod);
  let rise = smoothstep(0.0, 0.08, h);
  var d = base - (1.0 - cov) * 0.95 - h * h * 0.42 - (1.0 - rise) * 0.25;
  if (d <= 0.0) { return 0.0; }
  d = saturate(d * 5.0);
  if (detail) {
    // 3D erosion carves billows and wisps (the slab would otherwise look like extruded 2D shapes).
    let p3 = vec3f(q.x + frame.cloudTime * 4.0, p.y * 1.4, q.y) / 2600.0;
    let n = textureSampleLevel(noise3D, noiseSampler, p3, max(lod - 2.0, 0.0));
    let ero = n.r * 0.65 + n.g * 0.25 + n.b * 0.1;
    d = saturate((d - (1.0 - ero) * 0.55) / max(1.0 - (1.0 - ero) * 0.55, 0.05));
  }
  return d;
}
fn hg(mu: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}
// Returns inscattered light (rgb) and transmittance (a) of the cloud slab along a ray.
fn marchClouds(ro: vec3f, rd: vec3f, steps: i32, jitter: f32, pixelAngle: f32, ambient: vec3f, haze: vec3f) -> vec4f {
  if (rd.y < 0.01) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let t0 = (CLOUD_BOTTOM - ro.y) / rd.y;
  let t1 = (CLOUD_TOP - ro.y) / rd.y;
  let maxDist = 160000.0;
  if (t0 > maxDist) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let tEnd = min(t1, maxDist);
  let dt = (tEnd - t0) / f32(steps);
  let L = frame.sunDir;
  let mu = dot(rd, L);
  let phase = mix(hg(mu, 0.72), hg(mu, -0.2), 0.35) + hg(mu, 0.95) * 0.25;
  let sunCol = frame.sunColor * 1.25;
  // cool skylight from above for the shadowed sides, warmer bounce near the base
  let skyAmb = mix(sampleSky(vec3f(0.0, 1.0, 0.0)) * PI, ambient, 0.35);
  var T = 1.0;
  var S = vec3f(0.0);
  var t = t0 + dt * jitter;
  var tWeighted = 0.0;
  var wSum = 0.0;
  for (var i = 0; i < steps; i++) {
    let p = ro + rd * t;
    let lod = max(0.0, log2(t * pixelAngle / rd.y * 0.5 * CLOUD_SCALE * 512.0));
    let d = cloudDensity(p, true, lod);
    if (d > 0.002) {
      var od = 0.0;
      var lt = 0.0;
      for (var j = 0; j < LIGHT_STEPS; j++) {
        let seg = 70.0 * f32(j + 1) * f32(j + 1);
        lt += seg;
        od += cloudDensity(p + L * lt, j < 2, lod + f32(j)) * seg;
      }
      let beer = exp(-od * CLOUD_SIGMA);
      // multiple-scattering approximation: a softer second lobe that penetrates deeper
      let ms = exp(-od * CLOUD_SIGMA * 0.22) * 0.55;
      let powder = 1.0 - exp(-d * 6.0);
      let h = saturate((p.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM));
      let lum = sunCol * (beer * phase + ms * 0.08) * mix(0.45, 1.0, powder) + skyAmb * (0.22 + 0.4 * h) / PI;
      let stepT = exp(-d * CLOUD_SIGMA * dt);
      S += T * lum * (1.0 - stepT);
      tWeighted += t * T * (1.0 - stepT);
      wSum += T * (1.0 - stepT);
      T *= stepT;
      if (T < 0.01) { break; }
    }
    t += dt;
  }
  // aerial perspective: distant clouds dissolve into the horizon haze
  let tc = select(t0, tWeighted / max(wSum, 1e-5), wSum > 1e-5);
  let fade = exp(-tc / 38000.0);
  let alpha = 1.0 - T;
  let farFade = 1.0 - smoothstep(0.5 * maxDist, maxDist, tc);
  S = (S * fade + haze * alpha * (1.0 - fade)) * farFade;
  return vec4f(S, mix(1.0, T, farFade));
}
`;

const SKY_BINDINGS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var noiseTex: texture_2d<f32>;
@group(0) @binding(2) var noiseSampler: sampler;
@group(0) @binding(3) var skyLut: texture_2d<f32>;
@group(0) @binding(4) var skySampler: sampler;
@group(0) @binding(5) var noise3D: texture_3d<f32>;
`;

const LUT_WGSL = FRAME_WGSL + ATMOSPHERE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var lutOut: texture_storage_2d<rgba16float, write>;
fn skyDir(uv: vec2f) -> vec3f {
  let x = 1.0 - 2.0 * uv.y;
  let el = sign(x) * x * x * 0.5 * PI;
  let az = (uv.x - 0.5) * 2.0 * PI;
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(lutOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let c = skyMarch(skyDir(uv), frame.sunDir);
  textureStore(lutOut, id.xy, vec4f(c, 1.0));
}
`;

const ENV_WGSL = FRAME_WGSL + SKY_BINDINGS_WGSL + SKY_SAMPLE_WGSL + CLOUD_WGSL + /* wgsl */ `
@group(0) @binding(6) var envOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<storage, read_write> sunVis: array<f32>;
override ENV_STEPS: i32 = 18;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(envOut);
  let ambient = skyIrradiance();
  if (id.x == 0u && id.y == 0u) {
    // how much of the sun disk is blocked by clouds (drives the sun glitter on the water)
    let c = marchClouds(frame.camPos, frame.sunDir, 16, 0.5, 0.0, ambient, vec3f(0.0));
    sunVis[0] = c.a;
  }
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let d = envDir(uv);
  var col = sampleSky(d);
  let haze = sampleSky(normalize(vec3f(d.x, 0.02, d.z)));
  let c = marchClouds(frame.camPos, d, ENV_STEPS, ign(vec2f(id.xy)), 2.0 * PI / f32(size.x), ambient, haze);
  col = col * c.a + c.rgb;
  textureStore(envOut, id.xy, vec4f(col, 1.0));
}
`;

const SKY_RENDER_WGSL = FRAME_WGSL + SKY_BINDINGS_WGSL + SKY_SAMPLE_WGSL + CLOUD_WGSL + /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
override CLOUD_STEPS: i32 = 40;
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VSOut;
  o.pos = vec4f(p, 0.0, 1.0);   // depth 0 = infinitely far with reversed Z
  o.ndc = p;
  return o;
}
fn sunDisk(rd: vec3f) -> vec3f {
  let mu = dot(rd, frame.sunDir);
  let radius = 0.0095;  // ~2x the real sun for a readable disk
  let cosR = cos(radius);
  if (mu < cosR) {
    // soft glow just outside the disk
    let g = exp(-(acos(clamp(mu, -1.0, 1.0)) - radius) * 260.0);
    return frame.sunColor * g * 0.35 * frame.sunDiskScale * 0.02;
  }
  let r = acos(clamp(mu, -1.0, 1.0)) / radius;
  let limb = pow(max(1.0 - r * r, 0.0), 0.35);
  return frame.sunColor * frame.sunDiskScale * mix(0.35, 1.0, limb);
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let w = frame.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let rd = normalize(w.xyz / w.w);
  var col = sampleSky(rd);
  if (rd.y > -0.02) { col += sunDisk(rd) * smoothstep(-0.02, 0.0, rd.y); }
  let ambient = skyIrradiance();
  let haze = sampleSky(normalize(vec3f(rd.x, 0.02, rd.z)));
  let c = marchClouds(frame.camPos, rd, CLOUD_STEPS, ign(in.pos.xy), 0.75 * length(fwidth(rd)), ambient, haze);
  col = col * c.a + c.rgb;
  return vec4f(col, 1.0);
}
`;

// ---------------------------------------------------------------- CPU side

// Tileable noise texture: R = Perlin fbm, G = inverted Worley (billows), B = fine Perlin fbm, A = fine Worley.
function makeNoise(size) {
  const rng = mulberry32(4242);
  const perlin = (period) => {
    const grads = new Float32Array(period * period * 2);
    for (let i = 0; i < period * period; i++) {
      const a = rng() * Math.PI * 2;
      grads[i * 2] = Math.cos(a);
      grads[i * 2 + 1] = Math.sin(a);
    }
    return (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y);
      const fx = x - xi, fy = y - yi;
      const dot = (ix, iy, dx, dy) => {
        const g = (((iy % period) + period) % period) * period + (((ix % period) + period) % period);
        return grads[g * 2] * dx + grads[g * 2 + 1] * dy;
      };
      const u = fx * fx * fx * (fx * (fx * 6 - 15) + 10), v = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
      const a = dot(xi, yi, fx, fy), b = dot(xi + 1, yi, fx - 1, fy);
      const c = dot(xi, yi + 1, fx, fy - 1), d = dot(xi + 1, yi + 1, fx - 1, fy - 1);
      return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
    };
  };
  const worley = (period) => {
    const pts = new Float32Array(period * period * 2);
    for (let i = 0; i < pts.length; i++) pts[i] = rng();
    return (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y);
      let best = 9;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          const cx = xi + ox, cy = yi + oy;
          const g = (((cy % period) + period) % period) * period + (((cx % period) + period) % period);
          const dx = cx + pts[g * 2] - x, dy = cy + pts[g * 2 + 1] - y;
          best = Math.min(best, dx * dx + dy * dy);
        }
      }
      return Math.sqrt(best);
    };
  };
  const fbm = (fns, periods, weights, x, y) => {
    let s = 0;
    for (let i = 0; i < fns.length; i++) s += fns[i](x * periods[i], y * periods[i]) * weights[i];
    return s;
  };
  const pR = [4, 8, 16, 32], wR = [0.55, 0.27, 0.13, 0.065];
  const fnR = pR.map(perlin);
  const pG = [6, 12, 24], wG = [0.6, 0.28, 0.12];
  const fnG = pG.map(worley);
  const pB = [16, 32, 64], wB = [0.57, 0.29, 0.14];
  const fnB = pB.map(perlin);
  const pA = [24, 48], wA = [0.68, 0.32];
  const fnA = pA.map(worley);
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size, v = y / size;
      const r = 0.5 + 0.85 * fbm(fnR, pR, wR, u, v);
      const g = 1 - fbm(fnG, pG, wG, u, v) * 1.1;
      const b = 0.5 + 0.85 * fbm(fnB, pB, wB, u, v);
      const a = 1 - fbm(fnA, pA, wA, u, v) * 1.1;
      const i = (y * size + x) * 4;
      data[i] = clamp255(r); data[i + 1] = clamp255(g); data[i + 2] = clamp255(b); data[i + 3] = clamp255(a);
    }
  }
  return data;
}
const clamp255 = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Tileable 3D inverted-Worley fbm (R: 4/8/16 cells, G: 8/16/32, B: 16/32/64) for volumetric erosion.
function makeNoise3D(size) {
  const rng = mulberry32(777);
  const worley3 = (period) => {
    const pts = new Float32Array(period ** 3 * 3);
    for (let i = 0; i < pts.length; i++) pts[i] = rng();
    const w = (v) => ((v % period) + period) % period;
    return (x, y, z) => {
      const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
      let best = 9;
      for (let oz = -1; oz <= 1; oz++) for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
        const cx = xi + ox, cy = yi + oy, cz = zi + oz;
        const g = ((w(cz) * period + w(cy)) * period + w(cx)) * 3;
        const dx = cx + pts[g] - x, dy = cy + pts[g + 1] - y, dz = cz + pts[g + 2] - z;
        const d = dx * dx + dy * dy + dz * dz;
        if (d < best) best = d;
      }
      return Math.min(1, Math.sqrt(best));
    };
  };
  const octaves = [[4, 8, 16], [8, 16, 32], [16, 32, 64]].map((ps) => ps.map(worley3));
  const freqs = [[4, 8, 16], [8, 16, 32], [16, 32, 64]];
  const data = new Uint8Array(size ** 3 * 4);
  for (let z = 0; z < size; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = ((z * size + y) * size + x) * 4;
    for (let c = 0; c < 3; c++) {
      let v = 0;
      const wts = [0.625, 0.25, 0.125];
      for (let o = 0; o < 3; o++) {
        const f = freqs[c][o] / size;
        v += (1 - octaves[c][o](x * f, y * f, z * f)) * wts[o];
      }
      data[i + c] = clamp255(v);
    }
    data[i + 3] = 255;
  }
  return data;
}

function upload3D(device, data, size) {
  const levels = Math.log2(size) + 1;
  const tex = device.createTexture({
    size: [size, size, size], dimension: '3d', format: 'rgba8unorm', mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  let cur = data, s = size;
  for (let l = 0; l < levels; l++) {
    device.queue.writeTexture({ texture: tex, mipLevel: l }, cur, { bytesPerRow: s * 4, rowsPerImage: s }, [s, s, s]);
    if (s === 1) break;
    const n = s >> 1, next = new Uint8Array(n * n * n * 4);
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let o = 0; o < 8; o++) sum += cur[(((2 * z + (o >> 2)) * s + 2 * y + ((o >> 1) & 1)) * s + 2 * x + (o & 1)) * 4 + c];
      next[((z * n + y) * n + x) * 4 + c] = (sum + 4) >> 3;
    }
    cur = next; s = n;
  }
  return tex;
}

function uploadWithMips(device, data, size) {
  const levels = Math.log2(size) + 1;
  const tex = device.createTexture({
    size: [size, size], format: 'rgba8unorm', mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  let cur = data, s = size;
  for (let l = 0; l < levels; l++) {
    device.queue.writeTexture({ texture: tex, mipLevel: l }, cur, { bytesPerRow: s * 4 }, [s, s]);
    if (s === 1) break;
    const n = s >> 1, next = new Uint8Array(n * n * 4);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 4; c++) {
      const at = (xx, yy) => cur[(yy * s + xx) * 4 + c];
      next[(y * n + x) * 4 + c] = (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1) + 2) >> 2;
    }
    cur = next; s = n;
  }
  return tex;
}

// CPU copy of the transmittance integral, used for the direct sun colour.
export function sunTransmittance(sunDir, height = 30) {
  const RG = 6360e3, RT = 6460e3;
  const ray = [5.802e-6, 13.558e-6, 33.1e-6], mieExt = 8.396e-6, oz = [0.65e-6, 1.881e-6, 0.085e-6];
  const ro = [0, RG + height, 0];
  const b = ro[1] * sunDir[1];
  const sphere = (r) => { const c = ro[1] * ro[1] - r * r, d = b * b - c; return d < 0 ? null : [-b - Math.sqrt(d), -b + Math.sqrt(d)]; };
  const g = sphere(RG);
  if (g && g[0] > 0) return [0, 0, 0];
  const tTop = sphere(RT)[1];
  const steps = 64, od = [0, 0, 0];
  for (let i = 0; i < steps; i++) {
    const t = tTop * Math.pow((i + 0.5) / steps, 2), dt = tTop * (Math.pow((i + 1) / steps, 2) - Math.pow(i / steps, 2));
    const p = [sunDir[0] * t, ro[1] + sunDir[1] * t, sunDir[2] * t];
    const h = Math.hypot(p[0], p[1], p[2]) - RG;
    const dr = Math.exp(-h / 8000), dm = Math.exp(-h / 1200), dz = Math.max(0, 1 - Math.abs(h - 25000) / 15000);
    for (let c = 0; c < 3; c++) od[c] += (ray[c] * dr + mieExt * dm + oz[c] * dz) * dt;
  }
  return od.map((x) => Math.exp(-x));
}

// Noise generation is CPU work (~1 s on a phone), so keep the data across quality rebuilds.
let noiseCache = null;
function noiseData() {
  noiseCache ??= { tex2D: makeNoise(512), tex3D: makeNoise3D(64) };
  return noiseCache;
}

export class Sky {
  // q: { samples, cloudSteps, lightSteps, envSize: [w, h], envSteps }
  constructor(device, frame, hdrFormat, q) {
    this.device = device;
    this.q = q;
    const noise = noiseData();
    this.noise = uploadWithMips(device, noise.tex2D, 512);
    this.noiseView = this.noise.createView();
    this.noise3D = upload3D(device, noise.tex3D, 64);
    this.noise3DView = this.noise3D.createView();
    this.noiseSampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.skySampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear' });
    const tex = (w, h, label) => device.createTexture({ size: [w, h], format: 'rgba16float', label, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.lut = tex(256, 128, 'sky lut');
    this.env = tex(q.envSize[0], q.envSize[1], 'reflection env');
    this.lutView = this.lut.createView();
    this.envView = this.env.createView();
    this.sunVis = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });

    const C = VIS.C, F = VIS.F;
    const lutLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'storageTex']]);
    this.lutPipe = computePipeline(device, lutLayout, LUT_WGSL, 'sky lut');
    this.lutBG = makeBindGroup(device, lutLayout, [frame.buffer, this.lut.createView()]);

    const shared = (vis) => [[0, vis, 'uniform'], [1, vis, 'tex'], [2, vis, 'sampler'], [3, vis, 'tex'], [4, vis, 'sampler'], [5, vis, 'tex3d']];
    const sharedRes = [frame.buffer, this.noiseView, this.noiseSampler, this.lutView, this.skySampler, this.noise3DView];
    const envLayout = makeLayout(device, [...shared(C), [6, C, 'storageTex'], [7, C, 'storage']]);
    this.envPipe = computePipeline(device, envLayout, ENV_WGSL, 'sky env', { ENV_STEPS: q.envSteps, LIGHT_STEPS: q.lightSteps });
    this.envBG = makeBindGroup(device, envLayout, [...sharedRes, this.env.createView(), this.sunVis]);

    const renderLayout = makeLayout(device, shared(F));
    const module = device.createShaderModule({ code: SKY_RENDER_WGSL, label: 'sky render' });
    this.renderPipe = device.createRenderPipeline({
      label: 'sky',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: hdrFormat }], constants: { CLOUD_STEPS: q.cloudSteps, LIGHT_STEPS: q.lightSteps } },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' },
      multisample: { count: q.samples },
    });
    this.renderBG = makeBindGroup(device, renderLayout, sharedRes);
  }

  update(encoder) {
    const pass = encoder.beginComputePass({ label: 'sky' });
    pass.setPipeline(this.lutPipe);
    pass.setBindGroup(0, this.lutBG);
    pass.dispatchWorkgroups(256 / 8, 128 / 8);
    pass.end();
    // separate pass so the LUT writes are visible to the env pass
    const pass2 = encoder.beginComputePass({ label: 'sky env' });
    pass2.setPipeline(this.envPipe);
    pass2.setBindGroup(0, this.envBG);
    pass2.dispatchWorkgroups(Math.ceil(this.q.envSize[0] / 8), Math.ceil(this.q.envSize[1] / 8));
    pass2.end();
  }

  draw(pass) {
    pass.setPipeline(this.renderPipe);
    pass.setBindGroup(0, this.renderBG);
    pass.draw(3);
  }

  destroy() {
    for (const t of [this.noise, this.noise3D, this.lut, this.env, this.sunVis]) t.destroy();
  }
}
