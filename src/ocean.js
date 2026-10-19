// FFT ocean simulation on WebGPU compute shaders.
//  1. JONSWAP + directional spreading spectrum -> h0(k)       (only when params change)
//  2. Time evolution h(k,t) and its derivative spectra        (every frame)
//  3. 2D inverse FFT (radix-2 Stockham, 2 x log2(N) passes)   (every frame)
//  4. Assemble displacement / derivative textures + foam      (every frame)
//  5. Mip chain for filtered sampling                         (every frame)
// All cascades are processed in a single dispatch (global_invocation_id.z = cascade).
import { makeLayout, makeBindGroup, computePipeline, VIS } from './common.js';
import { lgamma } from './math.js';

const SIM_WGSL = /* wgsl */ `
struct SimParams {
  N: u32, cascades: u32, time: f32, dt: f32,
  windDir: vec2f, windSpeed: f32, fetch: f32,
  gamma: f32, spread: f32, spreadNorm: f32, choppy: f32,
  foamDecay: f32, foamThreshold: f32, foamInject: f32, seed: u32,
  cascadeL: vec4f, kLo: vec4f, kHi: vec4f,
  amplitude: f32, loopPeriod: f32, alpha: f32, omegaP: f32,
};
const TAU = 6.28318530718;
const G = 9.81;
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn conj(a: vec2f) -> vec2f { return vec2f(a.x, -a.y); }
fn mulI(a: vec2f) -> vec2f { return vec2f(-a.y, a.x); }
fn signedIndex(i: u32, N: u32) -> f32 { return select(f32(i), f32(i) - f32(N), i >= N / 2u); }
`;

const SPECTRUM_WGSL = SIM_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4f>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand01(state: ptr<function, u32>) -> f32 {
  *state = pcg(*state);
  return (f32(*state >> 8u) + 0.5) / 16777216.0;
}
fn gaussian2(state: ptr<function, u32>) -> vec2f {
  let u1 = rand01(state);
  let u2 = rand01(state);
  let r = sqrt(-2.0 * log(u1));
  return vec2f(r * cos(TAU * u2), r * sin(TAU * u2));
}
fn jonswap(omega: f32) -> f32 {
  let wp = P.omegaP;
  let sigma = select(0.09, 0.07, omega <= wp);
  let d = omega - wp;
  let r = exp(-d * d / (2.0 * sigma * sigma * wp * wp));
  return P.alpha * G * G / pow(omega, 5.0) * exp(-1.25 * pow(wp / omega, 4.0)) * pow(P.gamma, r);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let N = P.N;
  if (id.x >= N || id.y >= N) { return; }
  let c = id.z;
  let idx = c * N * N + id.y * N + id.x;
  let L = P.cascadeL[c];
  let dk = TAU / L;
  let k = vec2f(signedIndex(id.x, N), signedIndex(id.y, N)) * dk;
  let kl = length(k);
  var state = pcg(idx * 1664525u + pcg(P.seed + c * 7919u));
  let xi = gaussian2(&state);
  var h0 = vec2f(0.0);
  var omega = 0.0;
  if (kl > 1e-6 && kl >= P.kLo[c] && kl < P.kHi[c]) {
    omega = sqrt(G * kl);
    let dOmegaDk = G / (2.0 * omega);
    let theta = atan2(k.y, k.x) - atan2(P.windDir.y, P.windDir.x);
    let D = P.spreadNorm * pow(max(abs(cos(0.5 * theta)), 1e-5), 2.0 * P.spread);
    // S(kx,kz) = S(omega) * domega/dk / k * D(theta)
    let S = jonswap(omega) * dOmegaDk / kl * D;
    let damp = exp(-kl * kl * 1e-4);  // remove sub-centimetre capillaries
    h0 = xi * sqrt(max(S, 0.0) * dk * dk * 0.25) * P.amplitude * damp;
  }
  // Quantize the dispersion so the whole simulation loops seamlessly every loopPeriod seconds.
  let w0 = TAU / P.loopPeriod;
  omega = floor(omega / w0) * w0;
  spectrum[idx] = vec4f(h0, omega, 0.0);
}
`;

const TIME_WGSL = SIM_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> spectrum: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outA: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outB: array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let N = P.N;
  if (id.x >= N || id.y >= N) { return; }
  let c = id.z;
  let base = c * N * N;
  let idx = base + id.y * N + id.x;
  let idxM = base + ((N - id.y) % N) * N + ((N - id.x) % N);
  let s = spectrum[idx];
  let h0k = s.xy;
  let h0mk = spectrum[idxM].xy;
  let phase = s.z * P.time;
  let e = vec2f(cos(phase), sin(phase));
  let h = cmul(h0k, e) + cmul(conj(h0mk), conj(e));

  let k = vec2f(signedIndex(id.x, N), signedIndex(id.y, N)) * (TAU / P.cascadeL[c]);
  let kl = length(k);
  let invK = select(0.0, 1.0 / kl, kl > 1e-6);
  let ih = mulI(h);
  let Dx = ih * (k.x * invK);
  let Dz = ih * (k.y * invK);
  let hx = ih * k.x;
  let hz = ih * k.y;
  let Dxx = h * (-k.x * k.x * invK);
  let Dzz = h * (-k.y * k.y * invK);
  let Dxz = h * (-k.x * k.y * invK);
  // Two real fields per complex FFT: C = A + iB  ->  ifft(C) = a + i b
  outA[idx] = vec4f(Dx + mulI(h), Dz + mulI(Dxx));
  outB[idx] = vec4f(hx + mulI(hz), Dzz + mulI(Dxz));
}
`;

const FFT_WGSL = SIM_WGSL + /* wgsl */ `
struct Stage { Ns: u32, dir: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> S: Stage;
@group(0) @binding(1) var<uniform> P: SimParams;
@group(0) @binding(2) var<storage, read> inA: array<vec4f>;
@group(0) @binding(3) var<storage, read> inB: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outA: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outB: array<vec4f>;

fn at(base: u32, line: u32, pos: u32) -> u32 {
  return select(base + pos * P.N + line, base + line * P.N + pos, S.dir == 0u);
}
fn twiddle(v: vec4f, w: vec2f) -> vec4f { return vec4f(cmul(v.xy, w), cmul(v.zw, w)); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let N = P.N;
  let half = N / 2u;
  let j = id.x;
  if (j >= half || id.y >= N) { return; }
  let base = id.z * N * N;
  let line = id.y;
  let Ns = S.Ns;
  let k = j & (Ns - 1u);
  let angle = 3.14159265359 * f32(k) / f32(Ns);   // +sign: inverse transform
  let w = vec2f(cos(angle), sin(angle));
  let i0 = at(base, line, j);
  let i1 = at(base, line, j + half);
  let o = (j / Ns) * Ns * 2u + k;
  let o0 = at(base, line, o);
  let o1 = at(base, line, o + Ns);
  let a0 = inA[i0];
  let a1 = twiddle(inA[i1], w);
  outA[o0] = a0 + a1;
  outA[o1] = a0 - a1;
  let b0 = inB[i0];
  let b1 = twiddle(inB[i1], w);
  outB[o0] = b0 + b1;
  outB[o1] = b0 - b1;
}
`;

const ASSEMBLE_WGSL = SIM_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> inA: array<vec4f>;
@group(0) @binding(2) var<storage, read> inB: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> foam: array<f32>;
@group(0) @binding(4) var dispOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var derivOut: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let N = P.N;
  if (id.x >= N || id.y >= N) { return; }
  let c = id.z;
  let idx = c * N * N + id.y * N + id.x;
  let a = inA[idx];
  let b = inB[idx];
  let lambda = P.choppy;
  let Dx = a.x; let h = a.y; let Dz = a.z; let Dxx = a.w;
  let hx = b.x; let hz = b.y; let Dzz = b.z; let Dxz = b.w;
  // Jacobian of the horizontal displacement: < 1 means the surface is compressing (crest), < 0 folds.
  let J = (1.0 + lambda * Dxx) * (1.0 + lambda * Dzz) - lambda * lambda * Dxz * Dxz;
  var f = foam[idx] * exp(-P.foamDecay * P.dt);
  f = max(f, clamp((P.foamThreshold - J) * P.foamInject, 0.0, 1.0));
  foam[idx] = f;
  textureStore(dispOut, vec2u(id.xy), c, vec4f(lambda * Dx, h, lambda * Dz, f));
  textureStore(derivOut, vec2u(id.xy), c, vec4f(hx, hz, lambda * Dxx, lambda * Dzz));
}
`;

const MIP_WGSL = /* wgsl */ `
@group(0) @binding(0) var srcA: texture_2d_array<f32>;
@group(0) @binding(1) var srcB: texture_2d_array<f32>;
@group(0) @binding(2) var dstA: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var dstB: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dstA);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let s = vec2i(id.xy) * 2;
  let c = i32(id.z);
  let a = textureLoad(srcA, s, c, 0) + textureLoad(srcA, s + vec2i(1, 0), c, 0) + textureLoad(srcA, s + vec2i(0, 1), c, 0) + textureLoad(srcA, s + vec2i(1, 1), c, 0);
  let b = textureLoad(srcB, s, c, 0) + textureLoad(srcB, s + vec2i(1, 0), c, 0) + textureLoad(srcB, s + vec2i(0, 1), c, 0) + textureLoad(srcB, s + vec2i(1, 1), c, 0);
  textureStore(dstA, vec2u(id.xy), c, a * 0.25);
  textureStore(dstB, vec2u(id.xy), c, b * 0.25);
}
`;

export class OceanSimulation {
  constructor(device, { N = 256, lengths = [256, 48, 9] } = {}) {
    this.device = device;
    this.N = N;
    this.lengths = lengths;
    this.cascades = lengths.length;
    this.log2N = Math.round(Math.log2(N));
    this.mipLevels = this.log2N + 1;
    this.params = {
      windSpeed: 9, windAngle: 0, fetch: 120000, gamma: 3.3, spread: 6, choppy: 1.1, amplitude: 1.0,
      foamDecay: 0.45, foamThreshold: 0.77, foamInject: 5.0, seed: 1337, loopPeriod: 600,
    };
    this.spectrumDirty = true;
    this.time = 0;

    const texels = N * N * this.cascades;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    this.paramBuffer = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.spectrum = device.createBuffer({ size: texels * 16, usage: storage });
    this.bufA = [0, 1].map(() => device.createBuffer({ size: texels * 16, usage: storage }));
    this.bufB = [0, 1].map(() => device.createBuffer({ size: texels * 16, usage: storage }));
    this.foam = device.createBuffer({ size: texels * 4, usage: storage });

    const texDesc = {
      size: [N, N, this.cascades], format: 'rgba16float', mipLevelCount: this.mipLevels,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    };
    this.displacement = device.createTexture({ ...texDesc, label: 'ocean displacement' });
    this.derivatives = device.createTexture({ ...texDesc, label: 'ocean derivatives' });
    this.displacementView = this.displacement.createView({ dimension: '2d-array' });
    this.derivativesView = this.derivatives.createView({ dimension: '2d-array' });

    const C = VIS.C;
    const specLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'storage']]);
    const timeLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'read'], [2, C, 'storage'], [3, C, 'storage']]);
    const fftLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'uniform'], [2, C, 'read'], [3, C, 'read'], [4, C, 'storage'], [5, C, 'storage']]);
    const asmLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'read'], [2, C, 'read'], [3, C, 'storage'], [4, C, 'storageTexArray'], [5, C, 'storageTexArray']]);
    const mipLayout = makeLayout(device, [[0, C, 'texArray'], [1, C, 'texArray'], [2, C, 'storageTexArray'], [3, C, 'storageTexArray']]);

    this.specPipe = computePipeline(device, specLayout, SPECTRUM_WGSL, 'spectrum');
    this.timePipe = computePipeline(device, timeLayout, TIME_WGSL, 'time evolution');
    this.fftPipe = computePipeline(device, fftLayout, FFT_WGSL, 'fft');
    this.asmPipe = computePipeline(device, asmLayout, ASSEMBLE_WGSL, 'assemble');
    this.mipPipe = computePipeline(device, mipLayout, MIP_WGSL, 'ocean mips');

    this.specBG = makeBindGroup(device, specLayout, [this.paramBuffer, this.spectrum]);
    this.timeBG = makeBindGroup(device, timeLayout, [this.paramBuffer, this.spectrum, this.bufA[0], this.bufB[0]]);

    // 2 * log2(N) Stockham stages, ping-ponging between buffer sets 0 and 1 (ends back in set 0).
    this.fftBGs = [];
    this.stageBuffers = [];
    for (let s = 0; s < 2 * this.log2N; s++) {
      const stage = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
      new Uint32Array(stage.getMappedRange()).set([1 << (s % this.log2N), s < this.log2N ? 0 : 1, 0, 0]);
      stage.unmap();
      this.stageBuffers.push(stage);
      const src = s % 2, dst = 1 - src;
      this.fftBGs.push(makeBindGroup(device, fftLayout, [stage, this.paramBuffer, this.bufA[src], this.bufB[src], this.bufA[dst], this.bufB[dst]]));
    }

    const level = (tex, l) => tex.createView({ dimension: '2d-array', baseMipLevel: l, mipLevelCount: 1 });
    this.asmBG = makeBindGroup(device, asmLayout, [this.paramBuffer, this.bufA[0], this.bufB[0], this.foam, level(this.displacement, 0), level(this.derivatives, 0)]);
    this.mipBGs = [];
    for (let l = 1; l < this.mipLevels; l++) {
      this.mipBGs.push(makeBindGroup(device, mipLayout, [level(this.displacement, l - 1), level(this.derivatives, l - 1), level(this.displacement, l), level(this.derivatives, l)]));
    }
  }

  destroy() {
    for (const r of [this.paramBuffer, this.spectrum, ...this.bufA, ...this.bufB, this.foam, this.displacement, this.derivatives, ...this.stageBuffers]) r.destroy();
  }

  setParams(p) {
    const spectral = ['windSpeed', 'windAngle', 'fetch', 'gamma', 'spread', 'amplitude', 'seed'];
    for (const [k, v] of Object.entries(p)) {
      if (this.params[k] === v) continue;
      this.params[k] = v;
      if (spectral.includes(k)) this.spectrumDirty = true;
    }
  }

  // Cascade i covers the wavenumber band [kLo_i, kHi_i) so that no wavelength is counted twice.
  bands() {
    const lo = this.lengths.map((L, i) => (i === 0 ? 1e-4 : (2 * Math.PI / L) * 6));
    const hi = lo.map((_, i) => (i + 1 < lo.length ? lo[i + 1] : 1e9));
    return { lo, hi };
  }

  writeParams(dt) {
    const p = this.params;
    const g = 9.81, U = Math.max(p.windSpeed, 0.5), F = p.fetch;
    const alpha = 0.076 * Math.pow((U * U) / (F * g), 0.22);
    const omegaP = 22 * Math.pow((g * g) / (U * F), 1 / 3);
    const s = p.spread;
    const spreadNorm = Math.exp((2 * s - 1) * Math.log(2) - Math.log(Math.PI) + 2 * lgamma(s + 1) - lgamma(2 * s + 1));
    const { lo, hi } = this.bands();
    const buf = new ArrayBuffer(128);
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    u[0] = this.N; u[1] = this.cascades; f[2] = this.time % p.loopPeriod; f[3] = dt;
    f[4] = Math.cos(p.windAngle); f[5] = Math.sin(p.windAngle); f[6] = U; f[7] = F;
    f[8] = p.gamma; f[9] = s; f[10] = spreadNorm; f[11] = p.choppy;
    f[12] = p.foamDecay; f[13] = p.foamThreshold; f[14] = p.foamInject; u[15] = p.seed;
    for (let i = 0; i < 4; i++) {
      f[16 + i] = this.lengths[i] ?? 1;
      f[20 + i] = lo[i] ?? 0;
      f[24 + i] = hi[i] ?? 0;
    }
    f[28] = p.amplitude; f[29] = p.loopPeriod; f[30] = alpha; f[31] = omegaP;
    this.device.queue.writeBuffer(this.paramBuffer, 0, buf);
  }

  update(encoder, time, dt) {
    this.time = time;
    this.writeParams(dt);
    const N = this.N, C = this.cascades;
    const pass = encoder.beginComputePass({ label: 'ocean simulation' });
    if (this.spectrumDirty) {
      pass.setPipeline(this.specPipe);
      pass.setBindGroup(0, this.specBG);
      pass.dispatchWorkgroups(N / 8, N / 8, C);
      this.spectrumDirty = false;
    }
    pass.setPipeline(this.timePipe);
    pass.setBindGroup(0, this.timeBG);
    pass.dispatchWorkgroups(N / 8, N / 8, C);

    pass.setPipeline(this.fftPipe);
    for (const bg of this.fftBGs) {
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(N / 2 / 64), N, C);
    }

    pass.setPipeline(this.asmPipe);
    pass.setBindGroup(0, this.asmBG);
    pass.dispatchWorkgroups(N / 8, N / 8, C);

    pass.setPipeline(this.mipPipe);
    for (let l = 0; l < this.mipBGs.length; l++) {
      const size = Math.max(1, N >> (l + 1));
      pass.setBindGroup(0, this.mipBGs[l]);
      pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), C);
    }
    pass.end();
  }
}
