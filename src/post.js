// HDR post-processing: dual-filter bloom (13-tap down / tent up) and ACES tonemapping.
import { FRAME_WGSL, makeLayout, makeBindGroup, VIS } from './common.js';

const FS_TRI = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VSOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`;

const DOWN_WGSL = FS_TRI + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
override KARIS: bool = false;
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn kw(c: vec3f) -> f32 { return 1.0 / (1.0 + luma(c)); }
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(src));
  let uv = in.uv;
  let a = textureSample(src, samp, uv + t * vec2f(-2.0, -2.0)).rgb;
  let b = textureSample(src, samp, uv + t * vec2f(0.0, -2.0)).rgb;
  let c = textureSample(src, samp, uv + t * vec2f(2.0, -2.0)).rgb;
  let d = textureSample(src, samp, uv + t * vec2f(-2.0, 0.0)).rgb;
  let e = textureSample(src, samp, uv).rgb;
  let f = textureSample(src, samp, uv + t * vec2f(2.0, 0.0)).rgb;
  let g = textureSample(src, samp, uv + t * vec2f(-2.0, 2.0)).rgb;
  let h = textureSample(src, samp, uv + t * vec2f(0.0, 2.0)).rgb;
  let i = textureSample(src, samp, uv + t * vec2f(2.0, 2.0)).rgb;
  let j = textureSample(src, samp, uv + t * vec2f(-1.0, -1.0)).rgb;
  let k = textureSample(src, samp, uv + t * vec2f(1.0, -1.0)).rgb;
  let l = textureSample(src, samp, uv + t * vec2f(-1.0, 1.0)).rgb;
  let m = textureSample(src, samp, uv + t * vec2f(1.0, 1.0)).rgb;
  var o: vec3f;
  if (KARIS) {
    // Karis average per 2x2 group tames single-pixel fireflies from the sun glitter
    let g0 = (j + k + l + m) * 0.25; let g1 = (a + b + d + e) * 0.25; let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25; let g4 = (e + f + h + i) * 0.25;
    let w0 = kw(g0) * 0.5; let w1 = kw(g1) * 0.125; let w2 = kw(g2) * 0.125; let w3 = kw(g3) * 0.125; let w4 = kw(g4) * 0.125;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  return vec4f(max(o, vec3f(0.0)), 1.0);
}
`;

const UP_WGSL = FS_TRI + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let t = 1.0 / vec2f(textureDimensions(src));
  let uv = in.uv;
  var s = textureSample(src, samp, uv).rgb * 4.0;
  s += (textureSample(src, samp, uv + vec2f(t.x, 0.0)).rgb + textureSample(src, samp, uv - vec2f(t.x, 0.0)).rgb
      + textureSample(src, samp, uv + vec2f(0.0, t.y)).rgb + textureSample(src, samp, uv - vec2f(0.0, t.y)).rgb) * 2.0;
  s += textureSample(src, samp, uv + t).rgb + textureSample(src, samp, uv - t).rgb
     + textureSample(src, samp, uv + vec2f(t.x, -t.y)).rgb + textureSample(src, samp, uv + vec2f(-t.x, t.y)).rgb;
  return vec4f(s / 16.0, 1.0);
}
`;

const COMPOSITE_WGSL = FRAME_WGSL + FS_TRI + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
fn aces(x: vec3f) -> vec3f {
  // Stephen Hill's fit of the ACES RRT+ODT
  let m1 = mat3x3f(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
  let m2 = mat3x3f(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
  let v = m1 * x;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), vec3f(0.0), vec3f(1.0));
}
fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let c = textureSample(hdr, samp, in.uv).rgb;
  let b = textureSample(bloom, samp, in.uv).rgb;
  var x = mix(c, b, frame.bloomStrength) * frame.exposure;
  // gentle vignette
  let q = in.uv - 0.5;
  x *= 1.0 - dot(q, q) * 0.35;
  var o = toSrgb(aces(x));
  o += (ign(in.pos.xy) - 0.5) / 255.0;
  return vec4f(o, 1.0);
}
`;

export class Post {
  constructor(device, frame, canvasFormat, hdrFormat) {
    this.device = device;
    this.frame = frame;
    this.hdrFormat = hdrFormat;
    this.levels = 6;
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.texLayout = makeLayout(device, [[0, VIS.F, 'tex'], [1, VIS.F, 'sampler']]);
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.texLayout] });
    const down = device.createShaderModule({ code: DOWN_WGSL, label: 'bloom down' });
    const up = device.createShaderModule({ code: UP_WGSL, label: 'bloom up' });
    const mk = (module, constants, blend) => device.createRenderPipeline({
      layout: pl,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', constants, targets: [{ format: hdrFormat, blend }] },
      primitive: { topology: 'triangle-list' },
    });
    this.downKaris = mk(down, { KARIS: 1 });
    this.down = mk(down, { KARIS: 0 });
    const add = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    this.up = mk(up, {}, add);

    this.compLayout = makeLayout(device, [[0, VIS.F, 'uniform'], [1, VIS.F, 'tex'], [2, VIS.F, 'tex'], [3, VIS.F, 'sampler']]);
    const comp = device.createShaderModule({ code: COMPOSITE_WGSL, label: 'composite' });
    this.composite = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compLayout] }),
      vertex: { module: comp, entryPoint: 'vs' },
      fragment: { module: comp, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  resize(width, height) {
    const d = this.device;
    for (const t of [this.msaa, this.depth, this.hdr, this.bloom]) t?.destroy();
    const RA = GPUTextureUsage.RENDER_ATTACHMENT, TB = GPUTextureUsage.TEXTURE_BINDING;
    this.msaa = d.createTexture({ size: [width, height], format: this.hdrFormat, sampleCount: 4, usage: RA });
    this.depth = d.createTexture({ size: [width, height], format: 'depth32float', sampleCount: 4, usage: RA });
    this.hdr = d.createTexture({ size: [width, height], format: this.hdrFormat, usage: RA | TB });
    const bw = Math.max(1, width >> 1), bh = Math.max(1, height >> 1);
    this.levels = Math.max(1, Math.min(6, Math.floor(Math.log2(Math.min(bw, bh))) - 1));
    this.bloom = d.createTexture({ size: [bw, bh], format: this.hdrFormat, mipLevelCount: this.levels, usage: RA | TB });
    this.msaaView = this.msaa.createView();
    this.depthView = this.depth.createView();
    this.hdrView = this.hdr.createView();
    this.bloomMips = [...Array(this.levels)].map((_, l) => this.bloom.createView({ baseMipLevel: l, mipLevelCount: 1 }));
    const bg = (view) => makeBindGroup(d, this.texLayout, [view, this.sampler]);
    this.downBGs = [bg(this.hdrView), ...this.bloomMips.slice(0, -1).map(bg)];
    this.upBGs = this.bloomMips.map(bg);
    this.compBG = makeBindGroup(d, this.compLayout, [this.frame.buffer, this.hdrView, this.bloomMips[0], this.sampler]);
  }

  beginScene(encoder) {
    return encoder.beginRenderPass({
      label: 'scene',
      colorAttachments: [{ view: this.msaaView, resolveTarget: this.hdrView, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: this.depthView, depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 0 },
    });
  }

  finish(encoder, canvasView) {
    const run = (pipe, bg, target, load) => {
      const p = encoder.beginRenderPass({ colorAttachments: [{ view: target, loadOp: load ? 'load' : 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.draw(3);
      p.end();
    };
    for (let l = 0; l < this.levels; l++) run(l === 0 ? this.downKaris : this.down, this.downBGs[l], this.bloomMips[l], false);
    for (let l = this.levels - 1; l > 0; l--) run(this.up, this.upBGs[l], this.bloomMips[l - 1], true);
    run(this.composite, this.compBG, canvasView, false);
  }
}
