// Weather: rain / snow particles, lightning bolts and the state that blends between
// clear (晴れ), rain (雨), thunderstorm (雷雨) and snow (雪).
// Particles keep no state: each one is a hashed point moving with a constant velocity, wrapped
// into a box that travels with the camera, so they cost one instanced draw.
import { FRAME_WGSL, makeLayout, makeBindGroup, VIS } from './common.js';
import { SKY_SAMPLE_WGSL } from './sky.js';

const PARTICLE_WGSL = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var skyLut: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;
` + SKY_SAMPLE_WGSL + /* wgsl */ `
override SNOW: bool = false;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) fade: f32,
};
fn hash3(i: u32) -> vec3f {
  var x = i * 747796405u + 2891336453u;
  var v = vec3u(x, x * 1664525u + 1013904223u, x * 22695477u + 1u);
  v = (v ^ (v >> vec3u(16u))) * 2246822519u;
  v = v ^ (v >> vec3u(13u));
  return vec3f(v & vec3u(0xffffffu)) / 16777216.0;
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let h = hash3(ii);
  let h2 = hash3(ii + 7919u);
  let box = select(vec3f(36.0, 26.0, 36.0), vec3f(26.0, 18.0, 26.0), SNOW);
  let wind = vec3f(frame.windDir.x, 0.0, frame.windDir.y);
  // rain: ~9 m/s slanted by the wind; snow: ~1 m/s drifting downwind
  let vel = select(wind * 2.5 + vec3f(0.0, -9.0 - h2.x * 2.0, 0.0), wind * 1.4 + vec3f(0.0, -0.9 - h2.x * 0.5, 0.0), SNOW);
  let t = frame.cloudTime;
  var p = h * box + vel * t;
  if (SNOW) {   // flakes flutter
    p += vec3f(sin(t * 1.3 + h2.y * 40.0), 0.0, cos(t * 1.1 + h2.z * 40.0)) * 0.35;
  }
  // wrap into a box centred on the camera: the particles stay put in the world as you move
  let rel = (fract((p - frame.camPos) / box + 0.5) - 0.5) * box;
  let dist = length(rel);
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];
  var offset: vec3f;
  if (SNOW) {
    let side = normalize(cross(vec3f(0.0, 1.0, 0.0), rel) + vec3f(1e-4, 0.0, 0.0));
    let up = normalize(cross(rel, side));
    offset = (side * c.x + up * c.y) * (0.011 + 0.012 * h2.y);
  } else {
    // a streak along the motion (motion blur), turned to face the camera
    let axis = normalize(vel);
    let side = normalize(cross(axis, rel) + vec3f(1e-4, 0.0, 0.0));
    offset = axis * c.y * length(vel) * 0.035 + side * c.x * 0.009 * max(1.0, dist * 0.08);
  }
  var o: VSOut;
  o.pos = frame.viewProj * vec4f(rel + offset, 1.0);
  o.uv = c;
  let worldY = frame.camPos.y + rel.y;
  o.fade = smoothstep(select(0.4, 1.2, SNOW), select(1.5, 3.0, SNOW), dist) * (1.0 - smoothstep(box.x * 0.3, box.x * 0.5, dist)) * smoothstep(-0.3, 0.3, worldY);
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let ambient = skyIrradiance() / PI + vec3f(0.6, 0.65, 0.85) * frame.lightning * 0.6;
  let key = frame.keyColor * saturate(frame.keyDir.y + 0.2) / PI;
  var a: f32;
  var col: vec3f;
  if (SNOW) {
    a = (1.0 - smoothstep(0.2, 1.0, length(in.uv))) * 0.8 * frame.snow;
    col = (ambient + key) * 0.8;
  } else {
    // drops catch the sky light, so streaks read lighter than the sea behind them
    a = (1.0 - abs(in.uv.x)) * (1.0 - smoothstep(0.5, 1.0, abs(in.uv.y))) * 0.3 * frame.rain;
    col = (ambient * 2.2 + key * 0.3);
  }
  a *= in.fade;
  return vec4f(col * a, a);   // premultiplied alpha
}
`;

const BOLT_WGSL = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
struct VSOut { @builtin(position) pos: vec4f, @location(0) across: f32, @location(1) glow: f32 };
@vertex fn vs(@location(0) p: vec3f, @location(1) dir: vec3f, @location(2) params: vec3f) -> VSOut {
  // params: side (-1/1), width (m), brightness
  let rel = p - frame.camPos;
  let side = normalize(cross(dir, rel) + vec3f(1e-4, 0.0, 0.0));
  let width = max(params.y, length(rel) * 0.0012);   // never thinner than ~1-2 pixels
  var o: VSOut;
  o.pos = frame.viewProj * vec4f(rel + side * params.x * width, 1.0);
  o.across = params.x;
  o.glow = params.z;
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let core = exp(-in.across * in.across * 6.0);
  let c = vec3f(0.85, 0.88, 1.0) * frame.lightning * in.glow * core * 60.0;
  return vec4f(c, 0.0);
}
`;

const MAX_BOLT_VERTS = 6 * 400;
const blend = {
  premultiplied: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } },
  additive: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } },
};

// Targets for each weather type: [overcast, rain, snow, thunder]
export const WEATHER = {
  clear: { name: '晴れ', overcast: 0, rain: 0, snow: 0, thunder: 0 },
  rain: { name: '雨', overcast: 0.8, rain: 0.8, snow: 0, thunder: 0 },
  storm: { name: '雷雨', overcast: 1, rain: 1, snow: 0, thunder: 1 },
  snow: { name: '雪', overcast: 0.9, rain: 0, snow: 1, thunder: 0 },
};

export class Weather {
  constructor(device, sky, frame, hdrFormat, { samples = 4, particles = 8000 } = {}) {
    this.device = device;
    this.maxParticles = particles;
    this.state = { overcast: 0, rain: 0, snow: 0, thunder: 0 };
    this.flash = { start: -1, strokes: [], bolt: false };
    this.nextStrike = 3;
    this.lightning = 0;
    this.lightningPos = [0, 2000, 0];
    this.boltVerts = 0;

    const VF = VIS.VF;
    const pLayout = makeLayout(device, [[0, VF, 'uniform'], [1, VF, 'tex'], [2, VF, 'sampler']]);
    this.particleBG = makeBindGroup(device, pLayout, [frame.buffer, sky.lutView, sky.skySampler]);
    const pModule = device.createShaderModule({ code: PARTICLE_WGSL, label: 'weather particles' });
    const particlePipe = (snow) => device.createRenderPipeline({
      label: snow ? 'snow' : 'rain',
      layout: device.createPipelineLayout({ bindGroupLayouts: [pLayout] }),
      vertex: { module: pModule, entryPoint: 'vs', constants: { SNOW: snow ? 1 : 0 } },
      fragment: { module: pModule, entryPoint: 'fs', constants: { SNOW: snow ? 1 : 0 }, targets: [{ format: hdrFormat, blend: blend.premultiplied }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater' },
      multisample: { count: samples },
    });
    this.rainPipe = particlePipe(false);
    this.snowPipe = particlePipe(true);

    const bLayout = makeLayout(device, [[0, VF, 'uniform']]);
    this.boltBG = makeBindGroup(device, bLayout, [frame.buffer]);
    const bModule = device.createShaderModule({ code: BOLT_WGSL, label: 'lightning bolt' });
    this.boltPipe = device.createRenderPipeline({
      label: 'lightning bolt',
      layout: device.createPipelineLayout({ bindGroupLayouts: [bLayout] }),
      vertex: {
        module: bModule, entryPoint: 'vs',
        buffers: [{ arrayStride: 36, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
        ] }],
      },
      fragment: { module: bModule, entryPoint: 'fs', targets: [{ format: hdrFormat, blend: blend.additive }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater-equal' },
      multisample: { count: samples },
    });
    this.boltBuffer = device.createBuffer({ size: MAX_BOLT_VERTS * 36, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  }

  // Blend toward the selected weather and run the lightning. dt in real seconds.
  update(dt, type, camera, sunUp) {
    const target = WEATHER[type] ?? WEATHER.clear;
    const k = 1 - Math.exp(-dt / 3);   // ~3 s to settle
    for (const key of Object.keys(this.state)) this.state[key] += (target[key] - this.state[key]) * k;
    this.updateLightning(dt, camera);
    const s = this.state;
    return {
      overcast: s.overcast, rain: s.rain, snow: s.snow,
      // visibility drops in rain and much more in snow
      fogDistance: 32000 / (1 + s.rain * 5 + s.snow * 14),
      lightning: this.lightning, lightningPos: this.lightningPos,
      // overcast skies let less light through (used for the auto exposure)
      dimming: 1 - 0.35 * s.overcast,
    };
  }

  updateLightning(dt, camera) {
    this.time = (this.time || 0) + dt;
    const t = this.time;
    if (this.debugFlash !== undefined) {
      if (!this.boltVerts) this.strike(camera);
      this.flash.start = Infinity;   // freeze the strike
      this.lightning = this.debugFlash;
      return;
    }
    if (this.state.thunder > 0.5) {
      this.nextStrike -= dt;
      if (this.nextStrike <= 0) {
        this.nextStrike = 2 + Math.random() * 7;
        this.strike(camera);
      }
    }
    // a strike is a few return strokes in quick succession, each decaying in ~60 ms
    let f = 0;
    for (const s of this.flash.strokes) {
      const u = t - (this.flash.start + s.at);
      if (u >= 0) f += s.amp * Math.exp(-u / 0.06);
    }
    this.lightning = Math.min(f, 1.5) * this.state.thunder;
    if (this.flash.bolt && t - this.flash.start > 0.7) { this.flash.bolt = false; this.boltVerts = 0; }
  }

  strike(camera) {
    const t = this.time;
    const strokes = [{ at: 0, amp: 1 }];
    for (let i = 0, n = 1 + Math.floor(Math.random() * 3); i < n; i++) strokes.push({ at: 0.06 + i * 0.09 + Math.random() * 0.05, amp: 0.4 + Math.random() * 0.6 });
    // somewhere 1.5-7 km away, mostly in front of the camera
    const az = camera.yaw + (Math.random() - 0.5) * 2.2;
    const dist = 1500 + Math.random() * 5500;
    const x = camera.position[0] + Math.cos(az) * dist, z = camera.position[2] + Math.sin(az) * dist;
    const visible = Math.random() < 0.65;   // the rest flash inside the clouds
    this.lightningPos = [x, visible ? 900 : 1900, z];
    this.flash = { start: t, strokes, bolt: visible };
    if (visible) this.buildBolt([x, 1450, z], [x + (Math.random() - 0.5) * 900, 0, z + (Math.random() - 0.5) * 900]);
  }

  // Jagged channel by midpoint displacement, plus a few dimmer branches.
  buildBolt(top, bottom) {
    const verts = [];
    const channel = (a, b, depth, rough, width, glow) => {
      let pts = [a, b];
      for (let d = 0; d < depth; d++) {
        const next = [pts[0]];
        for (let i = 1; i < pts.length; i++) {
          const p = pts[i - 1], q = pts[i];
          const len = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
          const j = () => (Math.random() - 0.5) * len * rough;
          next.push([(p[0] + q[0]) / 2 + j(), (p[1] + q[1]) / 2 + j() * 0.3, (p[2] + q[2]) / 2 + j()], q);
        }
        pts = next;
      }
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i - 1], q = pts[i];
        const dir = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        for (const [pt, side] of [[p, -1], [q, -1], [q, 1], [p, -1], [q, 1], [p, 1]]) {
          if (verts.length / 9 >= MAX_BOLT_VERTS) return pts;
          verts.push(...pt, ...dir, side, width, glow);
        }
      }
      return pts;
    };
    const main = channel(top, bottom, 6, 0.55, 3, 1);
    for (let b = 0; b < 4; b++) {
      const from = main[4 + Math.floor(Math.random() * (main.length / 2))];
      const to = [from[0] + (Math.random() - 0.5) * 700, from[1] - 200 - Math.random() * 500, from[2] + (Math.random() - 0.5) * 700];
      channel(from, to, 4, 0.6, 1.5, 0.35);
    }
    this.device.queue.writeBuffer(this.boltBuffer, 0, new Float32Array(verts));
    this.boltVerts = verts.length / 9;
  }

  draw(pass) {
    const s = this.state;
    if (s.rain > 0.01) {
      pass.setPipeline(this.rainPipe);
      pass.setBindGroup(0, this.particleBG);
      pass.draw(6, Math.ceil(this.maxParticles * s.rain));
    }
    if (s.snow > 0.01) {
      pass.setPipeline(this.snowPipe);
      pass.setBindGroup(0, this.particleBG);
      pass.draw(6, Math.ceil(this.maxParticles * 1.5 * s.snow));
    }
    if (this.boltVerts > 0 && this.lightning > 0.01) {
      pass.setPipeline(this.boltPipe);
      pass.setBindGroup(0, this.boltBG);
      pass.setVertexBuffer(0, this.boltBuffer);
      pass.draw(this.boltVerts);
    }
  }

  destroy() {
    this.boltBuffer.destroy();
  }
}
