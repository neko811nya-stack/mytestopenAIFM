// A procedural sailboat that floats on the simulated sea.
// Buoyancy runs on the GPU: a one-thread compute pass samples the displacement cascades under the
// hull, eases heave / pitch / roll toward them with spring-dampers and writes the boat transform
// into a storage buffer that the boat's vertex shader and the ocean shader (clipping, foam) read.
import { FRAME_WGSL, makeLayout, makeBindGroup, computePipeline, VIS } from './common.js';
import { SKY_SAMPLE_WGSL } from './sky.js';

// Hull shape, shared by the mesh builder (JS) and the ocean shader (WGSL) so the clipped water
// matches the hull. Boat space: +x bow, +y up, +z port; y = 0 is the design waterline.
const STERN = -3.6, BOW = 4.0;
function hullFrame(x) {
  const u = (x - STERN) / (BOW - STERN);
  const w = Math.max(u - 0.4, 0) / 0.6;
  const beam = u < 0.4 ? 0.95 + 0.4 * Math.sin((u / 0.4) * Math.PI / 2) : 1.35 * (1 - Math.pow(w, 1.8));
  return { beam, top: 0.9 + 0.35 * w * w, bottom: -0.72 + 0.6 * w * w };
}
const sectionWidth = (t) => Math.pow(t, 0.45);  // V-shaped bottom flaring to the gunwale

export const BOAT_WGSL = /* wgsl */ `
struct BoatState {
  pos: vec4f,   // xyz = waterline origin in world space, w = 1 when the boat is shown
  ax: vec4f, ay: vec4f, az: vec4f,  // boat-to-world basis (bow, up, port)
  s0: vec4f,    // heave, heave velocity, pitch, pitch velocity
  s1: vec4f,    // roll, roll velocity, initialised flag, unused
  s2: vec4f,    // smoothed horizontal drift (xz)
};
const BOAT_STERN = ${STERN.toFixed(2)};
const BOAT_BOW = ${BOW.toFixed(2)};
fn boatHalfWidthAtWaterline(x: f32) -> f32 {
  if (x <= BOAT_STERN || x >= BOAT_BOW) { return 0.0; }
  let u = (x - BOAT_STERN) / (BOAT_BOW - BOAT_STERN);
  let w = max(u - 0.4, 0.0) / 0.6;
  let beam = select(1.35 * (1.0 - pow(w, 1.8)), 0.95 + 0.4 * sin(u / 0.4 * PI * 0.5), u < 0.4);
  let top = 0.9 + 0.35 * w * w;
  let bottom = -0.72 + 0.6 * w * w;
  let t = saturate((0.0 - bottom) / (top - bottom));
  return beam * pow(t, 0.45);
}
fn boatLocal(b: BoatState, world: vec3f) -> vec3f {
  let d = world - b.pos.xyz;
  return vec3f(dot(d, b.ax.xyz), dot(d, b.ay.xyz), dot(d, b.az.xyz));
}
// Approximate distance (m) from a boat-space point to the waterline outline; negative inside.
fn boatWaterlineDistance(p: vec3f) -> f32 {
  let x = clamp(p.x, BOAT_STERN + 0.01, BOAT_BOW - 0.01);
  let dz = abs(p.z) - boatHalfWidthAtWaterline(x);
  let dx = max(max(p.x - BOAT_BOW, BOAT_STERN - p.x), 0.0);
  return select(dz, length(vec2f(dx, max(dz, 0.0))), dx > 0.0);
}
`;

const PHYSICS_WGSL = FRAME_WGSL + BOAT_WGSL + /* wgsl */ `
struct BoatParams { base: vec2f, yaw: f32, dt: f32, enabled: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> P: BoatParams;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var dispTex: texture_2d_array<f32>;
@group(0) @binding(3) var oceanSampler: sampler;
@group(0) @binding(4) var<storage, read_write> boat: BoatState;

// The hull averages out short ripples: big cascade at full detail, medium one blurred, tiny one skipped.
fn water(xz: vec2f) -> vec3f {
  return textureSampleLevel(dispTex, oceanSampler, xz / frame.cascadeL[0], 0, 0.0).xyz
       + textureSampleLevel(dispTex, oceanSampler, xz / frame.cascadeL[1], 1, 3.0).xyz;
}
fn spring(x: ptr<function, f32>, v: ptr<function, f32>, goal: f32, k: f32, zeta: f32, dt: f32) {
  *v += (k * (goal - *x) - 2.0 * zeta * sqrt(k) * *v) * dt;
  *x += *v * dt;
}

@compute @workgroup_size(1)
fn main() {
  let dt = min(P.dt, 0.05);
  let fwd = vec2f(cos(P.yaw), sin(P.yaw));
  let side = vec2f(-fwd.y, fwd.x);
  // the hull rides the water like a floating particle: probe points are Lagrangian coordinates
  let c = water(P.base);
  let hBow = water(P.base + fwd * 3.0).y;
  let hStern = water(P.base - fwd * 3.0).y;
  let hPort = water(P.base + side * 1.2).y;
  let hStbd = water(P.base - side * 1.2).y;
  let heaveT = (c.y * 2.0 + hBow + hStern + hPort + hStbd) / 6.0;
  let pitchT = atan2(hBow - hStern, 6.0);
  let rollT = -atan2(hPort - hStbd, 2.4) * 1.2;

  var y = boat.s0.x; var vy = boat.s0.y; var pitch = boat.s0.z; var vp = boat.s0.w;
  var roll = boat.s1.x; var vr = boat.s1.y;
  var drift = boat.s2.xy;
  if (boat.s1.z < 0.5) {
    y = heaveT; pitch = pitchT; roll = rollT; vy = 0.0; vp = 0.0; vr = 0.0; drift = c.xz;
  } else {
    spring(&y, &vy, heaveT, 10.0, 0.55, dt);
    spring(&pitch, &vp, pitchT, 7.0, 0.45, dt);
    spring(&roll, &vr, rollT, 5.0, 0.25, dt);   // lightly damped: boats keep rolling
    drift = mix(drift, c.xz, 1.0 - exp(-dt * 2.0));
  }
  boat.s0 = vec4f(y, vy, pitch, vp);
  boat.s1 = vec4f(roll, vr, 1.0, 0.0);
  boat.s2 = vec4f(drift, 0.0, 0.0);

  // basis = yaw * pitch (about the beam) * roll (about the keel)
  let cy = cos(P.yaw); let sy = sin(P.yaw);
  let ry = mat3x3f(vec3f(cy, 0.0, sy), vec3f(0.0, 1.0, 0.0), vec3f(-sy, 0.0, cy));
  let cp = cos(pitch); let sp = sin(pitch);
  let rz = mat3x3f(vec3f(cp, sp, 0.0), vec3f(-sp, cp, 0.0), vec3f(0.0, 0.0, 1.0));
  let cr = cos(roll); let sr = sin(roll);
  let rx = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, cr, sr), vec3f(0.0, -sr, cr));
  let m = ry * rz * rx;
  boat.pos = vec4f(P.base.x + drift.x, y - 0.08, P.base.y + drift.y, P.enabled);
  boat.ax = vec4f(m[0], 0.0);
  boat.ay = vec4f(m[1], 0.0);
  boat.az = vec4f(m[2], 0.0);
}
`;

const RENDER_WGSL = FRAME_WGSL + BOAT_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> boat: BoatState;
@group(0) @binding(2) var envTex: texture_2d<f32>;
@group(0) @binding(3) var skySampler: sampler;
@group(0) @binding(4) var skyLut: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> sunVis: array<f32>;
` + SKY_SAMPLE_WGSL + /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec4f,
};
@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) color: vec4f) -> VSOut {
  let m = mat3x3f(boat.ax.xyz, boat.ay.xyz, boat.az.xyz);
  var o: VSOut;
  o.rel = boat.pos.xyz + m * p - frame.camPos;
  o.pos = frame.viewProj * vec4f(o.rel, 1.0);
  o.normal = m * n;
  o.color = color;
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let dist = length(in.rel);
  let V = -in.rel / dist;
  var n = normalize(in.normal);
  let sail = in.color.a > 0.5;
  if (sail && dot(n, V) < 0.0) { n = -n; }   // sails, boom and stay are single-sided geometry
  let L = frame.sunDir;
  let sunCol = frame.sunColor * sunVis[0];
  let ambient = skyIrradiance();
  let albedo = in.color.rgb;

  // the sunset horizon makes the sky irradiance very bright; the hull and sails occlude half of it
  var col = albedo / PI * (sunCol * saturate(dot(n, L)) + ambient * (0.22 + 0.2 * n.y));
  if (sail) {
    // thin canvas glows when the low sun shines through it
    let through = pow(saturate(dot(-V, L)), 3.0) * 0.6 + 0.12;
    col += albedo / PI * sunCol * saturate(-dot(n, L) * 0.5 + 0.5) * through;
  } else {
    // glossy paint: a little sky reflection and a sun highlight
    let R = reflect(-V, n);
    let F = 0.04 + 0.96 * pow(1.0 - saturate(dot(n, V)), 5.0);
    let env = textureSampleLevel(envTex, skySampler, envUV(normalize(vec3f(R.x, abs(R.y), R.z))), 0.0).rgb;
    let H = normalize(L + V);
    let spec = pow(saturate(dot(n, H)), 120.0) * 4.0 * saturate(dot(n, L));
    col += F * env * 0.6 + sunCol * spec * 0.05;
  }
  let hazeCol = sampleSky(normalize(vec3f(-V.x, 0.015, -V.z)));
  col = mix(col, hazeCol, 1.0 - exp(-dist / 32000.0));
  return vec4f(col, 1.0);
}
`;

// ---------------------------------------------------------------- mesh

function buildMesh() {
  const out = [];
  // `outward` is a direction the face normal must point toward (sails are flipped in the shader instead)
  const tri = (a, b, c, color, sail = 0, outward = null) => {
    let n = normalOf(a, b, c);
    if (outward && n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2] < 0) n = n.map((v) => -v);
    for (const p of [a, b, c]) out.push(...p, ...n, ...color, sail);
  };
  const quad = (a, b, c, d, color, sail = 0, outward = null) => { tri(a, b, c, color, sail, outward); tri(a, c, d, color, sail, outward); };
  const box = (x0, x1, y0, y1, z0, z1, color) => {
    const p = (x, y, z) => [x, y, z];
    quad(p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1), p(x1, y1, z0), color, 0, [0, 1, 0]);
    quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1), color, 0, [0, -1, 0]);
    quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1), color, 0, [0, 0, 1]);
    quad(p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0), p(x1, y0, z0), color, 0, [0, 0, -1]);
    quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1), color, 0, [1, 0, 0]);
    quad(p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1), p(x0, y1, z0), color, 0, [-1, 0, 0]);
  };

  const WHITE = [0.82, 0.82, 0.78], RED = [0.42, 0.07, 0.05], NAVY = [0.04, 0.07, 0.16];
  const TEAK = [0.42, 0.27, 0.15], CREAM = [0.86, 0.83, 0.74], GLASS = [0.03, 0.04, 0.05];
  const METAL = [0.55, 0.56, 0.58], SAIL = [0.93, 0.9, 0.84];

  // hull: smooth lofted surface, both sides
  const NX = 48, NT = 14;
  const hullPoint = (i, j, side) => {
    const x = STERN + (BOW - STERN) * (i / NX);
    const f = hullFrame(x);
    const t = j / NT;
    return [x, f.bottom + (f.top - f.bottom) * t, side * f.beam * sectionWidth(t)];
  };
  const hullColor = (y) => (y < 0.1 ? RED : y < 0.24 ? NAVY : WHITE);
  const grid = [];
  for (const side of [1, -1]) {
    const pts = [];
    for (let i = 0; i <= NX; i++) for (let j = 0; j <= NT; j++) pts.push(hullPoint(i, j, side));
    const normals = pts.map(() => [0, 0, 0]);
    const faces = [];
    for (let i = 0; i < NX; i++) for (let j = 0; j < NT; j++) {
      const a = i * (NT + 1) + j, b = a + NT + 1;
      const fs = side > 0 ? [[a, b, b + 1], [a, b + 1, a + 1]] : [[a, b + 1, b], [a, a + 1, b + 1]];
      for (const f of fs) {
        const n = normalOf(pts[f[0]], pts[f[1]], pts[f[2]]);
        if (!Number.isFinite(n[0])) continue;  // degenerate triangles at the bow tip
        faces.push(f);
        for (const k of f) for (let c = 0; c < 3; c++) normals[k][c] += n[c];
      }
    }
    for (const f of faces) {
      for (const k of f) {
        const nn = normalize3(normals[k]);
        out.push(...pts[k], ...nn, ...hullColor(pts[k][1]), 0);
      }
    }
    grid.push(pts);
  }

  // transom and deck
  const sternTop = hullFrame(STERN);
  for (let j = 0; j < NT; j++) {
    const a = hullPoint(0, j, 1), b = hullPoint(0, j + 1, 1), c = hullPoint(0, j + 1, -1), d = hullPoint(0, j, -1);
    quad(a, d, c, b, WHITE, 0, [-1, 0, 0]);
  }
  for (let i = 0; i < NX; i++) {
    const a = hullPoint(i, NT, 1), b = hullPoint(i + 1, NT, 1), c = hullPoint(i + 1, NT, -1), d = hullPoint(i, NT, -1);
    quad(a, d, c, b, TEAK, 0, [0, 1, 0]);
  }

  // cabin with dark windows, cockpit coaming
  const deckY = sternTop.top;
  box(-1.9, 0.6, deckY - 0.05, deckY + 0.62, -0.78, 0.78, CREAM);
  box(-1.75, 0.45, deckY + 0.62, deckY + 0.7, -0.7, 0.7, CREAM);
  for (const z of [0.785, -0.835]) box(-1.5, 0.2, deckY + 0.25, deckY + 0.45, z, z + 0.05, GLASS);
  box(0.6, 0.66, deckY - 0.05, deckY + 0.4, -0.7, 0.7, GLASS);

  // rig: mast, boom, sails trimmed ~16 degrees off the centreline
  const mastX = 0.9, trim = Math.tan(16 * Math.PI / 180);
  box(mastX - 0.07, mastX + 0.07, deckY + 0.6, 9.4, -0.07, 0.07, METAL);
  const aft = (x, y) => [x, y, -(mastX - x) * trim];
  const boomY = deckY + 1.15;
  quad(aft(mastX, boomY - 0.05), aft(-2.9, boomY - 0.05), aft(-2.9, boomY + 0.07), aft(mastX, boomY + 0.07), METAL, 1);
  tri(aft(mastX - 0.08, boomY + 0.1), aft(mastX - 0.08, 9.1), aft(-2.85, boomY + 0.1), SAIL, 1);
  const jibTrim = Math.tan(12 * Math.PI / 180);
  tri([BOW - 0.25, deckY + 0.45, 0], [mastX + 0.12, 8.0, 0], [mastX + 0.4, deckY + 0.6, 0.0 - (BOW - mastX) * jibTrim * 0.5], SAIL, 1);
  // forestay
  quad([BOW - 0.25, deckY + 0.35, -0.02], [mastX + 0.1, 8.6, -0.02], [mastX + 0.1, 8.6, 0.02], [BOW - 0.25, deckY + 0.35, 0.02], METAL, 1);
  return new Float32Array(out);
}

function normalOf(a, b, c) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  return normalize3([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]);
}
function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

// ---------------------------------------------------------------- GPU objects

export class Boat {
  constructor(device, sim, sky, frame, hdrFormat, { samples = 4, base = [58, 9], yaw = 125 * Math.PI / 180 } = {}) {
    this.device = device;
    this.base = base;
    this.yaw = yaw;
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.state = device.createBuffer({ size: 112, usage: GPUBufferUsage.STORAGE });

    const verts = buildMesh();
    this.vertexCount = verts.length / 10;
    this.vbo = device.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
    new Float32Array(this.vbo.getMappedRange()).set(verts);
    this.vbo.unmap();

    const C = VIS.C, VF = VIS.VF, F = VIS.F;
    const physLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'uniform'], [2, C, 'texArray'], [3, C, 'sampler'], [4, C, 'storage']]);
    this.physPipe = computePipeline(device, physLayout, PHYSICS_WGSL, 'boat physics');
    const sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.physBG = makeBindGroup(device, physLayout, [this.params, frame.buffer, sim.displacementView, sampler, this.state]);

    const layout = makeLayout(device, [[0, VF, 'uniform'], [1, VF, 'read'], [2, F, 'tex'], [3, F, 'sampler'], [4, F, 'tex'], [5, F, 'read']]);
    this.bindGroup = makeBindGroup(device, layout, [frame.buffer, this.state, sky.envView, sky.skySampler, sky.lutView, sky.sunVis]);
    const module = device.createShaderModule({ code: RENDER_WGSL, label: 'boat' });
    this.pipeline = device.createRenderPipeline({
      label: 'boat',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module, entryPoint: 'vs',
        buffers: [{
          arrayStride: 40,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x4' },
          ],
        }],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: samples },
    });
  }

  update(encoder, dt, enabled) {
    this.enabled = enabled;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.base[0], this.base[1], this.yaw, dt, enabled ? 1 : 0, 0, 0, 0]));
    const pass = encoder.beginComputePass({ label: 'boat physics' });
    pass.setPipeline(this.physPipe);
    pass.setBindGroup(0, this.physBG);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  draw(pass) {
    if (!this.enabled) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbo);
    pass.draw(this.vertexCount);
  }

  destroy() {
    for (const b of [this.params, this.state, this.vbo]) b.destroy();
  }
}
