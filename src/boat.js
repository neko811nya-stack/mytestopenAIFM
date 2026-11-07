// A procedural sailboat floating on the simulated sea, as a rigid body simulated on the GPU.
// Each frame a one-thread compute pass:
//   - samples the wave field under ~24 hull columns (height and water particle velocity),
//   - applies Archimedes buoyancy and drag relative to the moving water at every column,
//     gravity with a low ballast centre, and wind pressure on the sails,
//   - lets water that washes over the deck, or a breaking crest, flood the hull (adding weight until it sinks),
//   - integrates position / quaternion with sub-steps and writes the boat transform.
// The boat's vertex shader and the ocean shader (clipping, foam, wake) read that state.
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

// Buoyancy columns: vertical prisms from the hull bottom to the deck, 3 across x 8 along.
const MAX_COLUMNS = 32;
function hullColumns() {
  const cols = [];
  const stations = 8, x0 = STERN + 0.25, x1 = BOW - 0.35, dx = (x1 - x0) / (stations - 1);
  for (let i = 0; i < stations; i++) {
    const x = x0 + dx * i;
    const f = hullFrame(x);
    for (const k of [-1, 0, 1]) {
      const z = k * f.beam * 0.6;
      const t = Math.pow(Math.min(Math.abs(z) / f.beam, 1), 1 / 0.45);
      cols.push([x, z, f.bottom + (f.top - f.bottom) * t, f.top, dx * (2 * f.beam / 3)]);
    }
  }
  return cols;
}

export const BOAT_WGSL = /* wgsl */ `
struct BoatState {
  pos: vec4f,     // xyz = body origin (design waterline) in world space, w = 1 when shown
  ax: vec4f, ay: vec4f, az: vec4f,  // boat-to-world basis (bow, up, port)
  vel: vec4f,     // linear velocity, w = flooding 0..1
  angVel: vec4f,  // angular velocity (world), w = splash strength
  rot: vec4f,     // orientation quaternion
  misc: vec4f,    // x initialised, y time spent sunk, z sailing course, w speed through water
  prev: array<vec4f, ${MAX_COLUMNS}>,  // last frame's water displacement under each column
};
const BOAT_STERN = ${STERN.toFixed(2)};
const BOAT_BOW = ${BOW.toFixed(2)};
fn boatHullTop(x: f32) -> f32 {
  let w = max((x - BOAT_STERN) / (BOAT_BOW - BOAT_STERN) - 0.4, 0.0) / 0.6;
  return 0.9 + 0.35 * w * w;
}
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
struct BoatParams {
  base: vec2f, yaw0: f32, dt: f32,
  enabled: f32, windSpeed: f32, mass: f32, speed: f32,
  reset: f32, columns: f32, pad0: f32, pad1: f32,
  manual: f32, throttle: f32, rudder: f32, pad2: f32,   // player at the helm: throttle / rudder in -1..1
};
struct Column { a: vec4f, b: vec4f };  // a = (x, z, bottom y, top y), b.x = horizontal area
@group(0) @binding(0) var<uniform> P: BoatParams;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var dispTex: texture_2d_array<f32>;
@group(0) @binding(3) var oceanSampler: sampler;
@group(0) @binding(4) var<storage, read_write> boat: BoatState;
@group(0) @binding(5) var<storage, read> cols: array<Column>;
@group(0) @binding(6) var derivTex: texture_2d_array<f32>;

const G = 9.81;
const RHO = 1025.0;
const SUBSTEPS = 6;
const FLOOD_MASS = 15000.0;      // water the hull can take on before it is completely full
const COURSE_RADIUS = 140.0;     // sailing mode circles the start point so the boat stays in view
const KNOT = 0.5144;

// Displacement of the water surface (largest two cascades: the hull averages out ripples).
fn disp(xz: vec2f) -> vec3f {
  return textureSampleLevel(dispTex, oceanSampler, xz / frame.cascadeL[0], 0, 0.0).xyz
       + textureSampleLevel(dispTex, oceanSampler, xz / frame.cascadeL[1], 1, 2.0).xyz;
}
// The field is stored per undisplaced position: invert the horizontal displacement once.
fn waterAt(xz: vec2f) -> vec3f {
  let d0 = disp(xz);
  return disp(xz - d0.xz);
}
// How hard the water here is breaking: the surface folds over where the horizontal
// Jacobian of the big + medium waves collapses (the same signal that makes whitecaps).
fn breaking(xz: vec2f) -> f32 {
  let a = textureSampleLevel(derivTex, oceanSampler, xz / frame.cascadeL[0], 0, 0.0);
  let b = textureSampleLevel(derivTex, oceanSampler, xz / frame.cascadeL[1], 1, 1.0);
  let J = (1.0 + a.z + b.z) * (1.0 + a.w + b.w);
  return saturate((0.55 - J) * 2.5);
}
fn qmul(a: vec4f, b: vec4f) -> vec4f { return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz)); }
fn qrot(q: vec4f, v: vec3f) -> vec3f { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
fn wrapAngle(a: f32) -> f32 { return a - 2.0 * PI * floor((a + PI) / (2.0 * PI)); }

fn initialise() {
  let q = vec4f(0.0, sin(-P.yaw0 * 0.5), 0.0, cos(-P.yaw0 * 0.5));
  let w = waterAt(P.base);
  boat.pos = vec4f(P.base.x, w.y, P.base.y, P.enabled);
  boat.vel = vec4f(0.0);
  boat.angVel = vec4f(0.0);
  boat.rot = q;
  boat.misc = vec4f(1.0, 0.0, P.yaw0, 0.0);
  for (var i = 0; i < ${MAX_COLUMNS}; i++) { boat.prev[i] = vec4f(0.0); }
}

@compute @workgroup_size(1)
fn main() {
  if (boat.misc.x < 0.5 || P.reset > 0.5) { initialise(); }
  let dt = clamp(P.dt, 1e-4, 0.05);
  let n = min(u32(P.columns), ${MAX_COLUMNS}u);

  var pos = boat.pos.xyz;
  var vel = boat.vel.xyz;
  var flood = boat.vel.w;
  var omega = boat.angVel.xyz;
  var splash = boat.angVel.w * exp(-dt * 1.8);
  var q = boat.rot;
  var course = boat.misc.z;
  let mass = P.mass + flood * FLOOD_MASS;
  let massScale = mass / 3000.0;
  // body-frame inertia (roll about the keel, yaw, pitch); the mast adds to roll
  let invI = 1.0 / (vec3f(5200.0, 16000.0, 15000.0) * massScale);
  let comLocal = vec3f(0.0, -0.55, 0.0);  // ballast keel keeps the centre of mass low

  // ---- water under each column (sampled once per frame; columns barely move in 1/60 s)
  var water: array<vec4f, ${MAX_COLUMNS}>;     // xyz = water velocity, w = surface height
  var breakers: array<f32, ${MAX_COLUMNS}>;
  var meanFlow = vec3f(0.0);
  for (var i = 0u; i < n; i++) {
    let c = cols[i];
    let p = pos + qrot(q, vec3f(c.a.x, 0.0, c.a.y));
    let w = waterAt(p.xz);
    let prev = boat.prev[i];
    let flow = select(vec3f(0.0), (w - prev.xyz) / dt, prev.w > 0.5);
    water[i] = vec4f(clamp(flow, vec3f(-15.0), vec3f(15.0)), w.y);
    breakers[i] = breaking(p.xz);
    boat.prev[i] = vec4f(w, 1.0);
    meanFlow += water[i].xyz / f32(n);
  }

  let h = dt / f32(SUBSTEPS);
  for (var s = 0; s < SUBSTEPS; s++) {
    let ax = qrot(q, vec3f(1.0, 0.0, 0.0));
    let ay = qrot(q, vec3f(0.0, 1.0, 0.0));
    var F = vec3f(0.0, -mass * G, 0.0);
    var T = cross(qrot(q, comLocal), F);

    for (var i = 0u; i < n; i++) {
      let c = cols[i];
      let wb = pos + qrot(q, vec3f(c.a.x, c.a.z, c.a.y));
      let wt = pos + qrot(q, vec3f(c.a.x, c.a.w, c.a.y));
      let lo = min(wb.y, wt.y);
      let hi = max(wb.y, wt.y);
      let surface = water[i].w;
      let sub = clamp(surface - lo, 0.0, hi - lo);
      if (sub <= 0.0) { continue; }
      let frac = sub / max(hi - lo, 0.05);
      let pt = mix(wb, wt, saturate((lo + 0.5 * sub - wb.y) / (wt.y - wb.y + 1e-4)));
      let r = pt - pos;
      // Archimedes: weight of the displaced water column
      var f = vec3f(0.0, RHO * G * c.b.x * sub, 0.0);
      // drag against the moving water: little along the hull, a lot sideways (keel) and vertically
      let rel = vel + cross(omega, r) - water[i].xyz;
      let relLocal = vec3f(dot(rel, ax), dot(rel, ay), dot(rel, qrot(q, vec3f(0.0, 0.0, 1.0))));
      let k = vec3f(120.0, 1500.0, 2600.0) * c.b.x * frac * (1.0 + 0.35 * length(relLocal));
      let dragLocal = -k * relLocal;
      f += ax * dragLocal.x + ay * dragLocal.y + qrot(q, vec3f(0.0, 0.0, 1.0)) * dragLocal.z;
      F += f;
      T += cross(r, f);
      // green water over the deck floods the hull
      if (surface > hi && ay.y > 0.2) { flood += (surface - hi) * c.b.x * 0.05 * h; }
      // a breaking crest slams into the topsides (shoving and rolling the hull) and pours aboard
      let brk = breakers[i];
      if (brk > 0.0) {
        let hit = vec3f(frame.windDir.x, 0.0, frame.windDir.y) * (brk * 14000.0 * c.b.x) + vec3f(0.0, -brk * 3000.0 * c.b.x, 0.0);
        let at = pos + qrot(q, vec3f(c.a.x, c.a.w, c.a.y)) - pos;
        F += hit;
        T += cross(at, hit);
        flood += brk * c.b.x * 0.02 * h;
        if (s == 0) { splash = max(splash, brk); }
      }
      // the bow slamming into a rising wave throws spray
      if (c.a.x > 1.5 && s == 0) {
        let impact = water[i].y - (vel.y + cross(omega, r).y);
        splash = max(splash, saturate((impact - 1.2) * 0.35));
      }
    }

    // wind pressure on ~16 m^2 of sail, centred 4.8 m up; spills as the boat heels
    let windDir = vec3f(frame.windDir.x, 0.0, frame.windDir.y);
    let upright = max(ay.y, 0.0);
    let fw = windDir * (0.5 * 1.225 * 1.1 * 16.0 * P.windSpeed * P.windSpeed * upright * upright);
    F += fw;
    T += cross(qrot(q, vec3f(-0.6, 4.8, 0.0)), fw);

    // helm: player control, hold the mooring (speed 0) or sail a big circle.
    // A positive torque about +y turns the bow toward decreasing yaw.
    let yaw = atan2(ax.z, ax.x);
    var targetYaw = P.yaw0;
    let fwdFlat = normalize(vec3f(ax.x, 0.0, ax.z) + vec3f(1e-5));
    let through = dot(vel - meanFlow, fwdFlat);   // speed through the water
    if (P.manual > 0.5) {
      // engine: full ahead ~8 kn, astern at 35 %
      let thrust = select(P.throttle * 0.35, P.throttle, P.throttle > 0.0) * 5200.0;
      F += fwdFlat * thrust * massScale * upright;
      // the rudder needs water flowing past it: little bite when stopped, reversed when going astern
      let bite = clamp(through / 1.5, -1.0, 1.0) * (0.4 + 0.6 * min(abs(through) / 3.0, 1.0)) + 0.15 * P.throttle;
      T.y += (-P.rudder * bite * 16000.0 - omega.y * 9000.0) * massScale * upright;
      course = yaw;
    } else if (P.speed > 0.05) {
      course += P.speed * KNOT / COURSE_RADIUS * h;
      targetYaw = course;
      F += fwdFlat * clamp((P.speed * KNOT - through) * 3000.0 * massScale, -9000.0, 9000.0) * upright;
      T.y += (wrapAngle(yaw - targetYaw) * 9000.0 - omega.y * 14000.0) * massScale * upright;
    } else {
      course = P.yaw0;
      let off = pos.xz - P.base;
      let pull = -350.0 * off;
      let pullLen = length(pull);
      let moor = select(pull, pull * (3500.0 / pullLen), pullLen > 3500.0);  // gentle even when far from the anchor
      F += vec3f(moor.x - 900.0 * vel.x, 0.0, moor.y - 900.0 * vel.z) * massScale;
      T.y += (wrapAngle(yaw - targetYaw) * 9000.0 - omega.y * 14000.0) * massScale * upright;
    }

    // integrate (semi-implicit Euler); inertia rotated into world space
    vel += F / mass * h;
    pos += vel * h;
    let tl = vec3f(dot(T, ax), dot(T, ay), dot(T, qrot(q, vec3f(0.0, 0.0, 1.0))));
    let al = tl * invI;
    omega += (ax * al.x + ay * al.y + qrot(q, vec3f(0.0, 0.0, 1.0)) * al.z) * h;
    omega *= exp(-0.3 * h);
    q = normalize(q + 0.5 * qmul(vec4f(omega, 0.0), q) * h);
  }

  // a capsized hull fills quickly; a little water drains when she is upright and dry
  let ayEnd = qrot(q, vec3f(0.0, 1.0, 0.0));
  if (ayEnd.y < 0.25) { flood += 0.08 * dt; } else { flood -= 0.002 * dt; }
  flood = saturate(flood);

  // sunk: wait on the bottom for a moment, then a fresh boat appears
  var sunk = boat.misc.y;
  if (pos.y < -10.0) { sunk += dt; } else { sunk = 0.0; }

  boat.pos = vec4f(pos, P.enabled);
  boat.vel = vec4f(vel, flood);
  boat.angVel = vec4f(omega, splash);
  boat.rot = q;
  let fwdNow = qrot(q, vec3f(1.0, 0.0, 0.0));
  boat.misc = vec4f(1.0, sunk, course, dot(vel - meanFlow, fwdNow));
  boat.ax = vec4f(fwdNow, 0.0);
  boat.ay = vec4f(ayEnd, 0.0);
  boat.az = vec4f(qrot(q, vec3f(0.0, 0.0, 1.0)), 0.0);
  if (sunk > 3.0) { boat.misc.x = 0.0; }
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
    this.home = { base: [...base], yaw };
    this.base = [...base];   // mooring point (moves to wherever the player stops steering)
    this.yaw = yaw;
    this.status = { flood: 0, sunk: false, heel: 0, position: null };
    this.params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.state = device.createBuffer({ size: (8 + MAX_COLUMNS) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    // small ring of readback buffers so the position reaches the CPU every frame (camera follow)
    this.readbacks = [0, 1, 2].map(() => ({ buffer: device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), busy: false }));
    this.pendingReadback = null;
    this.resetPending = false;

    const columns = hullColumns();
    this.columnCount = columns.length;
    const colData = new Float32Array(MAX_COLUMNS * 8);
    columns.forEach((c, i) => colData.set([c[0], c[1], c[2], c[3], c[4], 0, 0, 0], i * 8));
    this.columns = device.createBuffer({ size: colData.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(this.columns.getMappedRange()).set(colData);
    this.columns.unmap();

    const verts = buildMesh();
    this.vertexCount = verts.length / 10;
    this.vbo = device.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
    new Float32Array(this.vbo.getMappedRange()).set(verts);
    this.vbo.unmap();

    const C = VIS.C, VF = VIS.VF, F = VIS.F;
    const physLayout = makeLayout(device, [[0, C, 'uniform'], [1, C, 'uniform'], [2, C, 'texArray'], [3, C, 'sampler'], [4, C, 'storage'], [5, C, 'read'], [6, C, 'texArray']]);
    this.physPipe = computePipeline(device, physLayout, PHYSICS_WGSL, 'boat physics');
    const sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.physBG = makeBindGroup(device, physLayout, [this.params, frame.buffer, sim.displacementView, sampler, this.state, this.columns, sim.derivativesView]);

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

  respawn() {
    this.base = [...this.home.base];
    this.yaw = this.home.yaw;
    this.resetPending = true;
  }

  // drop anchor here (used when the player hands back the helm)
  setAnchor(xz, yaw) {
    this.base = [xz[0], xz[1]];
    this.yaw = yaw;
  }

  // opts: { enabled, windSpeed (m/s), mass (kg), speed (knots), helm: null | { throttle, rudder } }
  update(encoder, dt, opts) {
    this.enabled = opts.enabled;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.base[0], this.base[1], this.yaw, dt,
      opts.enabled ? 1 : 0, opts.windSpeed, opts.mass, opts.speed,
      this.resetPending ? 1 : 0, this.columnCount, 0, 0,
      opts.helm ? 1 : 0, opts.helm?.throttle ?? 0, opts.helm?.rudder ?? 0, 0,
    ]));
    this.resetPending = false;
    const pass = encoder.beginComputePass({ label: 'boat physics' });
    pass.setPipeline(this.physPipe);
    pass.setBindGroup(0, this.physBG);
    pass.dispatchWorkgroups(1);
    pass.end();
    // read back position / flooding / heel (arrives a frame or two later)
    const rb = this.readbacks.find((r) => !r.busy);
    if (rb) {
      encoder.copyBufferToBuffer(this.state, 0, rb.buffer, 0, 128);
      rb.busy = true;
      this.pendingReadback = rb;
    }
  }

  // call after queue.submit
  poll() {
    const rb = this.pendingReadback;
    if (!rb) return;
    this.pendingReadback = null;
    const stamp = this.readStamp = (this.readStamp || 0) + 1;
    rb.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const d = new Float32Array(rb.buffer.getMappedRange().slice(0));
      rb.buffer.unmap();
      rb.busy = false;
      if (stamp < (this.appliedStamp || 0)) return;  // an older read finished late
      this.appliedStamp = stamp;
      // pos (0-2), ay.y (9) = cosine of the tilt, vel.w (19) flooding, misc.y (29) time sunk, misc.w (31) speed
      this.status = {
        flood: d[19], heel: Math.acos(Math.min(1, Math.max(-1, d[9]))) * 180 / Math.PI,
        sunk: d[1] < -3, speed: d[31], position: [d[0], d[1], d[2]], heading: Math.atan2(d[6], d[4]),
      };
    }).catch(() => { rb.busy = false; });
  }

  draw(pass) {
    if (!this.enabled) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbo);
    pass.draw(this.vertexCount);
  }

  destroy() {
    for (const b of [this.params, this.state, this.columns, this.vbo, ...this.readbacks.map((r) => r.buffer)]) b.destroy();
  }
}
