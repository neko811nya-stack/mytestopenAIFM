// Ocean surface: a camera-centred polar grid whose rings grow geometrically toward the horizon.
// Vertices sample the displacement cascades (mip level matched to vertex spacing),
// fragments rebuild normals from the derivative cascades and shade the water.
import { FRAME_WGSL, makeLayout, makeBindGroup, VIS } from './common.js';
import { SKY_SAMPLE_WGSL } from './sky.js';

const OCEAN_WGSL = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var dispTex: texture_2d_array<f32>;
@group(0) @binding(2) var derivTex: texture_2d_array<f32>;
@group(0) @binding(3) var oceanSampler: sampler;
@group(0) @binding(4) var envTex: texture_2d<f32>;
@group(0) @binding(5) var skySampler: sampler;
@group(0) @binding(6) var skyLut: texture_2d<f32>;
@group(0) @binding(7) var noiseTex: texture_2d<f32>;
@group(0) @binding(8) var noiseSampler: sampler;
@group(0) @binding(9) var<storage, read> sunVis: array<f32>;
` + SKY_SAMPLE_WGSL + /* wgsl */ `
const CASCADES = 3;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) xz: vec2f,
  @location(1) rel: vec3f,
  @location(2) height: f32,
};

fn heightScale() -> f32 { return max(1.0, frame.camPos.y / 12.0); }

@vertex fn vs(@location(0) polar: vec2f) -> VSOut {
  let hs = heightScale();
  let r = polar.x * hs;
  let xz = frame.camPos.xz + vec2f(cos(polar.y), sin(polar.y)) * r;
  let spacing = max(frame.meshB * (polar.x + frame.meshA), 2.0 * PI * polar.x / frame.meshAngular) * hs;
  let texels = f32(textureDimensions(dispTex).x);
  var disp = vec3f(0.0);
  for (var c = 0; c < CASCADES; c++) {
    let L = frame.cascadeL[c];
    let lod = max(0.0, log2(spacing * 1.5 / (L / texels)));
    disp += textureSampleLevel(dispTex, oceanSampler, xz / L, c, lod).xyz;
  }
  let world = vec3f(xz.x + disp.x, disp.y, xz.y + disp.z);
  var o: VSOut;
  o.rel = world - frame.camPos;
  o.pos = frame.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  o.height = disp.y;
  return o;
}

fn ggx(NdotH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn smithG1(NdotX: f32, a: f32) -> f32 {
  let a2 = a * a;
  return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  // -------- surface derivatives from all cascades
  var hx = 0.0; var hz = 0.0; var jxx = 0.0; var jzz = 0.0; var foam = 0.0;
  var crestJ = 1.0;
  let foamW = array<f32, 3>(1.0, 0.3, 0.0);
  for (var c = 0; c < CASCADES; c++) {
    let uv = in.xz / frame.cascadeL[c];
    let d = textureSample(derivTex, oceanSampler, uv, c);
    hx += d.x; hz += d.y; jxx += d.z; jzz += d.w;
    foam += textureSample(dispTex, oceanSampler, uv, c).w * foamW[c];
    if (c == 1) { crestJ = (1.0 + jxx) * (1.0 + jzz); }  // big + medium waves only: ripples don't break
  }
  // foam texture stretched along the wind so whitecaps streak downwind
  let wd = frame.windDir;
  let wuv = vec2f(dot(in.xz, wd), dot(in.xz, vec2f(-wd.y, wd.x)));
  let fp = textureSample(noiseTex, noiseSampler, wuv * vec2f(0.006, 0.016));
  let fp2 = textureSample(noiseTex, noiseSampler, wuv * vec2f(0.03, 0.07) + vec2f(0.37, 0.71));
  let fp3 = textureSample(noiseTex, noiseSampler, wuv * vec2f(0.22, 0.4) + vec2f(0.11, 0.53));

  let dist = length(in.rel);
  let V = -in.rel / dist;
  var n = normalize(vec3f(-hx / max(1.0 + jxx, 0.25), 1.0, -hz / max(1.0 + jzz, 0.25)));
  // Far away, the filtered normal is smoother than the real surface: keep it but widen the lobe below.
  let NdotV0 = dot(n, V);
  if (NdotV0 < 0.02) { n = normalize(n + V * (0.02 - NdotV0)); }
  let NdotV = max(dot(n, V), 1e-3);

  let L = frame.sunDir;
  let sunVisible = sunVis[0];
  let sunCol = frame.sunColor * sunVisible;
  let ambient = skyIrradiance();

  // Unresolved slope variance grows with the pixel footprint (glitter road widens with distance).
  let footprint = length(fwidth(in.xz));
  let rough = clamp(0.045 + 0.06 * log2(1.0 + footprint * 1.5), 0.045, 0.38);

  // -------- reflection
  var R = reflect(-V, n);
  R.y = abs(R.y);
  let refl = textureSampleLevel(envTex, skySampler, envUV(normalize(R)), 0.0).rgb;
  let F = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0) * mix(1.0, 0.55, saturate(rough * 2.5));

  // -------- sun specular (GGX)
  let H = normalize(L + V);
  let NdotL = dot(n, L);
  var spec = vec3f(0.0);
  if (NdotL > 0.0) {
    let VdotH = saturate(dot(V, H));
    let Fs = 0.02 + 0.98 * pow(1.0 - VdotH, 5.0);
    let D = ggx(saturate(dot(n, H)), rough);
    let Gs = smithG1(NdotL, rough) * smithG1(NdotV, rough);
    spec = sunCol * (D * Gs * Fs / (4.0 * NdotV)) ;
  }

  // -------- water body: upwelling light + subsurface glow through thin crests
  let wc = frame.waterColor;
  let body = wc * (ambient * 0.55 + sunCol * saturate(L.y + 0.15) * 0.25);
  let crest = saturate(in.height * 0.35 + 0.25);
  let toward = pow(saturate(dot(-V, normalize(vec3f(L.x, 0.0, L.z) + vec3f(0.0, -0.15, 0.0)))), 3.0);
  let sssColor = vec3f(0.08, 0.42, 0.36);
  let sss = sssColor * sunCol * (crest * crest) * toward * pow(saturate(0.5 - 0.5 * dot(L, n)), 2.0) * 0.6
          + sssColor * ambient * crest * 0.035;
  var col = (1.0 - F) * (body + sss) + F * refl;

  // -------- whitecaps
  // persistent foam (decays slowly in the simulation, leaving trails) + instantaneous crest foam from the combined Jacobian
  let crestFoam = saturate((0.68 - crestJ) * 2.5);
  let foamRaw = (foam + crestFoam * 0.7) * frame.foamAmount * (0.55 + 0.9 * fp.a);
  // Foam amount acts as a threshold on cellular noise: crisp bubbly edges up close,
  // smooth coverage far away where the cells are smaller than a pixel.
  let cells = fp3.a * 0.65 + fp2.a * 0.35;
  let cut = 1.0 - saturate(foamRaw * 1.3);
  let aa = max(fwidth(cells) * 1.5, 0.03);
  let crisp = smoothstep(cut - aa, cut + aa, cells) * saturate(foamRaw * 4.0);
  let foamCov = mix(crisp, saturate(foamRaw * 0.8), saturate(footprint * 1.5 - 0.2));
  let foamLight = (sunCol * saturate(dot(n, L) + 0.1) + ambient) * (0.8 / PI);
  col = mix(col + spec, foamLight, foamCov);

  // -------- aerial perspective toward the horizon
  let hazeCol = sampleSky(normalize(vec3f(-V.x, 0.015, -V.z)));
  let fog = 1.0 - exp(-dist / (32000.0 * heightScale()));
  let edge = smoothstep(0.6, 1.0, length(in.xz - frame.camPos.xz) / (frame.meshRadial * heightScale()));
  col = mix(col, hazeCol, max(fog, edge));
  return vec4f(col, 1.0);
}
`;

// Ring radii r_j = A * (exp(b j) - 1): first ring spacing s0, outer radius rMax.
function solveRings(rings, s0, rMax) {
  let lo = 1e-5, hi = 1;
  for (let i = 0; i < 100; i++) {
    const b = 0.5 * (lo + hi);
    const r = (s0 / b) * (Math.exp(b * rings) - 1);
    if (r > rMax) hi = b; else lo = b;
  }
  const b = 0.5 * (lo + hi);
  return { A: s0 / b, b };
}

export class OceanRenderer {
  constructor(device, sim, sky, frame, hdrFormat, { rings = 512, segments = 512, s0 = 0.08, rMax = 24000, samples = 4 } = {}) {
    this.device = device;
    // keep the near-camera vertex spacing comparable when the ring count drops
    s0 *= 512 / rings;
    const { A, b } = solveRings(rings, s0, rMax);
    this.mesh = { A, b, rMax, segments };
    const verts = new Float32Array((rings + 1) * (segments + 1) * 2);
    let k = 0;
    for (let j = 0; j <= rings; j++) {
      const r = A * (Math.exp(b * j) - 1);
      for (let i = 0; i <= segments; i++) {
        verts[k++] = r;
        verts[k++] = (i / segments) * Math.PI * 2;
      }
    }
    const idx = new Uint32Array(rings * segments * 6);
    k = 0;
    const row = segments + 1;
    for (let j = 0; j < rings; j++) {
      for (let i = 0; i < segments; i++) {
        const a = j * row + i, c = a + row;
        idx[k++] = a; idx[k++] = c; idx[k++] = a + 1;
        idx[k++] = a + 1; idx[k++] = c; idx[k++] = c + 1;
      }
    }
    this.indexCount = idx.length;
    this.vbo = device.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true });
    new Float32Array(this.vbo.getMappedRange()).set(verts);
    this.vbo.unmap();
    this.ibo = device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX, mappedAtCreation: true });
    new Uint32Array(this.ibo.getMappedRange()).set(idx);
    this.ibo.unmap();

    this.sampler = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 16,
    });
    const VF = VIS.VF, F = VIS.F;
    const layout = makeLayout(device, [
      [0, VF, 'uniform'], [1, VF, 'texArray'], [2, F, 'texArray'], [3, VF, 'sampler'], [4, F, 'tex'], [5, F, 'sampler'],
      [6, F, 'tex'], [7, F, 'tex'], [8, F, 'sampler'], [9, F, 'read'],
    ]);
    this.bindGroup = makeBindGroup(device, layout, [
      frame.buffer, sim.displacementView, sim.derivativesView, this.sampler, sky.envView, sky.skySampler,
      sky.lutView, sky.noiseView, sky.noiseSampler, sky.sunVis,
    ]);
    const module = device.createShaderModule({ code: OCEAN_WGSL, label: 'ocean' });
    this.pipeline = device.createRenderPipeline({
      label: 'ocean',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs', buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }] },
      fragment: { module, entryPoint: 'fs', targets: [{ format: hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: samples },
    });
  }

  destroy() {
    this.vbo.destroy();
    this.ibo.destroy();
  }

  draw(pass) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbo);
    pass.setIndexBuffer(this.ibo, 'uint32');
    pass.drawIndexed(this.indexCount);
  }
}
