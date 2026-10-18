// Shared WGSL declarations and small WebGPU helpers.

// Per-frame uniforms shared by every render/compute pass (256 bytes).
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec3f, time: f32,
  sunDir: vec3f, exposure: f32,
  sunColor: vec3f, cloudCoverage: f32,
  resolution: vec2f, cloudTime: f32, frameIndex: f32,
  cascadeL: vec4f,
  windDir: vec2f, foamAmount: f32, sunDiskScale: f32,
  meshA: f32, meshB: f32, meshRadial: f32, meshAngular: f32,
  waterColor: vec3f, bloomStrength: f32,
};
const PI = 3.14159265358979;
fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
`;

export const FRAME_FLOATS = 64;

export class FrameUniforms {
  constructor(device) {
    this.data = new Float32Array(FRAME_FLOATS);
    this.buffer = device.createBuffer({ size: FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }
  set(name, value) {
    const o = FrameUniforms.offsets[name];
    if (typeof value === 'number') this.data[o] = value;
    else this.data.set(value, o);
  }
  upload(device) {
    device.queue.writeBuffer(this.buffer, 0, this.data);
  }
}
FrameUniforms.offsets = {
  viewProj: 0, invViewProj: 16, camPos: 32, time: 35, sunDir: 36, exposure: 39, sunColor: 40, cloudCoverage: 43,
  resolution: 44, cloudTime: 46, frameIndex: 47, cascadeL: 48, windDir: 52, foamAmount: 54, sunDiskScale: 55,
  meshA: 56, meshB: 57, meshRadial: 58, meshAngular: 59, waterColor: 60, bloomStrength: 63,
};

// Compact explicit bind group layouts: [binding, visibility, kind, extra]
const V = { C: GPUShaderStage.COMPUTE, F: GPUShaderStage.FRAGMENT, VF: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, ALL: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE };
export const VIS = V;

export function makeLayout(device, entries) {
  return device.createBindGroupLayout({
    entries: entries.map(([binding, visibility, kind, extra]) => {
      const e = { binding, visibility };
      switch (kind) {
        case 'uniform': e.buffer = { type: 'uniform' }; break;
        case 'storage': e.buffer = { type: 'storage' }; break;
        case 'read': e.buffer = { type: 'read-only-storage' }; break;
        case 'tex': e.texture = { sampleType: 'float', viewDimension: '2d' }; break;
        case 'tex3d': e.texture = { sampleType: 'float', viewDimension: '3d' }; break;
        case 'texArray': e.texture = { sampleType: 'float', viewDimension: '2d-array' }; break;
        case 'sampler': e.sampler = { type: 'filtering' }; break;
        case 'storageTex': e.storageTexture = { access: 'write-only', format: extra || 'rgba16float', viewDimension: '2d' }; break;
        case 'storageTexArray': e.storageTexture = { access: 'write-only', format: extra || 'rgba16float', viewDimension: '2d-array' }; break;
        default: throw new Error('unknown binding kind ' + kind);
      }
      return e;
    }),
  });
}

export function makeBindGroup(device, layout, resources) {
  return device.createBindGroup({
    layout,
    entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
  });
}

export function computePipeline(device, layout, code, label) {
  const module = device.createShaderModule({ code, label });
  return device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'main' },
  });
}
