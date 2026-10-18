import { FrameUniforms } from './common.js';
import { OceanSimulation } from './ocean.js';
import { OceanRenderer } from './oceanRender.js';
import { Sky, sunTransmittance, SUN_ILLUMINANCE } from './sky.js';
import { Post } from './post.js';
import { FlyCamera } from './camera.js';
import { mul, perspectiveReversedInfinite, viewRotation, invert, dirFromAngles } from './math.js';

const HDR = 'rgba16float';
const deg = Math.PI / 180;

// URL parameters make screenshots reproducible: ?t=12&cam=x,y,z,yawDeg,pitchDeg&sun=elev,azim&ui=0
const query = new URLSearchParams(location.search);
const num = (k, d) => (query.has(k) ? parseFloat(query.get(k)) : d);
const list = (k) => (query.has(k) ? query.get(k).split(',').map(Number) : null);

const settings = {
  windSpeed: num('wind', 9),
  windAngle: num('windDir', 200),
  choppy: num('choppy', 1.15),
  foam: num('foam', 1.0),
  sunElevation: list('sun')?.[0] ?? 2,
  sunAzimuth: list('sun')?.[1] ?? 20,
  clouds: num('clouds', 0.55),
  exposure: num('exposure', 1.8),
  timeScale: num('timeScale', 1),
  bloom: num('bloom', 0.05),
};

async function main() {
  const status = document.getElementById('status');
  if (!navigator.gpu) return fail('このブラウザは WebGPU に対応していません (Chrome / Edge 113+ を使ってください)。');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('WebGPU アダプタを取得できませんでした。');
  const device = await adapter.requestDevice();
  device.lost.then((info) => fail('GPU デバイスが失われました: ' + info.message));
  device.addEventListener('uncapturederror', (e) => {
    console.error(e.error.message);
    status.textContent = 'GPU error: ' + e.error.message;
  });

  const canvas = document.getElementById('gfx');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const frame = new FrameUniforms(device);
  const sim = new OceanSimulation(device, { N: 256, lengths: [256, 48, 9] });
  const sky = new Sky(device, frame, HDR);
  const ocean = new OceanRenderer(device, sim, sky, frame, HDR);
  const post = new Post(device, frame, format, HDR);

  const cam = list('cam');
  const camera = new FlyCamera(canvas, {
    position: cam ? cam.slice(0, 3) : [0, 7, 0],
    yaw: (cam ? cam[3] : settings.sunAzimuth) * deg,
    pitch: (cam ? cam[4] : -4) * deg,
  });

  bindUI(settings);

  let width = 0, height = 0;
  const renderScale = num('scale', Math.min(window.devicePixelRatio || 1, 1.5));
  const resize = () => {
    const w = Math.max(1, Math.floor(canvas.clientWidth * renderScale));
    const h = Math.max(1, Math.floor(canvas.clientHeight * renderScale));
    if (w === width && h === height) return;
    width = canvas.width = w;
    height = canvas.height = h;
    post.resize(w, h);
  };

  const fixedTime = query.has('t') ? num('t', 0) : null;
  let simTime = fixedTime ?? 0;
  let last = performance.now();
  let frameIndex = 0;
  let fpsAcc = 0, fpsFrames = 0;
  const fpsEl = document.getElementById('fps');

  const render = (now) => {
    resize();
    let dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    fpsAcc += dt; fpsFrames++;
    if (fpsAcc > 0.5) { fpsEl.textContent = `${Math.round(fpsFrames / fpsAcc)} fps`; fpsAcc = 0; fpsFrames = 0; }
    camera.update(dt);
    const simDt = fixedTime !== null ? 1 / 60 : dt * settings.timeScale;
    simTime += simDt;

    sim.setParams({ windSpeed: settings.windSpeed, windAngle: settings.windAngle * deg, choppy: settings.choppy });
    const sunDir = dirFromAngles(settings.sunElevation * deg, settings.sunAzimuth * deg);
    const T = sunTransmittance(sunDir);
    const proj = perspectiveReversedInfinite(camera.fov, width / height, 0.1);
    const viewProj = mul(proj, viewRotation(camera.forward()));
    frame.set('viewProj', viewProj);
    frame.set('invViewProj', invert(viewProj));
    frame.set('camPos', camera.position);
    frame.set('time', simTime);
    frame.set('sunDir', sunDir);
    frame.set('exposure', settings.exposure);
    frame.set('sunColor', T.map((t) => t * SUN_ILLUMINANCE));
    frame.set('cloudCoverage', settings.clouds);
    frame.set('resolution', [width, height]);
    frame.set('cloudTime', simTime);
    frame.set('frameIndex', frameIndex++);
    frame.set('cascadeL', [...sim.lengths, 1]);
    const wa = settings.windAngle * deg;
    frame.set('windDir', [Math.cos(wa), Math.sin(wa)]);
    frame.set('foamAmount', settings.foam);
    frame.set('sunDiskScale', 1800);
    frame.set('meshA', ocean.mesh.A);
    frame.set('meshB', ocean.mesh.b);
    frame.set('meshRadial', ocean.mesh.rMax);
    frame.set('meshAngular', ocean.mesh.segments);
    frame.set('waterColor', [0.012, 0.05, 0.075]);
    frame.set('bloomStrength', settings.bloom);
    frame.upload(device);

    if (frameIndex === 1) {
      // warm-up: let persistent foam build its trails before the first frame is shown.
      // One submit per step, because the step parameters go through queue.writeBuffer.
      for (let i = 120; i > 0; i--) {
        const warm = device.createCommandEncoder();
        sim.update(warm, simTime - i / 30, 1 / 30);
        device.queue.submit([warm.finish()]);
      }
    }
    const encoder = device.createCommandEncoder();
    sim.update(encoder, simTime, simDt);
    sky.update(encoder);
    const pass = post.beginScene(encoder);
    ocean.draw(pass);
    sky.draw(pass);
    pass.end();
    post.finish(encoder, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    window.__frames = frameIndex;
    requestAnimationFrame(render);
  };
  status.textContent = '';
  requestAnimationFrame(render);
}

function bindUI(s) {
  const panel = document.getElementById('panel');
  if (query.get('ui') === '0') {
    panel.style.display = 'none';
    document.getElementById('help').style.display = 'none';
  }
  for (const input of panel.querySelectorAll('input[data-key]')) {
    const key = input.dataset.key;
    const out = input.parentElement.querySelector('output');
    input.value = s[key];
    const sync = () => {
      s[key] = parseFloat(input.value);
      if (out) out.textContent = (+input.value).toFixed(input.step < 1 ? 2 : 0);
    };
    input.addEventListener('input', sync);
    sync();
  }
  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyH' && !(e.target instanceof HTMLInputElement)) panel.classList.toggle('hidden');
  });
}

function fail(msg) {
  document.getElementById('status').textContent = msg;
  document.getElementById('status').classList.add('error');
}

main();
