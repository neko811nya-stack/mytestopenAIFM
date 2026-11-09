import { FrameUniforms } from './common.js';
import { OceanSimulation } from './ocean.js';
import { OceanRenderer } from './oceanRender.js';
import { Sky, sunTransmittance, SUN_ILLUMINANCE, MOON_SCALE } from './sky.js';
import { Weather, WEATHER } from './weather.js';
import { celestial, moonPhaseName, formatHour, DEFAULT_HOUR } from './celestial.js';
import { Post } from './post.js';
import { Boat } from './boat.js';
import { FlyCamera } from './camera.js';
import { mul, perspectiveReversedInfinite, viewRotation, invert } from './math.js';

const HDR = 'rgba16float';

// Quality presets, from phone-friendly (0) to the original desktop quality (4).
//   pixelRatio: max device pixels per CSS pixel, samples: MSAA, fftN: FFT size per cascade,
//   mesh: rings = segments of the ocean grid, cloudSteps/lightSteps: cloud ray-march,
//   envSize/envSteps: reflection environment map, particles: max rain drops / snowflakes
const QUALITY = [
  { name: 'ライト', pixelRatio: 0.75, samples: 1, fftN: 128, mesh: 192, cloudSteps: 16, lightSteps: 2, envSize: [512, 128], envSteps: 8, bloom: false, particles: 2500 },
  { name: '低', pixelRatio: 1.0, samples: 1, fftN: 128, mesh: 256, cloudSteps: 22, lightSteps: 3, envSize: [512, 128], envSteps: 10, bloom: true, particles: 4000 },
  { name: '中', pixelRatio: 1.0, samples: 4, fftN: 256, mesh: 320, cloudSteps: 26, lightSteps: 3, envSize: [768, 192], envSteps: 14, bloom: true, particles: 6000 },
  { name: '高', pixelRatio: 1.25, samples: 4, fftN: 256, mesh: 448, cloudSteps: 34, lightSteps: 4, envSize: [1024, 256], envSteps: 18, bloom: true, particles: 8000 },
  { name: '最高', pixelRatio: 1.5, samples: 4, fftN: 256, mesh: 512, cloudSteps: 40, lightSteps: 4, envSize: [1024, 256], envSteps: 18, bloom: true, particles: 10000 },
];
// 時の流れ: game seconds per real second
const TIME_FLOW = [
  { name: '停止', rate: 0 }, { name: '実時間', rate: 1 }, { name: '×60', rate: 60 }, { name: '×300', rate: 300 }, { name: '×1200', rate: 1200 },
];
const isTouch = matchMedia('(pointer: coarse)').matches;
// Auto exposure is relative to the light of the default sunset, so that view keeps the exposure slider as is.
const REF_LIGHT = sunTransmittance(celestial(DEFAULT_HOUR, 0).sunDir)[1] * SUN_ILLUMINANCE;
const deg = Math.PI / 180;

// URL parameters make screenshots reproducible: ?t=12&cam=x,y,z,yawDeg,pitchDeg&hour=21.5&moon=14&weather=storm&ui=0
const query = new URLSearchParams(location.search);
const num = (k, d) => (query.has(k) ? parseFloat(query.get(k)) : d);
const list = (k) => (query.has(k) ? query.get(k).split(',').map(Number) : null);

// Slider defaults (the reset button returns to these). Quality is per device and is not reset.
const DEFAULTS = {
  windSpeed: 9, windAngle: 200, choppy: 1.15, foam: 1.0, hour: DEFAULT_HOUR, timeFlow: 0, moonAge: 14.8, weather: 'clear',
  clouds: 0.55, exposure: 1.8, autoExposure: 1, reflection: 1, timeScale: 1, bloom: 0.05, boat: true, waveHeight: 1, boatMass: 3, boatSpeed: 0,
};
const DEFAULT_CAMERA = { position: [0, 7, 0], yaw: 20 * deg, pitch: -4 * deg };

const settings = {
  windSpeed: num('wind', DEFAULTS.windSpeed),
  windAngle: num('windDir', DEFAULTS.windAngle),
  choppy: num('choppy', DEFAULTS.choppy),
  foam: num('foam', DEFAULTS.foam),
  hour: num('hour', DEFAULTS.hour),
  timeFlow: num('flow', DEFAULTS.timeFlow),
  moonAge: num('moon', DEFAULTS.moonAge),
  weather: WEATHER[query.get('weather')] ? query.get('weather') : DEFAULTS.weather,
  clouds: num('clouds', DEFAULTS.clouds),
  exposure: num('exposure', DEFAULTS.exposure),
  autoExposure: num('auto', DEFAULTS.autoExposure),
  reflection: num('reflection', DEFAULTS.reflection),
  timeScale: num('timeScale', DEFAULTS.timeScale),
  bloom: num('bloom', DEFAULTS.bloom),
  boat: query.get('boat') !== '0',
  waveHeight: num('waves', DEFAULTS.waveHeight),
  boatMass: num('mass', DEFAULTS.boatMass),
  boatSpeed: num('speed', DEFAULTS.boatSpeed),
  quality: num('q', storedQuality() ?? (isTouch ? 0 : 4)),
};

function storedQuality() {
  try {
    const v = localStorage.getItem('ocean.quality');
    return v === null ? null : +v;
  } catch {
    return null;
  }
}

async function main() {
  const status = document.getElementById('status');
  const support = '<br>PC: Chrome / Edge 113 以降<br>Android: Chrome 121 以降 (Android 12+)<br>iPhone / iPad: iOS 26 以降の Safari';
  if (!navigator.gpu) return fail('このブラウザは WebGPU に対応していません。' + support);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('WebGPU アダプタを取得できませんでした。' + support);
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

  // Everything whose size or pipeline depends on the quality preset is rebuilt together.
  let gfx = null;
  let width = 0, height = 0;
  let needsWarmup = true;
  const build = (level) => {
    const q = QUALITY[level];
    const weatherState = gfx?.weather.state;
    if (gfx) for (const part of [gfx.sim, gfx.sky, gfx.boat, gfx.ocean, gfx.post, gfx.weather]) part.destroy();
    const sim = new OceanSimulation(device, { N: q.fftN, lengths: [256, 48, 9] });
    const sky = new Sky(device, frame, HDR, q);
    const boat = new Boat(device, sim, sky, frame, HDR, { samples: q.samples });
    const ocean = new OceanRenderer(device, sim, sky, boat, frame, HDR, { rings: q.mesh, segments: q.mesh, samples: q.samples });
    const post = new Post(device, frame, format, HDR, { samples: q.samples, bloom: q.bloom });
    const weather = new Weather(device, sky, frame, HDR, { samples: q.samples, particles: q.particles });
    if (weatherState) Object.assign(weather.state, weatherState);   // keep the current weather through a rebuild
    else {
      // first start: begin in the chosen weather instead of fading in
      const w = WEATHER[settings.weather];
      Object.assign(weather.state, { overcast: w.overcast, rain: w.rain, snow: w.snow, thunder: w.thunder });
    }
    if (query.has('flash')) weather.debugFlash = num('flash', 1);   // screenshots: hold a lightning strike
    gfx = { q, sim, sky, boat, ocean, post, weather };
    width = height = 0;  // force a resize for the new render targets
    needsWarmup = true;
  };
  build(settings.quality);

  const cam = list('cam');
  const camera = new FlyCamera(canvas, cam
    ? { position: cam.slice(0, 3), yaw: cam[3] * deg, pitch: cam[4] * deg }
    : { ...DEFAULT_CAMERA });

  bindUI(settings, (level) => {
    try { localStorage.setItem('ocean.quality', level); } catch { /* storage may be unavailable */ }
    build(level);
  }, () => {
    Object.assign(settings, DEFAULTS);
    timeUI.sync();
    gfx.boat.respawn();
    camera.position = [...DEFAULT_CAMERA.position];
    camera.yaw = DEFAULT_CAMERA.yaw;
    camera.pitch = DEFAULT_CAMERA.pitch;
    camera.velocity = [0, 0, 0];
    camera.setMode('free');
  });
  bindTouch(camera);
  const timeUI = bindTimeAndWeather(settings);

  // Camera targets: anything with a name and a position (null while unavailable).
  // The boat is the only one for now; its position comes back from the GPU a frame or two late,
  // so the camera aims at a smoothed copy (and follows at a fixed altitude, ignoring wave bobbing).
  const targets = [{
    name: '船',
    position: () => (settings.boat ? gfx.boat.status.position : null),  // keep watching while it sinks
  }];
  const aim = { target: targets[0], smooth: null };
  const camButtons = bindCameraModes(camera, () => aim.smooth, () => gfx.boat);
  document.getElementById('respawn').addEventListener('click', () => gfx.boat.respawn());

  const resize = () => {
    const renderScale = query.has('scale') ? num('scale', 1) : Math.min(window.devicePixelRatio || 1, gfx.q.pixelRatio);
    const w = Math.max(1, Math.floor(canvas.clientWidth * renderScale));
    const h = Math.max(1, Math.floor(canvas.clientHeight * renderScale));
    if (w === width && h === height) return;
    width = canvas.width = w;
    height = canvas.height = h;
    gfx.post.resize(w, h);
    gfx.ocean.setReflection(gfx.post.reflectionView);
  };

  const fixedTime = query.has('t') ? num('t', 0) : null;
  let simTime = fixedTime ?? 0;
  let last = performance.now();
  let frameIndex = 0;
  let fpsAcc = 0, fpsFrames = 0;
  const fpsEl = document.getElementById('fps');

  const render = (now) => {
    resize();
    const { sim, sky, boat, ocean, post, weather } = gfx;
    let dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    fpsAcc += dt; fpsFrames++;
    if (fpsAcc > 0.5) { fpsEl.textContent = `${Math.round(fpsFrames / fpsAcc)} fps`; fpsAcc = 0; fpsFrames = 0; }
    const raw = aim.target.position();
    if (!raw) aim.smooth = null;
    else if (!aim.smooth) aim.smooth = [raw[0], raw[1] + 2.5, raw[2]];
    else {
      const a = 1 - Math.exp(-dt * 6), ay = 1 - Math.exp(-dt * 1.5);
      aim.smooth = [aim.smooth[0] + (raw[0] - aim.smooth[0]) * a, aim.smooth[1] + (raw[1] + 2.5 - aim.smooth[1]) * ay, aim.smooth[2] + (raw[2] - aim.smooth[2]) * a];
    }
    const heading = gfx.boat.status.heading ?? 0;
    if (query.has('cammode') && aim.smooth && !aim.started) { aim.started = true; camera.setMode(query.get('cammode'), aim.smooth, heading); }
    camera.update(dt, aim.smooth, heading);
    camButtons.sync();
    const simDt = fixedTime !== null ? 1 / 60 : dt * settings.timeScale;
    simTime += simDt;

    sim.setParams({
      windSpeed: settings.windSpeed, windAngle: settings.windAngle * deg, choppy: settings.choppy,
      // bigger seas: scaled up geometrically, plus a little extra steepness so they can swamp a boat
      scale: settings.waveHeight, amplitude: Math.pow(settings.waveHeight, 0.6) });
    // ---- time of day, sun, moon
    const flow = TIME_FLOW[settings.timeFlow]?.rate ?? 0;
    if (flow > 0) {
      settings.hour = (settings.hour + dt * flow / 3600) % 24;
      timeUI.sync();
    }
    const sky0 = celestial(settings.hour, settings.moonAge);
    const sunColor = sunTransmittance(sky0.sunDir).map((t) => t * SUN_ILLUMINANCE);
    const moonColor = sunTransmittance(sky0.moonDir).map((t) => t * SUN_ILLUMINANCE * MOON_SCALE);
    const moonLight = moonColor.map((c) => c * sky0.moonIllum);
    // key light: whichever of the sun and the moon is brighter right now
    const sunKey = sunColor[1] >= moonLight[1];
    const wx = weather.update(dt, settings.weather, camera);
    // auto exposure: daylight from the sun (with a twilight fall-off) plus moonlight, dimmed by cloud
    const el = sky0.sunElevation;
    const daylight = el > 2 ? Math.max(sunColor[1], REF_LIGHT) : REF_LIGHT * Math.pow(10, (el - 2) / 3.5);
    const level = (daylight + moonLight[1] * 1.0 + 0.003) * wx.dimming;
    // 露出オート: 0 = fixed exposure, 1 = full adaptation, above 1 = adapts even harder (in log space)
    const autoExposure = Math.pow(Math.min(Math.max(REF_LIGHT / level, 0.6), 3000), 0.8 * settings.autoExposure);
    // how "night" it looks: drives the blue, desaturated night vision in the tonemapper
    const nightness = Math.min(1, Math.max(0, Math.log10(REF_LIGHT / Math.max(daylight, 1e-6)) / 2.5));
    // portrait screens: widen the vertical FOV so the horizontal view stays at least ~50 degrees
    const aspect = width / height;
    const fovY = Math.min(85 * deg, Math.max(camera.fov, 2 * Math.atan(Math.tan(25 * deg) / aspect)));
    const proj = perspectiveReversedInfinite(fovY, aspect, 0.1);
    const viewProj = mul(proj, viewRotation(camera.forward()));
    frame.set('viewProj', viewProj);
    frame.set('invViewProj', invert(viewProj));
    frame.set('camPos', camera.position);
    frame.set('time', simTime);
    frame.set('sunDir', sky0.sunDir);
    frame.set('exposure', settings.exposure * autoExposure);
    frame.set('sunColor', sunColor);
    frame.set('moonDir', sky0.moonDir);
    frame.set('moonIllum', sky0.moonIllum);
    frame.set('moonColor', moonColor);
    frame.set('stars', sky0.starAngle);
    frame.set('keyDir', sunKey ? sky0.sunDir : sky0.moonDir);
    frame.set('keyColor', sunKey ? sunColor : moonLight);
    frame.set('rain', wx.rain);
    frame.set('snow', wx.snow);
    frame.set('overcast', wx.overcast);
    frame.set('fogDistance', wx.fogDistance);
    // a flash is bright relative to the scene: scale it against the auto exposure so a night strike doesn't white out
    frame.set('lightning', wx.lightning * 0.7 / autoExposure);
    frame.set('lightningPos', wx.lightningPos);
    frame.set('night', nightness);
    frame.set('reflection', settings.reflection);
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
    frame.set('bloomStrength', gfx.q.bloom ? settings.bloom : 0);
    frame.upload(device);

    if (needsWarmup) {
      needsWarmup = false;
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
    const driving = camera.mode === 'drive';
    const input = camera.input();
    boat.update(encoder, simDt, {
      enabled: settings.boat, windSpeed: settings.windSpeed, mass: settings.boatMass * 1000, speed: settings.boatSpeed,
      helm: driving ? { throttle: input.ahead, rudder: input.side } : null,
    });
    sky.update(encoder);
    const reflectionPass = post.beginReflection(encoder);
    boat.drawReflection(reflectionPass);
    reflectionPass.end();
    const pass = post.beginScene(encoder);
    ocean.draw(pass);
    boat.draw(pass);
    sky.draw(pass);
    weather.draw(pass);
    pass.end();
    post.finish(encoder, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    boat.poll();
    showBoatStatus(boat.status, settings.boat, driving);
    window.__frames = frameIndex;
    requestAnimationFrame(render);
  };
  status.textContent = '';
  requestAnimationFrame(render);
}

function bindUI(s, onQuality, onReset) {
  const panel = document.getElementById('panel');
  const gear = document.getElementById('gear');
  if (query.get('ui') === '0') {
    for (const id of ['panel', 'gear', 'help', 'fps', 'touch', 'camModes']) document.getElementById(id).style.display = 'none';
  }
  if (isTouch) document.body.classList.add('touch');
  const setOpen = (open) => {
    panel.classList.toggle('hidden', !open);
    gear.setAttribute('aria-expanded', String(open));
  };
  // the panel starts open on desktop (as before) and closed on phones, where it would cover the view
  setOpen(query.has('panel') ? query.get('panel') === '1' : !isTouch);
  gear.addEventListener('click', () => setOpen(panel.classList.contains('hidden')));
  window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyH' && !(e.target instanceof HTMLInputElement)) setOpen(panel.classList.contains('hidden'));
  });

  const sliders = [...panel.querySelectorAll('input[data-key]')];
  const show = (input) => {
    const out = input.parentElement.querySelector('output');
    if (out) out.textContent = formatSlider(input);
  };
  for (const input of sliders) {
    input.value = s[input.dataset.key];
    input.addEventListener('input', () => {
      s[input.dataset.key] = parseFloat(input.value);
      show(input);
    });
    show(input);
  }
  const boatToggle = document.getElementById('boat');
  boatToggle.checked = s.boat;
  boatToggle.addEventListener('change', () => { s.boat = boatToggle.checked; });

  document.getElementById('reset').addEventListener('click', () => {
    onReset();
    for (const input of sliders) {
      input.value = s[input.dataset.key];
      show(input);
    }
    boatToggle.checked = s.boat;
  });

  const quality = document.getElementById('quality');
  const name = document.getElementById('qualityName');
  quality.value = s.quality;
  name.textContent = QUALITY[s.quality].name;
  quality.addEventListener('input', () => { name.textContent = QUALITY[+quality.value].name; });
  // rebuild only when the slider is released, not on every step while dragging
  quality.addEventListener('change', () => {
    if (+quality.value === s.quality) return;
    s.quality = +quality.value;
    onQuality(s.quality);
  });
}

// Slider read-outs: plain numbers, or a clock / moon phase / time-flow name.
function formatSlider(input) {
  const v = +input.value;
  switch (input.dataset.format) {
    case 'hour': return formatHour(v);
    case 'flow': return TIME_FLOW[v]?.name ?? '';
    case 'moon': return moonPhaseName(v);
    case 'auto': return v === 0 ? 'オフ' : v.toFixed(2);
    default: return v.toFixed(input.step < 1 ? 2 : 0);
  }
}

// The weather buttons, and keeping the clock slider in step while time flows.
function bindTimeAndWeather(s) {
  const buttons = [...document.querySelectorAll('#weather button')];
  const hour = document.querySelector('input[data-key="hour"]');
  const sync = () => {
    for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.weather === s.weather));
    // throttle DOM writes while time flows
    const now = performance.now();
    if (now - (sync.last || 0) > 150 || Math.abs(+hour.value - s.hour) > 0.5) {
      sync.last = now;
      hour.value = s.hour;
      hour.parentElement.querySelector('output').textContent = formatHour(s.hour);
    }
  };
  for (const b of buttons) {
    b.addEventListener('click', () => { s.weather = b.dataset.weather; sync(); });
  }
  sync();
  return { sync };
}

function showBoatStatus(st, enabled, driving) {
  const el = document.getElementById('boatStatus');
  const text = !enabled ? '' : st.sunk ? '沈没…まもなく新しい船が出ます'
    : st.heel > 100 ? `転覆！ 浸水 ${Math.round(st.flood * 100)}%`
    : `傾き ${Math.round(st.heel)}° · 浸水 ${Math.round(st.flood * 100)}%`;
  if (el.textContent !== text) el.textContent = text;
  // speed / heading readout while steering
  const hud = document.getElementById('helmHud');
  const show = enabled && driving;
  if (hud.hidden === show) {
    hud.hidden = !show;
    document.body.classList.toggle('driving', show);  // the key help would sit under the HUD
  }
  if (show) {
    const kn = Math.max(0, (st.speed ?? 0) / 0.5144);
    // compass-style: turning to starboard (D) increases the heading
    const deg = Math.round((((st.heading ?? 0) * 180 / Math.PI) % 360 + 360) % 360);
    const t = `${kn.toFixed(1)} kn · 針路 ${String(deg).padStart(3, '0')}° · ${text}`;
    if (hud.textContent !== t) hud.textContent = t;
  }
}

function bindCameraModes(camera, getTarget, getBoat) {
  const buttons = {
    lock: document.getElementById('camLock'), follow: document.getElementById('camFollow'), drive: document.getElementById('camDrive'),
  };
  const setMode = (mode) => {
    const boat = getBoat();
    // handing back the helm: drop anchor where the boat is instead of hauling it home
    if (camera.mode === 'drive' && mode !== 'drive' && boat.status.position) {
      boat.setAnchor([boat.status.position[0], boat.status.position[2]], boat.status.heading);
    }
    camera.setMode(mode, getTarget(), boat.status.heading ?? 0);
  };
  const toggle = (mode) => setMode(camera.mode === mode ? 'free' : mode);
  for (const [mode, b] of Object.entries(buttons)) b.addEventListener('click', () => toggle(mode));
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.code === 'KeyL') toggle('lock');
    if (e.code === 'KeyF') toggle('follow');
    if (e.code === 'KeyB') toggle('drive');
    if (e.code === 'Escape') setMode('free');
  });
  return {
    sync() {
      const available = !!getTarget();
      for (const [mode, b] of Object.entries(buttons)) {
        const on = camera.mode === mode;
        if (b.getAttribute('aria-pressed') !== String(on)) b.setAttribute('aria-pressed', String(on));
        if (b.disabled === available) b.disabled = !available;
      }
    },
  };
}

function bindTouch(camera) {
  const stick = document.getElementById('stick');
  const knob = document.getElementById('knob');
  let id = null;
  const move = (e) => {
    const r = stick.getBoundingClientRect();
    const radius = r.width / 2;
    let x = (e.clientX - r.left - radius) / radius, y = (e.clientY - r.top - radius) / radius;
    const l = Math.hypot(x, y);
    if (l > 1) { x /= l; y /= l; }
    camera.touch.x = x;
    camera.touch.y = y;
    knob.style.transform = `translate(${x * radius * 0.6}px, ${y * radius * 0.6}px)`;
  };
  const release = (e) => {
    if (e.pointerId !== id) return;
    id = null;
    camera.touch.x = camera.touch.y = 0;
    knob.style.transform = '';
  };
  stick.addEventListener('pointerdown', (e) => { id = e.pointerId; stick.setPointerCapture(id); move(e); });
  stick.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
  stick.addEventListener('pointerup', release);
  stick.addEventListener('pointercancel', release);

  for (const b of document.querySelectorAll('#updown button')) {
    const v = +b.dataset.v;
    const on = (e) => { b.setPointerCapture(e.pointerId); b.classList.add('on'); camera.touch.vertical = v; };
    const off = () => { b.classList.remove('on'); if (camera.touch.vertical === v) camera.touch.vertical = 0; };
    b.addEventListener('pointerdown', on);
    b.addEventListener('pointerup', off);
    b.addEventListener('pointercancel', off);
  }
  // keep long-presses from opening the context menu on the controls
  document.getElementById('touch').addEventListener('contextmenu', (e) => e.preventDefault());
}

function fail(msg) {
  document.getElementById('status').innerHTML = msg;
  document.getElementById('status').classList.add('error');
}

main();
