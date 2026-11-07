// Camera with three modes:
//   free   — fly anywhere: WASD to move, Space/E up, Q/C down, Shift to boost, drag (or pointer lock) to look
//   lock   — fly freely, but the view stays aimed at the target
//   follow — keep a fixed distance / bearing to the target at a fixed altitude (no wave bobbing);
//            drag or A/D orbits, W/S or the wheel changes distance, Space/Q changes altitude
// Touch devices feed the same controls through `touch` (virtual stick + up/down buttons).
export class FlyCamera {
  constructor(canvas, { position = [0, 6, 0], yaw = 0, pitch = 0 } = {}) {
    this.position = [...position];
    this.yaw = yaw;
    this.pitch = pitch;
    this.fov = 55 * Math.PI / 180;
    this.speed = 12;
    this.keys = new Set();
    this.velocity = [0, 0, 0];
    this.touch = { x: 0, y: 0, vertical: 0 };
    this.mode = 'free';
    this.follow = { distance: 30, bearing: 0, altitude: 6 };
    let dragging = false, lx = 0, ly = 0;
    const look = (dx, dy) => {
      if (this.mode === 'follow') {
        this.follow.bearing += dx * 0.005;
        this.follow.altitude = clampAltitude(this.follow.altitude + dy * 0.05 * Math.max(1, this.follow.distance / 30));
      } else if (this.mode === 'free') {
        this.yaw += dx * 0.0025;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - dy * 0.0025));
      }
    };
    canvas.addEventListener('pointerdown', (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointerup', (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', (e) => {
      if (document.pointerLockElement === canvas) return look(e.movementX, e.movementY);
      if (!dragging) return;
      look(e.clientX - lx, e.clientY - ly);
      lx = e.clientX; ly = e.clientY;
    });
    canvas.addEventListener('dblclick', () => canvas.requestPointerLock?.());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.mode === 'follow') this.follow.distance = clampDistance(this.follow.distance * Math.exp(e.deltaY * 0.001));
      else this.speed = Math.max(1, Math.min(400, this.speed * Math.exp(-e.deltaY * 0.001)));
    }, { passive: false });
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  forward() {
    const cp = Math.cos(this.pitch);
    return [cp * Math.cos(this.yaw), Math.sin(this.pitch), cp * Math.sin(this.yaw)];
  }

  // Switch mode; `target` is the target's current position (needed for lock / follow).
  setMode(mode, target) {
    if (mode !== 'free' && !target) mode = 'free';
    if (mode === 'follow') {
      // start from wherever the camera is now, relative to the target
      const dx = this.position[0] - target[0], dz = this.position[2] - target[2];
      this.follow.distance = clampDistance(Math.hypot(dx, dz));
      this.follow.bearing = Math.atan2(dz, dx);
      this.follow.altitude = clampAltitude(this.position[1]);
    }
    this.velocity = [0, 0, 0];
    this.mode = mode;
  }

  update(dt, target) {
    if (this.mode !== 'free' && !target) this.mode = 'free';
    const k = this.keys;
    const axis = (pos, neg) => (pos.some((c) => k.has(c)) ? 1 : 0) - (neg.some((c) => k.has(c)) ? 1 : 0);
    const ahead = axis(['KeyW', 'ArrowUp'], ['KeyS', 'ArrowDown']) - this.touch.y;
    const side = axis(['KeyD', 'ArrowRight'], ['KeyA', 'ArrowLeft']) + this.touch.x;
    const up = axis(['Space', 'KeyE'], ['KeyQ', 'KeyC']) + this.touch.vertical;
    const boost = k.has('ShiftLeft') || k.has('ShiftRight') ? 5 : 1;

    if (this.mode === 'follow') {
      const f = this.follow;
      f.distance = clampDistance(f.distance * Math.exp(-ahead * dt * boost));
      f.bearing += side * dt * 0.8 * boost;
      f.altitude = clampAltitude(f.altitude + up * dt * this.speed * boost);
      this.position = [target[0] + Math.cos(f.bearing) * f.distance, f.altitude, target[2] + Math.sin(f.bearing) * f.distance];
    } else {
      const f = this.forward();
      const right = [-Math.sin(this.yaw), 0, Math.cos(this.yaw)];
      const want = [f[0] * ahead + right[0] * side, f[1] * ahead + up, f[2] * ahead + right[2] * side];
      const s = this.speed * boost;
      // critically damped velocity for smooth starts and stops
      const a = 1 - Math.exp(-dt * 8);
      for (let i = 0; i < 3; i++) {
        this.velocity[i] += (want[i] * s - this.velocity[i]) * a;
        this.position[i] += this.velocity[i] * dt;
      }
      this.position[1] = clampAltitude(this.position[1]);
    }

    if (this.mode !== 'free') {
      const d = [target[0] - this.position[0], target[1] - this.position[1], target[2] - this.position[2]];
      this.yaw = Math.atan2(d[2], d[0]);
      this.pitch = Math.max(-1.45, Math.min(1.45, Math.atan2(d[1], Math.hypot(d[0], d[2]))));
    }
  }
}

const clampAltitude = (y) => Math.max(1.2, Math.min(1200, y));
const clampDistance = (d) => Math.max(6, Math.min(3000, d));
