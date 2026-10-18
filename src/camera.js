// Free-flying camera: WASD to move, Space/E up, Q/C down, Shift to boost, mouse drag (or pointer lock) to look.
export class FlyCamera {
  constructor(canvas, { position = [0, 6, 0], yaw = 0, pitch = 0 } = {}) {
    this.position = [...position];
    this.yaw = yaw;
    this.pitch = pitch;
    this.fov = 55 * Math.PI / 180;
    this.speed = 12;
    this.keys = new Set();
    this.velocity = [0, 0, 0];
    let dragging = false, lx = 0, ly = 0;
    const look = (dx, dy) => {
      this.yaw += dx * 0.0025;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - dy * 0.0025));
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
      this.speed = Math.max(1, Math.min(400, this.speed * Math.exp(-e.deltaY * 0.001)));
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

  update(dt) {
    const k = this.keys;
    const f = this.forward();
    const right = [-Math.sin(this.yaw), 0, Math.cos(this.yaw)];
    const want = [0, 0, 0];
    const add = (v, s) => { want[0] += v[0] * s; want[1] += v[1] * s; want[2] += v[2] * s; };
    if (k.has('KeyW') || k.has('ArrowUp')) add(f, 1);
    if (k.has('KeyS') || k.has('ArrowDown')) add(f, -1);
    if (k.has('KeyD') || k.has('ArrowRight')) add(right, 1);
    if (k.has('KeyA') || k.has('ArrowLeft')) add(right, -1);
    if (k.has('Space') || k.has('KeyE')) want[1] += 1;
    if (k.has('KeyQ') || k.has('KeyC')) want[1] -= 1;
    const boost = k.has('ShiftLeft') || k.has('ShiftRight') ? 5 : 1;
    const s = this.speed * boost;
    // critically damped velocity for smooth starts and stops
    const a = 1 - Math.exp(-dt * 8);
    for (let i = 0; i < 3; i++) {
      this.velocity[i] += (want[i] * s - this.velocity[i]) * a;
      this.position[i] += this.velocity[i] * dt;
    }
    this.position[1] = Math.max(1.2, Math.min(1200, this.position[1]));
  }
}
