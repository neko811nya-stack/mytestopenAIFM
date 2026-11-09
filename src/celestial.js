// Sun, moon and stars from the time of day and the moon's age.
// A fixed latitude / season keeps it simple: 35°N with the sun at +10° declination (late spring).
import { dirFromAngles } from './math.js';

const deg = Math.PI / 180;
const LAT = 35 * deg;
const SUN_DECLINATION = 10 * deg;
export const SYNODIC_MONTH = 29.53;
export const DEFAULT_HOUR = 18.3;      // ~2° sun elevation: the original sunset view
const DEFAULT_SUN_AZIMUTH = 20 * deg;  // world azimuth of the sun at DEFAULT_HOUR (where the camera looks)

// Elevation and compass azimuth (from north, clockwise) for an hour angle and declination.
function horizontal(hourAngle, decl) {
  const sinEl = Math.sin(LAT) * Math.sin(decl) + Math.cos(LAT) * Math.cos(decl) * Math.cos(hourAngle);
  const el = Math.asin(sinEl);
  const az = Math.atan2(Math.sin(hourAngle), Math.cos(hourAngle) * Math.sin(LAT) - Math.tan(decl) * Math.cos(LAT)) + Math.PI;
  return { el, az };
}

// The world is turned so that the sun sits where the camera looks at the default hour;
// compass clockwise maps to increasing world yaw (to the right).
const anchor = horizontal((DEFAULT_HOUR - 12) * 15 * deg, SUN_DECLINATION).az;
const toWorld = ({ el, az }) => dirFromAngles(el, az - anchor + DEFAULT_SUN_AZIMUTH);

export function celestial(hour, moonAge) {
  const H = (hour - 12) * 15 * deg;
  const sun = horizontal(H, SUN_DECLINATION);
  // elongation from the sun: 0 new moon, 180° full moon. The moon lags the sun by that angle,
  // and sits on the opposite side of the ecliptic, so a summer full moon rides low.
  const e = (moonAge / SYNODIC_MONTH) * 2 * Math.PI;
  const moon = horizontal(H - e, SUN_DECLINATION * Math.cos(e));
  return {
    sunDir: toWorld(sun),
    sunElevation: sun.el / deg,
    moonDir: toWorld(moon),
    moonIllum: (1 - Math.cos(e)) / 2,
    // the stars turn once per sidereal day
    starAngle: (hour / 23.934) * 2 * Math.PI,
  };
}

export function moonPhaseName(age) {
  const names = [[1.0, '新月'], [5.5, '三日月'], [9.0, '上弦'], [13.0, '十三夜'], [16.5, '満月'], [20.5, '寝待月'], [24.0, '下弦'], [28.5, '有明月'], [30, '新月']];
  return names.find(([limit]) => age < limit)[1];
}

export const formatHour = (h) => {
  const m = Math.round(((h % 24) + 24) % 24 * 60) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};
