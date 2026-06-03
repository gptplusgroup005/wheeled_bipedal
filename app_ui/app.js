const sceneCanvas = document.getElementById("scene");
const triadCanvas = document.getElementById("triad");
const sceneCtx = sceneCanvas.getContext("2d");
const triadCtx = triadCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const telemetryEl = document.getElementById("telemetry");
const cameraMode = document.getElementById("cameraMode");

const jointKeys = [
  "left_thigh_a_deg",
  "left_thigh_b_deg",
  "right_thigh_a_deg",
  "right_thigh_b_deg",
];

const cameraPresets = {
  ISO: { elev: 30, azim: 135, roll: 0, span: 1.35 },
  Front: { elev: 90, azim: 0, roll: 0, span: 1.35 },
  Side: { elev: 90, azim: -90, roll: 0, span: 1.35 },
  Top: { elev: 0, azim: 0, roll: 0, span: 1.35 },
};

const colors = {
  sceneBg: "#0d1117",
  grid: "#21262d",
  balanceFill: "rgba(56, 139, 253, 0.22)",
  balanceStroke: "#58a6ff",
  balanceText: "#79c0ff",
  pointStroke: "#0d1117",
  pointText: "#c9d1d9",
  triadBg: "#0d1117",
};

let scene = null;
let camera = { mode: "ISO", ...cameraPresets.ISO };
let dragging = null;

function degToRad(value) {
  return (value * Math.PI) / 180;
}

function rotatePoint(point, cam = camera) {
  const [x0, y0, z0] = point;
  const az = degToRad(cam.azim);
  const el = degToRad(cam.elev);
  const ca = Math.cos(-az);
  const sa = Math.sin(-az);
  const ce = Math.cos(-el);
  const se = Math.sin(-el);
  const x1 = ca * x0 - sa * y0;
  const y1 = sa * x0 + ca * y0;
  const z1 = z0;
  const y2 = ce * y1 - se * z1;
  const z2 = se * y1 + ce * z1;
  return [x1, y2, z2];
}

function project(point, width, height, center, span, cam = camera) {
  const shifted = [point[0] - center[0], point[1] - center[1], point[2] - center[2]];
  const rotated = rotatePoint(shifted, cam);
  const scale = Math.min(width, height) / (span * 2);
  return {
    x: width * 0.5 + rotated[0] * scale,
    y: height * 0.5 - rotated[1] * scale,
    z: rotated[2],
  };
}

function collectPoints(linkage) {
  const points = [];
  for (const line of linkage.grid) points.push(line.a, line.b);
  for (const line of linkage.lines) points.push(line.a, line.b);
  for (const point of linkage.points) points.push(point.p);
  for (const wheel of linkage.wheels) points.push(wheel.center);
  if (linkage.balance) points.push(...linkage.balance.corners, linkage.balance.center);
  return points;
}

function sceneCenter(linkage) {
  const points = collectPoints(linkage);
  if (!points.length) return [0, 0, 0.6];
  const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]], [0, 0, 0]);
  return sum.map((value) => value / points.length);
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function drawLine(ctx, a, b, style) {
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function drawScene() {
  const rect = resizeCanvas(sceneCanvas);
  sceneCtx.clearRect(0, 0, rect.width, rect.height);
  sceneCtx.fillStyle = colors.sceneBg;
  sceneCtx.fillRect(0, 0, rect.width, rect.height);
  if (!scene || !scene.ok) return;

  const linkage = scene.linkage;
  const center = sceneCenter(linkage);
  const span = camera.span;
  const projected = (point) => project(point, rect.width, rect.height, center, span);

  for (const line of linkage.grid) {
    drawLine(sceneCtx, projected(line.a), projected(line.b), { color: colors.grid, width: 1 });
  }

  if (linkage.balance) {
    const corners = linkage.balance.corners.map(projected);
    sceneCtx.fillStyle = colors.balanceFill;
    sceneCtx.strokeStyle = colors.balanceStroke;
    sceneCtx.lineWidth = 1.5;
    sceneCtx.beginPath();
    sceneCtx.moveTo(corners[0].x, corners[0].y);
    for (const corner of corners.slice(1)) sceneCtx.lineTo(corner.x, corner.y);
    sceneCtx.closePath();
    sceneCtx.fill();
    sceneCtx.stroke();
    const label = projected(linkage.balance.center);
    sceneCtx.fillStyle = colors.balanceText;
    sceneCtx.font = "13px Segoe UI";
    sceneCtx.textAlign = "center";
    sceneCtx.fillText("BALANCE PLANE", label.x, label.y - 14);
    sceneCtx.fillText(`pitch=${linkage.balance.pitch.toFixed(1)} roll=${linkage.balance.roll.toFixed(1)}`, label.x, label.y);
  }

  const drawableLines = linkage.lines
    .map((line) => ({ line, pa: projected(line.a), pb: projected(line.b) }))
    .sort((a, b) => (a.pa.z + a.pb.z) - (b.pa.z + b.pb.z));
  for (const item of drawableLines) {
    drawLine(sceneCtx, item.pa, item.pb, { color: item.line.color, width: item.line.width });
  }

  for (const wheel of linkage.wheels) {
    const p = projected(wheel.center);
    const radius = Math.max(14, (wheel.radius * Math.min(rect.width, rect.height)) / (span * 2));
    sceneCtx.strokeStyle = wheel.color;
    sceneCtx.lineWidth = 4;
    sceneCtx.beginPath();
    sceneCtx.ellipse(p.x, p.y, radius * 0.72, radius, 0, 0, Math.PI * 2);
    sceneCtx.stroke();
  }

  for (const point of linkage.points) {
    const p = projected(point.p);
    sceneCtx.fillStyle = point.color;
    sceneCtx.strokeStyle = colors.pointStroke;
    sceneCtx.lineWidth = 2;
    sceneCtx.beginPath();
    sceneCtx.arc(p.x, p.y, point.label === "P" ? 6 : 5, 0, Math.PI * 2);
    sceneCtx.fill();
    sceneCtx.stroke();
    sceneCtx.fillStyle = colors.pointText;
    sceneCtx.font = "13px Segoe UI";
    sceneCtx.textAlign = "left";
    sceneCtx.fillText(point.label, p.x + 6, p.y - 6);
  }

  drawTriad();
  updateTelemetry();
}

function drawTriad() {
  const rect = resizeCanvas(triadCanvas);
  triadCtx.clearRect(0, 0, rect.width, rect.height);
  triadCtx.fillStyle = colors.triadBg;
  triadCtx.fillRect(0, 0, rect.width, rect.height);

  const origin = { x: rect.width * 0.45, y: rect.height * 0.62 };
  const axes = [
    { label: "X", p: [0.58, 0, 0], color: "#dc2626" },
    { label: "Y", p: [0, 0.58, 0], color: "#16a34a" },
    { label: "Z", p: [0, 0, 0.86], color: "#1d4ed8" },
  ];
  for (const axis of axes) {
    const r = rotatePoint(axis.p);
    const end = { x: origin.x + r[0] * 48, y: origin.y - r[1] * 48 };
    triadCtx.strokeStyle = axis.color;
    triadCtx.fillStyle = axis.color;
    triadCtx.lineWidth = 3;
    triadCtx.beginPath();
    triadCtx.moveTo(origin.x, origin.y);
    triadCtx.lineTo(end.x, end.y);
    triadCtx.stroke();
    const angle = Math.atan2(end.y - origin.y, end.x - origin.x);
    triadCtx.beginPath();
    triadCtx.moveTo(end.x, end.y);
    triadCtx.lineTo(end.x - Math.cos(angle - 0.45) * 9, end.y - Math.sin(angle - 0.45) * 9);
    triadCtx.lineTo(end.x - Math.cos(angle + 0.45) * 9, end.y - Math.sin(angle + 0.45) * 9);
    triadCtx.closePath();
    triadCtx.fill();
    triadCtx.font = "12px Segoe UI";
    triadCtx.fillText(axis.label, end.x + 5, end.y + 3);
  }
}

function updateTelemetry() {
  if (!scene || !scene.ok) {
    telemetryEl.textContent = scene?.error || "Robot data unavailable";
    return;
  }
  const balance = scene.linkage.balance || { pitch: 0, roll: 0, tilt: 0 };
  const lines = [
    `solver        ${scene.solver.backend}`,
    `closure L/R   ${scene.closure.left.toFixed(5)} / ${scene.closure.right.toFixed(5)} m`,
    `active L      ${scene.angles.left_thigh_a_deg.toFixed(2)} / ${scene.angles.left_thigh_b_deg.toFixed(2)} deg`,
    `active R      ${scene.angles.right_thigh_a_deg.toFixed(2)} / ${scene.angles.right_thigh_b_deg.toFixed(2)} deg`,
    `passive L     ${scene.angles.left_calf_a_deg.toFixed(2)} / ${scene.angles.left_calf_b_deg.toFixed(2)} deg`,
    `passive R     ${scene.angles.right_calf_a_deg.toFixed(2)} / ${scene.angles.right_calf_b_deg.toFixed(2)} deg`,
    `balance       pitch ${balance.pitch.toFixed(2)}  roll ${balance.roll.toFixed(2)} deg`,
    `tilt          ${balance.tilt.toFixed(2)} deg`,
    "",
    `urdf          ${scene.meta.urdf}`,
    `links/joints  ${scene.meta.links} / ${scene.meta.joints}`,
    `meshes        ${scene.meta.meshes}`,
  ];
  telemetryEl.textContent = lines.join("\n");
  statusEl.textContent = `solver=${scene.solver.backend}   closure L=${scene.closure.left.toFixed(5)} m   R=${scene.closure.right.toFixed(5)} m`;
}

function setCameraMode(mode) {
  camera = { mode, ...cameraPresets[mode] };
  cameraMode.value = mode;
  drawScene();
}

function syncAngles(angles) {
  for (const key of Object.keys(angles)) {
    const range = document.getElementById(key);
    const number = document.getElementById(`${key}_num`);
    const output = document.getElementById(`${key}_out`);
    if (range) range.value = angles[key];
    if (number) number.value = angles[key];
    if (output) output.textContent = Number(angles[key]).toFixed(1);
  }
}

async function refresh() {
  scene = await window.pywebview.api.get_scene();
  if (scene.ok) syncAngles(scene.angles);
  drawScene();
}

async function updateJoint(key, value) {
  scene = await window.pywebview.api.update_angles({ [key]: Number(value) });
  if (scene.ok) syncAngles(scene.angles);
  drawScene();
}

function wireControls() {
  for (const key of jointKeys) {
    const range = document.getElementById(key);
    const number = document.getElementById(`${key}_num`);
    const output = document.getElementById(`${key}_out`);
    const apply = (value) => {
      output.textContent = Number(value).toFixed(1);
      range.value = value;
      number.value = value;
      updateJoint(key, value);
    };
    range.addEventListener("input", () => apply(range.value));
    number.addEventListener("change", () => apply(number.value));
  }
  document.getElementById("resetAngles").addEventListener("click", async () => {
    scene = await window.pywebview.api.reset_angles();
    if (scene.ok) syncAngles(scene.angles);
    drawScene();
  });
  document.getElementById("resetCamera").addEventListener("click", () => setCameraMode("ISO"));
  document.getElementById("redraw").addEventListener("click", refresh);
  cameraMode.addEventListener("change", () => setCameraMode(cameraMode.value));

  sceneCanvas.addEventListener("mousedown", (event) => {
    if (event.button !== 1) return;
    dragging = { x: event.clientX, y: event.clientY, elev: camera.elev, azim: camera.azim };
    event.preventDefault();
  });
  window.addEventListener("mouseup", () => {
    dragging = null;
  });
  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    camera.elev = Math.max(-89, Math.min(89, dragging.elev - (event.clientY - dragging.y) * 0.35));
    camera.azim = dragging.azim - (event.clientX - dragging.x) * 0.35;
    drawScene();
  });
  sceneCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    camera.span = Math.max(0.45, Math.min(8, camera.span * (event.deltaY < 0 ? 0.9 : 1.1)));
    drawScene();
  });
  window.addEventListener("resize", drawScene);
}

window.addEventListener("pywebviewready", () => {
  wireControls();
  refresh();
});
