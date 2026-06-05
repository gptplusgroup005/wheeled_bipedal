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
  Front: { elev: 90, azim: -90, roll: 0, span: 1.35 },
  Side: { elev: 90, azim: 0, roll: 0, span: 1.35 },
  Top: { elev: 0, azim: 0, roll: 0, span: 1.35 },
};

const colors = {
  sceneBg: "#0d1117",
  grid: "#21262d",
  balanceFill: "rgba(56, 139, 253, 0.15)",
  balanceStroke: "#58a6ff",
  balanceText: "#79c0ff",
  pointStroke: "#0d1117",
  pointText: "#c9d1d9",
  triadBg: "#0d1117",
};

const wheelUnitCircle = Array.from({ length: 45 }, (_, index) => {
  const t = (index / 44) * Math.PI * 2;
  return [Math.cos(t), Math.sin(t)];
});

let scene = null;
let sceneCenterCache = null;
let camera = { mode: "ISO", ...cameraPresets.ISO };
let dragging = null;
let framePending = false;
let updateInFlight = false;
let updateScheduled = false;
let pendingAngleValues = {};

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

function makeProjector(width, height, center, span, cam = camera) {
  const az = degToRad(cam.azim);
  const el = degToRad(cam.elev);
  const ca = Math.cos(-az);
  const sa = Math.sin(-az);
  const ce = Math.cos(-el);
  const se = Math.sin(-el);
  const scale = Math.min(width, height) / (span * 2);
  return (point) => {
    const x0 = point[0] - center[0];
    const y0 = point[1] - center[1];
    const z0 = point[2] - center[2];
    const x1 = ca * x0 - sa * y0;
    const y1 = sa * x0 + ca * y0;
    const y2 = ce * y1 - se * z0;
    const z2 = se * y1 + ce * z0;
    return {
      x: width * 0.5 + x1 * scale,
      y: height * 0.5 - y2 * scale,
      z: z2,
    };
  };
}

function sceneCenter(linkage) {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let count = 0;
  const add = (point) => {
    sx += point[0];
    sy += point[1];
    sz += point[2];
    count += 1;
  };
  for (const line of linkage.grid) {
    add(line.a);
    add(line.b);
  }
  for (const line of linkage.lines) {
    add(line.a);
    add(line.b);
  }
  for (const point of linkage.points) add(point.p);
  for (const wheel of linkage.wheels) {
    add(wheel.center);
    if (wheel.contact) add(wheel.contact);
  }
  if (linkage.balance) {
    for (const corner of linkage.balance.corners) add(corner);
    add(linkage.balance.center);
  }
  return count ? [sx / count, sy / count, sz / count] : [0, 0, 0.6];
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height || canvas.dataset.dpr !== String(dpr)) {
    canvas.width = width;
    canvas.height = height;
    canvas.dataset.dpr = String(dpr);
    canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return rect;
}

function drawLine(ctx, a, b, style) {
  const hasAlpha = style.alpha !== undefined && style.alpha < 1;
  if (hasAlpha) {
    ctx.save();
    ctx.globalAlpha = style.alpha;
  }
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  if (hasAlpha) ctx.restore();
}

function wheelPoint(wheel, cosValue, sinValue, axialOffset = 0) {
  const r = wheel.radius;
  return [
    wheel.center[0] + wheel.axisX[0] * cosValue * r + wheel.axisZ[0] * sinValue * r + wheel.axisY[0] * axialOffset,
    wheel.center[1] + wheel.axisX[1] * cosValue * r + wheel.axisZ[1] * sinValue * r + wheel.axisY[1] * axialOffset,
    wheel.center[2] + wheel.axisX[2] * cosValue * r + wheel.axisZ[2] * sinValue * r + wheel.axisY[2] * axialOffset,
  ];
}

function drawProjectedRing(ctx, wheel, projected, axialOffset, color, width, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i < wheelUnitCircle.length; i += 1) {
    const [cosValue, sinValue] = wheelUnitCircle[i];
    const p = projected(wheelPoint(wheel, cosValue, sinValue, axialOffset));
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawWheel(ctx, wheel, projected) {
  if (!wheel.axisX || !wheel.axisY || !wheel.axisZ) {
    const p = projected(wheel.center);
    ctx.strokeStyle = wheel.color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }

  const halfWidth = wheel.halfWidth || 0;
  const backOffset = -halfWidth;
  const frontOffset = halfWidth;
  drawProjectedRing(ctx, wheel, projected, backOffset, "#6e7681", 2, 0.65);
  drawProjectedRing(ctx, wheel, projected, frontOffset, wheel.color, 4, 1);

  for (const angle of [0, Math.PI * 0.5, Math.PI, Math.PI * 1.5]) {
    drawLine(
      ctx,
      projected(wheelPoint(wheel, Math.cos(angle), Math.sin(angle), frontOffset)),
      projected(wheel.center),
      { color: "#8b949e", width: 1.5 },
    );
  }

  drawLine(
    ctx,
    projected([
      wheel.center[0] - wheel.axisY[0] * halfWidth,
      wheel.center[1] - wheel.axisY[1] * halfWidth,
      wheel.center[2] - wheel.axisY[2] * halfWidth,
    ]),
    projected([
      wheel.center[0] + wheel.axisY[0] * halfWidth,
      wheel.center[1] + wheel.axisY[1] * halfWidth,
      wheel.center[2] + wheel.axisY[2] * halfWidth,
    ]),
    { color: "#c9d1d9", width: 2 },
  );

  if (wheel.contact) {
    const contact = projected(wheel.contact);
    ctx.fillStyle = "#d29922";
    ctx.strokeStyle = "#0d1117";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(contact.x, contact.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawScene() {
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    drawSceneNow();
  });
}

function drawSceneNow() {
  const rect = resizeCanvas(sceneCanvas);
  sceneCtx.clearRect(0, 0, rect.width, rect.height);
  sceneCtx.fillStyle = colors.sceneBg;
  sceneCtx.fillRect(0, 0, rect.width, rect.height);
  if (!scene || !scene.ok) return;

  const linkage = scene.linkage;
  const center = sceneCenterCache || sceneCenter(linkage);
  const span = camera.span;
  const projected = makeProjector(rect.width, rect.height, center, span);

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
    drawWheel(sceneCtx, wheel, projected);
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

function setScene(nextScene) {
  scene = nextScene;
  sceneCenterCache = scene?.ok ? sceneCenter(scene.linkage) : null;
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
  setScene(await window.pywebview.api.get_scene());
  if (scene.ok) syncAngles(scene.angles);
  drawScene();
}

async function flushAngleUpdates() {
  if (updateInFlight || !Object.keys(pendingAngleValues).length) return;
  const values = pendingAngleValues;
  pendingAngleValues = {};
  updateInFlight = true;
  try {
    setScene(await window.pywebview.api.update_angles(values));
    if (scene.ok) syncAngles(scene.angles);
    drawScene();
  } finally {
    updateInFlight = false;
    if (Object.keys(pendingAngleValues).length) scheduleAngleFlush();
  }
}

function scheduleAngleFlush() {
  if (updateScheduled) return;
  updateScheduled = true;
  requestAnimationFrame(() => {
    updateScheduled = false;
    flushAngleUpdates();
  });
}

function updateJoint(key, value) {
  pendingAngleValues[key] = Number(value);
  scheduleAngleFlush();
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
    setScene(await window.pywebview.api.reset_angles());
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
