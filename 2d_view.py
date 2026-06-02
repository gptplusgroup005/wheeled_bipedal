import re
import time
import threading
import queue

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import serial
from serial.tools import list_ports

# ===================== Geometry (meters) =====================
d = 0.0685
La = 0.070
Lp = 0.100

A1 = np.array([0.0, 0.0])
A2 = np.array([d, 0.0])
EPS = 1e-9
DEFAULT_START_Q_DEG = np.array([90.0, 90.0], float)
VIEW_FILL_RATIO = 0.75

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def circle_intersections(c1, r1, c2, r2):
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    dist = np.linalg.norm(c2 - c1)
    if dist > r1 + r2 + 1e-12:
        return None
    if dist < abs(r1 - r2) - 1e-12:
        return None
    if dist < 1e-12 and abs(r1 - r2) < 1e-12:
        return None

    a = (r1 * r1 - r2 * r2 + dist * dist) / (2 * dist)
    h2 = r1 * r1 - a * a
    if h2 < -1e-12:
        return None
    h = np.sqrt(max(0.0, h2))

    p0 = c1 + a * (c2 - c1) / dist
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / dist
    return p0 + h * perp, p0 - h * perp

def fk(q, branch=+1):
    th1, th2 = q
    B1 = A1 + La * np.array([np.cos(th1), np.sin(th1)])
    B2 = A2 + La * np.array([np.cos(th2), np.sin(th2)])

    pts = circle_intersections(B1, Lp, B2, Lp)
    if pts is None:
        return None, None, None

    p1, p2 = pts
    p_high = p1 if p1[1] >= p2[1] else p2
    p_low = p2 if p1[1] >= p2[1] else p1
    return B1, B2, p_high if branch == +1 else p_low

def ik_2r_first_joint(A, P, elbow=+1):
    r = P - A
    dist = np.linalg.norm(r)
    if dist < EPS:
        return None
    if dist > La + Lp + 1e-12 or dist < abs(La - Lp) - 1e-12:
        return None

    phi = np.arctan2(r[1], r[0])
    c = (La * La + dist * dist - Lp * Lp) / (2 * La * dist)
    alpha = np.arccos(np.clip(c, -1.0, 1.0))
    return wrap_pi(phi + elbow * alpha)

def ik_5bar(P, prev_q=None, prefer=(+1, -1)):
    sols = []
    for e1 in (+1, -1):
        th1 = ik_2r_first_joint(A1, P, elbow=e1)
        if th1 is None:
            continue
        for e2 in (+1, -1):
            th2 = ik_2r_first_joint(A2, P, elbow=e2)
            if th2 is None:
                continue
            sols.append((th1, th2, e1, e2))
    if not sols:
        return None, None

    if prev_q is None:
        for th1, th2, e1, e2 in sols:
            if (e1, e2) == prefer:
                return np.array([th1, th2], float), (e1, e2)
        th1, th2, e1, e2 = sols[0]
        return np.array([th1, th2], float), (e1, e2)

    best = None
    best_meta = None
    best_cost = float("inf")
    for th1, th2, e1, e2 in sols:
        cost = abs(wrap_pi(th1 - prev_q[0])) + abs(wrap_pi(th2 - prev_q[1]))
        if cost < best_cost:
            best_cost = cost
            best = np.array([th1, th2], float)
            best_meta = (e1, e2)
    return best, best_meta

PAIR_RE = re.compile(
    r"(?P<key>theta1|theta2|th1|th2|q1|q2|a1|a2|x|y|px|py|set_theta1|set_theta2|set_th1|set_th2)"
    r"\s*[:=]\s*(?P<val>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

def list_serial_ports():
    return [p.device for p in list_ports.comports()]

def parse_mcu_line(line):
    values = {}
    for m in PAIR_RE.finditer(line):
        values[m.group("key").lower()] = float(m.group("val"))

    def pick(*keys):
        for key in keys:
            if key in values:
                return values[key]
        return None

    th1 = pick("theta1", "th1", "q1", "a1")
    th2 = pick("theta2", "th2", "q2", "a2")
    set_th1 = pick("set_theta1", "set_th1")
    set_th2 = pick("set_theta2", "set_th2")
    x = pick("x", "px")
    y = pick("y", "py")

    if not values:
        return None
    return {
        "theta1": th1,
        "theta2": th2,
        "set_theta1": set_th1,
        "set_theta2": set_th2,
        "x": x,
        "y": y,
        "raw": line,
    }

class SerialWorker:
    def __init__(self, timeout=0.5, idle_sleep=0.05):
        self.ser = None
        self.thread = None
        self.stop_event = threading.Event()
        self.q = queue.Queue(maxsize=5000)
        self.last_line = ""
        self.t0 = None
        self.timeout = float(timeout)
        self.idle_sleep = float(idle_sleep)

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def connect(self, port, baud):
        if self.is_connected():
            return
        self.stop_event.clear()
        self.ser = serial.Serial(port, baudrate=baud, timeout=self.timeout)
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def disconnect(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.thread = None

    def send(self, text):
        if not self.is_connected():
            return
        self.ser.write((text.strip() + "\n").encode("utf-8"))

    def _run(self):
        while not self.stop_event.is_set():
            if self.ser is None:
                break
            try:
                raw = self.ser.readline()
                if not raw:
                    time.sleep(self.idle_sleep)
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.last_line = line
                parsed = parse_mcu_line(line)
                t = time.perf_counter() - self.t0
                try:
                    self.q.put_nowait((t, parsed, line))
                except queue.Full:
                    for _ in range(50):
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            break
                    self.q.put_nowait((t, parsed, line))
            except (serial.SerialException, OSError):
                break
            except Exception:
                continue

class FiveBarHardwareUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("5-Bar Parallel Hardware UI")
        self.geometry("1200x700")

        self.worker = SerialWorker()
        self.branch = +1
        self.angle_unit = tk.StringVar(value="deg")
        self.command_mode = tk.StringVar(value="joint")
        q_init = np.deg2rad(DEFAULT_START_Q_DEG)
        self.q_meas = q_init.copy()
        self.q_set = q_init.copy()
        _B1, _B2, p_init = fk(q_init, branch=self.branch)
        self.p_set = p_init if p_init is not None else np.array([d / 2, 0.107], float)
        self.setpoint_reachable = True
        self.p_meas = None
        self.dragging = False
        self._redraw_pending = False
        self._static_scene_drawn = False
        self._view_transform_cache = None

        self._syncing_controls = False
        self.last_rx_time = None
        self.last_raw = ""

        self._build_layout()
        self._refresh_ports()
        self._redraw()
        self.after(80, self._poll_serial)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.paned = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            sashwidth=7,
            sashrelief=tk.RAISED,
            bg="#d0d0d0",
            bd=0,
        )
        self.paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(self.paned)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right_container = ttk.Frame(self.paned)
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)

        self.paned.add(left, minsize=360, stretch="always")
        self.paned.add(right_container, minsize=360)

        self.canvas = tk.Canvas(left, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.control_canvas = tk.Canvas(right_container, highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(
            right_container,
            orient=tk.VERTICAL,
            command=self.control_canvas.yview,
        )
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        self.control_scrollbar.grid(row=0, column=1, sticky="ns")

        right = ttk.Frame(self.control_canvas, padding=8)
        right.bind(
            "<Configure>",
            lambda _e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            ),
        )
        self.control_canvas_window = self.control_canvas.create_window(
            (0, 0),
            window=right,
            anchor="nw",
        )
        self.control_canvas.bind("<Configure>", self._on_control_canvas_configure)
        self.control_canvas.bind("<Enter>", self._bind_control_mousewheel)
        self.control_canvas.bind("<Leave>", self._unbind_control_mousewheel)

        right.columnconfigure(0, weight=1)

        serial_box = ttk.LabelFrame(right, text="Serial")
        serial_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        serial_box.columnconfigure(1, weight=1)
        ttk.Label(serial_box, text="COM").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.cmb_port = ttk.Combobox(serial_box, width=18, state="readonly")
        self.cmb_port.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        ttk.Button(serial_box, text="Refresh", command=self._refresh_ports).grid(row=0, column=2, padx=6, pady=4)
        ttk.Label(serial_box, text="Baud").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        self.ent_baud = ttk.Entry(serial_box, width=12)
        self.ent_baud.insert(0, "115200")
        self.ent_baud.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        self.btn_connect = ttk.Button(serial_box, text="Connect", command=self._toggle_connect)
        self.btn_connect.grid(row=1, column=2, padx=6, pady=4)
        self.lbl_status = ttk.Label(serial_box, text="Disconnected")
        self.lbl_status.grid(row=2, column=0, columnspan=3, padx=6, pady=4, sticky="w")

        set_box = ttk.LabelFrame(right, text="Setpoint")
        set_box.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        set_box.columnconfigure(1, weight=1)
        ttk.Radiobutton(set_box, text="Joint", variable=self.command_mode, value="joint").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Radiobutton(set_box, text="XY", variable=self.command_mode, value="xy").grid(row=0, column=1, padx=6, pady=4, sticky="w")

        self.th1_set_var = tk.DoubleVar(value=float(np.rad2deg(self.q_set[0])))
        self.th2_set_var = tk.DoubleVar(value=float(np.rad2deg(self.q_set[1])))
        self.x_set_var = tk.DoubleVar(value=float(self.p_set[0]))
        self.y_set_var = tk.DoubleVar(value=float(self.p_set[1]))

        ttk.Label(set_box, text="theta1").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(set_box, textvariable=self.th1_set_var).grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        ttk.Label(set_box, text="theta2").grid(row=2, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(set_box, textvariable=self.th2_set_var).grid(row=2, column=1, padx=6, pady=4, sticky="ew")
        ttk.Label(set_box, text="unit").grid(row=3, column=0, padx=6, pady=4, sticky="w")
        self.cmb_angle_unit = ttk.Combobox(set_box, textvariable=self.angle_unit, values=["deg", "rad"], width=8, state="readonly")
        self.cmb_angle_unit.grid(row=3, column=1, padx=6, pady=4, sticky="w")
        self.cmb_angle_unit.bind("<<ComboboxSelected>>", self._on_angle_unit_change)
        ttk.Button(set_box, text="Submit Joint", command=self._submit_joint_setpoint).grid(row=3, column=2, padx=6, pady=4, sticky="ew")

        ttk.Label(set_box, text="x (m)").grid(row=4, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(set_box, textvariable=self.x_set_var).grid(row=4, column=1, padx=6, pady=4, sticky="ew")
        ttk.Label(set_box, text="y (m)").grid(row=5, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(set_box, textvariable=self.y_set_var).grid(row=5, column=1, padx=6, pady=4, sticky="ew")
        ttk.Button(set_box, text="Submit XY", command=self._submit_xy_setpoint).grid(row=5, column=2, padx=6, pady=4, sticky="ew")

        ttk.Button(set_box, text="Apply selected mode", command=self._apply_setpoint_to_ui).grid(row=6, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(set_box, text="Send Setpoint", command=self._send_setpoint).grid(row=6, column=1, columnspan=2, padx=6, pady=6, sticky="ew")
        ttk.Button(set_box, text="Stop", command=lambda: self._send_text("stop")).grid(row=7, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="ew")

        cmd_box = ttk.LabelFrame(right, text="Command format")
        cmd_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        cmd_box.columnconfigure(0, weight=1)
        self.cmd_template = tk.StringVar(value="q={theta1:.3f},{theta2:.3f}")
        ttk.Entry(cmd_box, textvariable=self.cmd_template).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.ent_raw_cmd = ttk.Entry(cmd_box)
        self.ent_raw_cmd.insert(0, "q=40,140")
        self.ent_raw_cmd.grid(row=1, column=0, padx=6, pady=(0, 6), sticky="ew")
        self.ent_raw_cmd.bind("<Return>", lambda _e: self._send_raw_cmd())
        ttk.Button(cmd_box, text="Send Raw", command=self._send_raw_cmd).grid(row=1, column=1, padx=6, pady=(0, 6))

        action_box = ttk.LabelFrame(right, text="View")
        action_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_box.columnconfigure(0, weight=1)
        action_box.columnconfigure(1, weight=1)
        ttk.Button(action_box, text="Flip branch", command=self._flip_branch).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(action_box, text="Use measured as setpoint", command=self._copy_measured_to_setpoint).grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        status_box = ttk.LabelFrame(right, text="Status")
        status_box.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        right.rowconfigure(4, weight=1)
        self.txt_status = tk.Text(status_box, height=10, wrap="word")
        self.txt_status.pack(fill="both", expand=True, padx=6, pady=6)

        log_box = ttk.LabelFrame(right, text="Serial log")
        log_box.grid(row=5, column=0, sticky="nsew")
        right.rowconfigure(5, weight=1)
        self.txt_log = tk.Text(log_box, height=8, wrap="none")
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)
        self.txt_log.configure(state=tk.DISABLED)

    def _on_control_canvas_configure(self, event):
        self.control_canvas.itemconfigure(self.control_canvas_window, width=event.width)

    def _bind_control_mousewheel(self, _event):
        self.control_canvas.bind_all("<MouseWheel>", self._on_control_mousewheel)
        self.control_canvas.bind_all("<Button-4>", self._on_control_mousewheel)
        self.control_canvas.bind_all("<Button-5>", self._on_control_mousewheel)

    def _unbind_control_mousewheel(self, _event):
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")

    def _on_control_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -int(event.delta / 120)
        self.control_canvas.yview_scroll(delta, "units")

    def _on_angle_unit_change(self, _event=None):
        self._sync_setpoint_angle_entries()
        self._request_redraw()

    def _refresh_ports(self):
        ports = list_serial_ports()
        self.cmb_port["values"] = ports
        if ports and not self.cmb_port.get():
            self.cmb_port.set(ports[0])
        if not ports:
            self.cmb_port.set("")

    def _toggle_connect(self):
        if self.worker.is_connected():
            self.worker.disconnect()
            self.btn_connect.configure(text="Connect")
            self.lbl_status.configure(text="Disconnected")
            return

        port = self.cmb_port.get().strip()
        if not port:
            messagebox.showerror("Error", "Chua chon COM.")
            return
        try:
            baud = int(self.ent_baud.get().strip())
            self.worker.connect(port, baud)
        except Exception as exc:
            messagebox.showerror("Error", f"Khong mo duoc {port}:\n{exc}")
            return
        self.btn_connect.configure(text="Disconnect")
        self.lbl_status.configure(text=f"Connected to {port} @ {baud}")

    def _send_text(self, text):
        if not self.worker.is_connected():
            messagebox.showwarning("Warning", "Chua ket noi COM.")
            return
        try:
            self.worker.send(text)
            self._append_log(f">> {text}")
        except Exception as exc:
            messagebox.showerror("Error", f"Gui lenh that bai:\n{exc}")

    def _send_raw_cmd(self):
        cmd = self.ent_raw_cmd.get().strip()
        if cmd:
            self._send_text(cmd)

    def _submit_joint_setpoint(self):
        old_mode = self.command_mode.get()
        self.command_mode.set("joint")
        ok = self._apply_setpoint_to_ui(show_error=True)
        if not ok:
            self.command_mode.set(old_mode)

    def _submit_xy_setpoint(self):
        old_mode = self.command_mode.get()
        self.command_mode.set("xy")
        ok = self._apply_setpoint_to_ui(show_error=True)
        if not ok:
            self.command_mode.set(old_mode)

    def _send_setpoint(self):
        if not self._apply_setpoint_to_ui(show_error=True):
            return
        if not self.setpoint_reachable:
            messagebox.showwarning("Warning", "Setpoint nam ngoai workspace, khong gui lenh.")
            return
        if self.command_mode.get() == "joint":
            if self.angle_unit.get() == "deg":
                th1 = float(np.rad2deg(self.q_set[0]))
                th2 = float(np.rad2deg(self.q_set[1]))
            else:
                th1 = float(self.q_set[0])
                th2 = float(self.q_set[1])
            data = {
                "theta1": th1,
                "theta2": th2,
                "x": float(self.p_set[0]),
                "y": float(self.p_set[1]),
            }
        else:
            data = {
                "theta1": float(np.rad2deg(self.q_set[0])),
                "theta2": float(np.rad2deg(self.q_set[1])),
                "x": float(self.p_set[0]),
                "y": float(self.p_set[1]),
            }

        try:
            cmd = self.cmd_template.get().format(**data)
        except Exception as exc:
            messagebox.showerror("Error", f"Command format khong hop le:\n{exc}")
            return
        self._send_text(cmd)

    def _apply_setpoint_to_ui(self, show_error=False):
        try:
            if self.command_mode.get() == "joint":
                th1 = float(self.th1_set_var.get())
                th2 = float(self.th2_set_var.get())
                if self.angle_unit.get() == "deg":
                    self.q_set = np.array([np.deg2rad(th1), np.deg2rad(th2)], float)
                else:
                    self.q_set = np.array([th1, th2], float)
                _, _, p = fk(self.q_set, branch=self.branch)
                if p is not None:
                    self.p_set = p
                    self.setpoint_reachable = True
                    self._syncing_controls = True
                    try:
                        self.x_set_var.set(float(p[0]))
                        self.y_set_var.set(float(p[1]))
                    finally:
                        self._syncing_controls = False
                else:
                    self.setpoint_reachable = False
                    raise ValueError("Joint setpoint khong dong duoc co cau 5-bar.")
            else:
                self.p_set = np.array([float(self.x_set_var.get()), float(self.y_set_var.get())], float)
                q_new, _meta = ik_5bar(self.p_set, prev_q=self.q_set, prefer=(+1, -1))
                if q_new is None:
                    self.setpoint_reachable = False
                    raise ValueError("XY setpoint nam ngoai workspace.")
                self.q_set = q_new
                self.setpoint_reachable = True
                self._sync_setpoint_angle_entries()
        except Exception as exc:
            if show_error:
                messagebox.showerror("Error", f"Setpoint khong hop le:\n{exc}")
            self._redraw()
            return False
        self._redraw()
        return True

    def _copy_measured_to_setpoint(self):
        self.q_set = self.q_meas.copy()
        _, _, p = fk(self.q_set, branch=self.branch)
        if p is not None:
            self.p_set = p
            self.setpoint_reachable = True
            self._syncing_controls = True
            try:
                self.x_set_var.set(float(p[0]))
                self.y_set_var.set(float(p[1]))
            finally:
                self._syncing_controls = False
        else:
            self.setpoint_reachable = False
        self._sync_setpoint_angle_entries()
        self._redraw()

    def _sync_setpoint_angle_entries(self):
        self._syncing_controls = True
        try:
            if self.angle_unit.get() == "deg":
                self.th1_set_var.set(float(np.rad2deg(self.q_set[0])))
                self.th2_set_var.set(float(np.rad2deg(self.q_set[1])))
            else:
                self.th1_set_var.set(float(self.q_set[0]))
                self.th2_set_var.set(float(self.q_set[1]))
        finally:
            self._syncing_controls = False

    def _flip_branch(self):
        self.branch *= -1
        self._redraw()

    def _poll_serial(self):
        got_any = False
        while True:
            try:
                t, parsed, line = self.worker.q.get_nowait()
            except queue.Empty:
                break

            got_any = True
            self.last_rx_time = t
            self.last_raw = line
            self._append_log(line)
            if parsed:
                self._update_from_parsed(parsed)

        if got_any:
            self._redraw()

        if not self.worker.is_connected() and self.btn_connect.cget("text") != "Connect":
            self.btn_connect.configure(text="Connect")
            self.lbl_status.configure(text="Disconnected")
        self.after(80, self._poll_serial)

    def _update_from_parsed(self, parsed):
        th1 = parsed.get("theta1")
        th2 = parsed.get("theta2")
        if th1 is not None and th2 is not None:
            if self.angle_unit.get() == "deg":
                self.q_meas = np.array([np.deg2rad(th1), np.deg2rad(th2)], float)
            else:
                self.q_meas = np.array([th1, th2], float)

        x = parsed.get("x")
        y = parsed.get("y")
        if x is not None and y is not None:
            self.p_meas = np.array([x, y], float)
        else:
            _, _, p = fk(self.q_meas, branch=self.branch)
            self.p_meas = p

        set_th1 = parsed.get("set_theta1")
        set_th2 = parsed.get("set_theta2")
        if set_th1 is not None and set_th2 is not None:
            if self.angle_unit.get() == "deg":
                self.q_set = np.array([np.deg2rad(set_th1), np.deg2rad(set_th2)], float)
            else:
                self.q_set = np.array([set_th1, set_th2], float)
            self._sync_setpoint_angle_entries()

    def _append_log(self, line):
        self.txt_log.configure(state=tk.NORMAL)
        self.txt_log.insert(tk.END, line + "\n")
        lines = int(self.txt_log.index("end-1c").split(".")[0])
        if lines > 300:
            self.txt_log.delete("1.0", f"{lines - 300}.0")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state=tk.DISABLED)

    def _on_mouse_down(self, event):
        self.dragging = True
        self.command_mode.set("xy")
        self._update_setpoint_from_mouse(event.x, event.y)

    def _on_mouse_drag(self, event):
        if self.dragging:
            self._update_setpoint_from_mouse(event.x, event.y)

    def _on_mouse_up(self, _event):
        self.dragging = False

    def _on_canvas_configure(self, _event):
        self._view_transform_cache = None
        self._draw_static_scene()
        self._redraw()

    def _update_setpoint_from_mouse(self, cx, cy):
        self.p_set = self._canvas_to_world(np.array([cx, cy], float))
        self._syncing_controls = True
        try:
            self.x_set_var.set(float(self.p_set[0]))
            self.y_set_var.set(float(self.p_set[1]))
        finally:
            self._syncing_controls = False
        q_new, _meta = ik_5bar(self.p_set, prev_q=self.q_set, prefer=(+1, -1))
        if q_new is not None:
            self.q_set = q_new
            self.setpoint_reachable = True
            self._sync_setpoint_angle_entries()
        else:
            self.setpoint_reachable = False
        self._request_redraw()

    def _view_content_bounds(self):
        if hasattr(self, "_view_bounds"):
            return self._view_bounds

        r_min = abs(Lp - La)
        r_max = La + Lp
        xs = [A1[0], A2[0]]
        ys = [A1[1], A2[1]]

        for center, radius in ((A1, La), (A2, La)):
            xs.extend([center[0] - radius, center[0] + radius])
            ys.extend([center[1] - radius, center[1] + radius])

        x_scan = np.linspace(A1[0] - r_max, A2[0] + r_max, 260)
        y_scan = np.linspace(-r_max, r_max, 260)
        xx, yy = np.meshgrid(x_scan, y_scan)
        r1 = np.hypot(xx - A1[0], yy - A1[1])
        r2 = np.hypot(xx - A2[0], yy - A2[1])
        mask = (r_min <= r1) & (r1 <= r_max) & (r_min <= r2) & (r2 <= r_max)
        if np.any(mask):
            xs.extend(xx[mask])
            ys.extend(yy[mask])

        self._view_bounds = (min(xs), max(xs), min(ys), max(ys))
        return self._view_bounds

    def _view_transform(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        if self._view_transform_cache and self._view_transform_cache[0] == (w, h):
            return self._view_transform_cache[1]

        x_min, x_max, y_min, y_max = self._view_content_bounds()
        world_w = max(x_max - x_min, EPS)
        world_h = max(y_max - y_min, EPS)
        scale = VIEW_FILL_RATIO * min(w / world_w, h / world_h)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        ox = w / 2 - cx * scale
        oy = h / 2 + cy * scale
        transform = (scale, ox, oy)
        self._view_transform_cache = ((w, h), transform)
        return transform

    def _world_to_canvas(self, p):
        scale, ox, oy = self._view_transform()
        return np.array([p[0] * scale + ox, oy - p[1] * scale])

    def _canvas_to_world(self, c):
        scale, ox, oy = self._view_transform()
        return np.array([(c[0] - ox) / scale, (oy - c[1]) / scale])

    def _redraw(self):
        self._redraw_pending = False
        if not self._static_scene_drawn:
            self._draw_static_scene()
        self.canvas.delete("dynamic")
        self._draw_robot(self.q_meas, "#222222", "#0066ff", "meas")
        self._draw_robot(self.q_set, "#999999", "#ff0000", "set", dash=(4, 3))
        target_color = "#ff0000" if self.setpoint_reachable else "#ff8800"
        target_label = "Pd" if self.setpoint_reachable else "Pd outside"
        self._draw_point(self.p_set, r=6, label=target_label, fill=target_color)
        if self.p_meas is not None:
            self._draw_point(self.p_meas, r=5, label="P rx", fill="#00a86b")
        self._draw_status()

    def _request_redraw(self):
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.after_idle(self._redraw)

    def _draw_static_scene(self):
        self.canvas.delete("static")
        self._draw_grid()
        self._draw_workspace_region()
        self._static_scene_drawn = True

    def _draw_grid(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        for x in range(0, w, 80):
            self.canvas.create_line(x, 0, x, h, fill="#f0f0f0", tags=("static",))
        for y in range(0, h, 80):
            self.canvas.create_line(0, y, w, y, fill="#f0f0f0", tags=("static",))

    def _draw_workspace_region(self):
        step = 14
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        scale, ox, oy = self._view_transform()
        r_min = abs(Lp - La)
        r_max = La + Lp

        x_cells = np.arange(0, w, step)
        y_cells = np.arange(0, h, step)
        if x_cells.size == 0 or y_cells.size == 0:
            return

        cx = x_cells + step / 2
        cy = y_cells + step / 2
        xx, yy = np.meshgrid(cx, cy)
        wx = (xx - ox) / scale
        wy = (oy - yy) / scale
        r1 = np.hypot(wx - A1[0], wy - A1[1])
        r2 = np.hypot(wx - A2[0], wy - A2[1])
        reachable_mask = (r_min <= r1) & (r1 <= r_max) & (r_min <= r2) & (r2 <= r_max)

        reachable_cells = set()
        rows, cols = np.nonzero(reachable_mask)
        for row, col in zip(rows, cols):
            x = int(x_cells[col])
            y = int(y_cells[row])
            reachable_cells.add((x, y))
            self.canvas.create_rectangle(
                x,
                y,
                min(x + step, w),
                min(y + step, h),
                fill="#edf8ef",
                outline="#edf8ef",
                tags=("static", "workspace"),
            )

        for x, y in reachable_cells:
            near_outside = False
            for nx, ny in ((x - step, y), (x + step, y), (x, y - step), (x, y + step)):
                if (nx, ny) not in reachable_cells:
                    near_outside = True
                    break
            if near_outside:
                self.canvas.create_rectangle(
                    x,
                    y,
                    min(x + step, w),
                    min(y + step, h),
                    fill="",
                    outline="#8cbf9a",
                    dash=(3, 3),
                    tags=("static", "workspace"),
                )

    def _is_reachable_point(self, p):
        r_min = abs(Lp - La)
        r_max = La + Lp
        r1 = np.linalg.norm(np.asarray(p, float) - A1)
        r2 = np.linalg.norm(np.asarray(p, float) - A2)
        return (r_min <= r1 <= r_max) and (r_min <= r2 <= r_max)

    def _draw_robot(self, q, link_color, point_color, label, dash=None):
        B1, B2, P = fk(q, branch=self.branch)
        self._draw_point(A1, r=5, label="A1", fill="#000000")
        self._draw_point(A2, r=5, label="A2", fill="#000000")
        if B1 is None or B2 is None or P is None:
            return
        self._draw_link(A1, B1, fill=link_color, width=3, dash=dash)
        self._draw_link(A2, B2, fill=link_color, width=3, dash=dash)
        self._draw_link(B1, P, fill=link_color, width=3, dash=dash)
        self._draw_link(B2, P, fill=link_color, width=3, dash=dash)
        self._draw_point(B1, r=4, label=f"B1 {label}", fill=point_color)
        self._draw_point(B2, r=4, label=f"B2 {label}", fill=point_color)
        self._draw_point(P, r=5, label=f"P {label}", fill=point_color)

    def _draw_point(self, p, r=4, label=None, fill="#000000"):
        c = self._world_to_canvas(np.array(p, float))
        x, y = c
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="", tags=("dynamic",))
        if label:
            self.canvas.create_text(x + 10, y - 10, text=label, fill="#333333", anchor="nw", tags=("dynamic",))

    def _draw_circle(self, center, radius, outline="#999999", width=1, dash=None):
        c = self._world_to_canvas(np.array(center, float))
        edge = self._world_to_canvas(np.array([center[0] + radius, center[1]], float))
        rp = abs(edge[0] - c[0])
        self.canvas.create_oval(
            c[0] - rp,
            c[1] - rp,
            c[0] + rp,
            c[1] + rp,
            outline=outline,
            width=width,
            dash=dash,
            tags=("dynamic",),
        )

    def _draw_link(self, p1, p2, fill="#222222", width=2, dash=None):
        c1 = self._world_to_canvas(np.array(p1, float))
        c2 = self._world_to_canvas(np.array(p2, float))
        self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill=fill, width=width, dash=dash, tags=("dynamic",))

    def _draw_status(self):
        _, _, p_calc = fk(self.q_meas, branch=self.branch)
        th_meas = np.rad2deg(self.q_meas)
        th_set = np.rad2deg(self.q_set)
        lines = [
            f"branch={self.branch}   unit={self.angle_unit.get()}",
            f"measured theta1={th_meas[0]:.2f} deg   theta2={th_meas[1]:.2f} deg",
            f"setpoint theta1={th_set[0]:.2f} deg   theta2={th_set[1]:.2f} deg",
        ]
        if p_calc is not None:
            lines.append(f"P from measured FK = [{p_calc[0]:.4f}, {p_calc[1]:.4f}] m")
        if self.p_meas is not None:
            lines.append(f"P received = [{self.p_meas[0]:.4f}, {self.p_meas[1]:.4f}] m")
        lines.append(f"Pd/set = [{self.p_set[0]:.4f}, {self.p_set[1]:.4f}] m")
        lines.append(f"setpoint reachable = {self.setpoint_reachable}")
        lines.append(f"last rx t = {self.last_rx_time:.2f} s" if self.last_rx_time is not None else "last rx t = --")
        if self.last_raw:
            lines.append(f"raw: {self.last_raw}")

        self.txt_status.delete("1.0", tk.END)
        self.txt_status.insert(tk.END, "\n".join(lines))

    def _on_close(self):
        try:
            self.worker.disconnect()
        finally:
            self.destroy()

if __name__ == "__main__":
    app = FiveBarHardwareUI()
    app.mainloop()
