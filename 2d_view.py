import tkinter as tk
from tkinter import ttk
import numpy as np
import time

# ===================== Geometry (meters) =====================
d  = 0.115     # A1A2
La = 0.050     # active
Lp = 0.100     # passive

A1 = np.array([0.0, 0.0])
A2 = np.array([d,   0.0])

EPS = 1e-9

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def circle_intersections(c1, r1, c2, r2):
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    D = np.linalg.norm(c2 - c1)
    if D > r1 + r2 + 1e-12: return None
    if D < abs(r1 - r2) - 1e-12: return None
    if D < 1e-12 and abs(r1 - r2) < 1e-12: return None

    a = (r1*r1 - r2*r2 + D*D) / (2*D)
    h2 = r1*r1 - a*a
    if h2 < -1e-12: return None
    h = np.sqrt(max(0.0, h2))

    p0 = c1 + a * (c2 - c1) / D
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / D

    p_int1 = p0 + h * perp
    p_int2 = p0 - h * perp
    return p_int1, p_int2

def fk(q, branch=+1):
    th1, th2 = q
    B1 = A1 + La * np.array([np.cos(th1), np.sin(th1)])
    B2 = A2 + La * np.array([np.cos(th2), np.sin(th2)])

    pts = circle_intersections(B1, Lp, B2, Lp)
    if pts is None:
        return None, None, None

    p1, p2 = pts
    # branch by y: +1 => pick higher y
    P = p1 if (p1[1] >= p2[1]) else p2
    if branch == -1:
        P = p2 if (p1[1] >= p2[1]) else p1

    return B1, B2, P

def ik_2R_first_joint(A, P, elbow=+1):
    r = P - A
    R = np.linalg.norm(r)
    if R < EPS: return None
    if R > La + Lp + 1e-12 or R < abs(La - Lp) - 1e-12:
        return None

    phi = np.arctan2(r[1], r[0])
    c = (La*La + R*R - Lp*Lp) / (2*La*R)
    c = np.clip(c, -1.0, 1.0)
    alpha = np.arccos(c)
    theta = phi + elbow * alpha
    return wrap_pi(theta)

def ik_5bar(P, prev_q=None, prefer=(+1, -1)):
    sols = []
    for e1 in (+1, -1):
        th1 = ik_2R_first_joint(A1, P, elbow=e1)
        if th1 is None: continue
        for e2 in (+1, -1):
            th2 = ik_2R_first_joint(A2, P, elbow=e2)
            if th2 is None: continue
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
    best_cost = 1e18
    for th1, th2, e1, e2 in sols:
        dq1 = wrap_pi(th1 - prev_q[0])
        dq2 = wrap_pi(th2 - prev_q[1])
        cost = abs(dq1) + abs(dq2)
        if cost < best_cost:
            best_cost = cost
            best = np.array([th1, th2], float)
            best_meta = (e1, e2)
    return best, best_meta

def jacobian_num(q, branch=+1, h=1e-6):
    B1, B2, p0 = fk(q, branch=branch)
    if p0 is None:
        return None, None

    J = np.zeros((2,2), float)
    for i in range(2):
        dq = np.zeros(2); dq[i] = h
        _, _, p1 = fk(q + dq, branch=branch)
        _, _, p2 = fk(q - dq, branch=branch)
        if p1 is None or p2 is None:
            return None, None
        J[:, i] = (p1 - p2) / (2*h)
    return J, p0

def dls_qdot(J, vref, lam=2e-3):
    # qdot = J^T (J J^T + lam^2 I)^-1 v
    JJt = J @ J.T
    inv = np.linalg.inv(JJt + (lam**2)*np.eye(2))
    return (J.T @ inv @ vref.reshape(2,1)).reshape(2)

# ===================== UI App =====================
class FiveBarUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("5-Bar Parallel Control UI (FK / IK / Jacobian - Tkinter)")
        self.geometry("1100x650")

        # --- State ---
        self.branch = +1            # FK intersection branch (+1: higher y)
        self.mode = tk.StringVar(value="xy")  # "xy" or "joint"
        self.running = False

        # Kinematic control params
        self.dt = 0.01              # UI update (s)
        self.Kp = 6.0               # task-space gain
        self.lam = 2e-3             # DLS damping
        self.qdot_limit = np.deg2rad(300)  # rad/s

        # Robot joint state (rad)
        self.q = np.array([np.deg2rad(40.0), np.deg2rad(140.0)], float)

        # Workspace target point (meters)
        self.pd = np.array([d/2, 0.08], float)
        self.dragging = False

        # IK init to be consistent with pd
        q0, meta = ik_5bar(self.pd, prev_q=None, prefer=(+1, -1))
        if q0 is not None:
            self.q = q0

        # --- Layout ---
        self._build_layout()
        self._redraw()

        self.after(int(self.dt*1000), self._loop)

    # ---------- UI construction ----------
    def _build_layout(self):
        self.columnconfigure(0, weight=6)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Canvas panel
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Control panel
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.columnconfigure(0, weight=1)

        # Mode selection
        mode_box = ttk.LabelFrame(right, text="Mode")
        mode_box.grid(row=0, column=0, sticky="ew", pady=(0,10))
        ttk.Radiobutton(mode_box, text="XY control (Jacobian)", variable=self.mode, value="xy",
                        command=self._on_mode_change).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(mode_box, text="Joint control (FK)", variable=self.mode, value="joint",
                        command=self._on_mode_change).grid(row=1, column=0, sticky="w", padx=8, pady=4)

        # Buttons
        btn_box = ttk.LabelFrame(right, text="Actions")
        btn_box.grid(row=1, column=0, sticky="ew", pady=(0,10))
        btn_box.columnconfigure(0, weight=1)
        btn_box.columnconfigure(1, weight=1)

        self.btn_run = ttk.Button(btn_box, text="Run", command=self._run)
        self.btn_stop = ttk.Button(btn_box, text="Stop", command=self._stop)
        self.btn_home = ttk.Button(btn_box, text="Home (IK to Pd)", command=self._home_to_pd)
        self.btn_flip = ttk.Button(btn_box, text="Flip branch", command=self._flip_branch)

        self.btn_run.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self.btn_home.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self.btn_flip.grid(row=1, column=1, sticky="ew", padx=6, pady=6)

        # XY target
        xy_box = ttk.LabelFrame(right, text="Target Pd (meters)")
        xy_box.grid(row=2, column=0, sticky="ew", pady=(0,10))
        xy_box.columnconfigure(1, weight=1)

        self.px_var = tk.DoubleVar(value=float(self.pd[0]))
        self.py_var = tk.DoubleVar(value=float(self.pd[1]))
        ttk.Label(xy_box, text="x").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(xy_box, textvariable=self.px_var).grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        ttk.Label(xy_box, text="y").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(xy_box, textvariable=self.py_var).grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        ttk.Button(xy_box, text="Set Pd", command=self._set_pd_from_entries).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        # Joint sliders
        joint_box = ttk.LabelFrame(right, text="Joint angles (deg)")
        joint_box.grid(row=3, column=0, sticky="ew", pady=(0,10))
        joint_box.columnconfigure(0, weight=1)

        self.th1_var = tk.DoubleVar(value=np.rad2deg(self.q[0]))
        self.th2_var = tk.DoubleVar(value=np.rad2deg(self.q[1]))

        self.s_th1 = ttk.Scale(joint_box, from_=-180, to=180, variable=self.th1_var, command=self._on_joint_slider)
        self.s_th2 = ttk.Scale(joint_box, from_=-180, to=180, variable=self.th2_var, command=self._on_joint_slider)
        ttk.Label(joint_box, text="theta1").grid(row=0, column=0, sticky="w", padx=6, pady=(6,2))
        self.s_th1.grid(row=1, column=0, sticky="ew", padx=6, pady=(0,6))
        ttk.Label(joint_box, text="theta2").grid(row=2, column=0, sticky="w", padx=6, pady=(6,2))
        self.s_th2.grid(row=3, column=0, sticky="ew", padx=6, pady=(0,6))

        # Control params
        param_box = ttk.LabelFrame(right, text="Jacobian control params")
        param_box.grid(row=4, column=0, sticky="ew", pady=(0,10))
        param_box.columnconfigure(1, weight=1)

        self.kp_var = tk.DoubleVar(value=self.Kp)
        self.lam_var = tk.DoubleVar(value=self.lam)
        ttk.Label(param_box, text="Kp").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(param_box, textvariable=self.kp_var).grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        ttk.Label(param_box, text="lambda").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(param_box, textvariable=self.lam_var).grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        ttk.Button(param_box, text="Apply", command=self._apply_params).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        # Status
        status_box = ttk.LabelFrame(right, text="Status")
        status_box.grid(row=5, column=0, sticky="nsew")
        right.rowconfigure(5, weight=1)

        self.status = tk.Text(status_box, height=10, wrap="word")
        self.status.pack(fill="both", expand=True, padx=6, pady=6)

        # Canvas interactions for XY mode
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Make canvas redraw on resize
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    # ---------- Coordinate mapping ----------
    def _world_to_canvas(self, p):
        # define view bounds in world (meters)
        # show a box that covers typical workspace
        x_min, x_max = -0.06, d + 0.06
        y_min, y_max = -0.02, 0.18

        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)

        sx = w / (x_max - x_min)
        sy = h / (y_max - y_min)

        x = (p[0] - x_min) * sx
        y = h - (p[1] - y_min) * sy
        return np.array([x, y])

    def _canvas_to_world(self, c):
        x_min, x_max = -0.06, d + 0.06
        y_min, y_max = -0.02, 0.18

        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)

        sx = w / (x_max - x_min)
        sy = h / (y_max - y_min)

        x = c[0] / sx + x_min
        y = (h - c[1]) / sy + y_min
        return np.array([x, y])

    # ---------- Actions ----------
    def _run(self):
        self.running = True

    def _stop(self):
        self.running = False

    def _flip_branch(self):
        self.branch *= -1
        self._log(f"Flip FK branch -> {self.branch}")
        self._redraw()

    def _home_to_pd(self):
        q_new, meta = ik_5bar(self.pd, prev_q=self.q, prefer=(+1, -1))
        if q_new is None:
            self._log("Home failed: Pd not reachable")
            return
        self.q = q_new
        self._sync_joint_vars()
        self._log(f"Home IK ok. elbows={meta}")
        self._redraw()

    def _on_mode_change(self):
        self._log(f"Mode: {self.mode.get()}")
        self._redraw()

    def _apply_params(self):
        try:
            self.Kp = float(self.kp_var.get())
            self.lam = float(self.lam_var.get())
            self._log(f"Apply params: Kp={self.Kp}, lambda={self.lam}")
        except Exception:
            self._log("Apply params failed (invalid number).")

    def _set_pd_from_entries(self):
        self.pd = np.array([float(self.px_var.get()), float(self.py_var.get())], float)
        self._log(f"Set Pd: {self.pd}")
        self._redraw()

    def _on_joint_slider(self, _evt=None):
        if self.mode.get() != "joint":
            return
        th1 = np.deg2rad(self.th1_var.get())
        th2 = np.deg2rad(self.th2_var.get())
        self.q = np.array([wrap_pi(th1), wrap_pi(th2)], float)
        self._redraw()

    # ---------- Mouse control for XY mode ----------
    def _on_mouse_down(self, event):
        if self.mode.get() != "xy":
            return
        self.dragging = True
        self._update_pd_from_mouse(event.x, event.y)

    def _on_mouse_drag(self, event):
        if self.mode.get() != "xy" or not self.dragging:
            return
        self._update_pd_from_mouse(event.x, event.y)

    def _on_mouse_up(self, event):
        if self.mode.get() != "xy":
            return
        self.dragging = False

    def _update_pd_from_mouse(self, cx, cy):
        self.pd = self._canvas_to_world(np.array([cx, cy], float))
        self.px_var.set(float(self.pd[0]))
        self.py_var.set(float(self.pd[1]))
        self._redraw()

    # ---------- Hardware hook (replace this) ----------
    def send_to_hw(self, theta1, theta2):
        """
        Replace this with real command sending:
        - UART: serial.write(...)
        - WiFi/WebSocket: send json
        - etc.
        theta1, theta2 in radians.
        """
        # For now: do nothing
        pass

    # ---------- Main loop ----------
    def _loop(self):
        if self.running and self.mode.get() == "xy":
            # Resolved-rate control step: qdot = J^# (vd + Kp*e)
            J, p = jacobian_num(self.q, branch=self.branch)
            if J is not None and p is not None:
                e = (self.pd - p)
                vref = (self.Kp * e)  # since Pd is static in UI (vd = 0)
                qdot = dls_qdot(J, vref, lam=self.lam)

                # clamp qdot
                qdot = np.clip(qdot, -self.qdot_limit, self.qdot_limit)

                # integrate
                self.q = self.q + qdot * self.dt
                self.q[0] = wrap_pi(self.q[0])
                self.q[1] = wrap_pi(self.q[1])

                # send to HW as setpoint
                self.send_to_hw(self.q[0], self.q[1])

                # sync sliders display
                self._sync_joint_vars()

        self._redraw()
        self.after(int(self.dt*1000), self._loop)

    # ---------- Drawing ----------
    def _redraw(self):
        self.canvas.delete("all")

        # Draw axes / bounds (optional)
        self._draw_grid()

        # Compute points
        B1, B2, P = fk(self.q, branch=self.branch)

        # Draw base points
        self._draw_point(A1, r=5, label="A1")
        self._draw_point(A2, r=5, label="A2")

        # Draw desired target
        self._draw_point(self.pd, r=6, label="Pd", fill="#ff0000")
        self._draw_circle(self.pd, 0.002, outline="#ff0000")  # tiny ring

        if P is None or B1 is None or B2 is None:
            self._log_once("FK invalid (out of closure).")
            self._draw_status(extra="FK invalid")
            return

        # Draw links
        self._draw_link(A1, B1, width=3)
        self._draw_link(A2, B2, width=3)
        self._draw_link(B1, P, width=3)
        self._draw_link(B2, P, width=3)

        # Draw joints
        self._draw_point(B1, r=4, label="B1")
        self._draw_point(B2, r=4, label="B2")
        self._draw_point(P,  r=5, label="P", fill="#0066ff")

        # Status and numeric readout
        self._draw_status(P=P)

    def _draw_grid(self):
        # light grid for reference
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        for i in range(0, w, 80):
            self.canvas.create_line(i, 0, i, h, fill="#f0f0f0")
        for j in range(0, h, 80):
            self.canvas.create_line(0, j, w, j, fill="#f0f0f0")

    def _draw_point(self, p, r=4, label=None, fill="#000000"):
        c = self._world_to_canvas(np.array(p, float))
        x, y = c
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill, outline="")
        if label:
            self.canvas.create_text(x+12, y-12, text=label, fill="#333333", anchor="nw")

    def _draw_link(self, p1, p2, width=2):
        c1 = self._world_to_canvas(np.array(p1, float))
        c2 = self._world_to_canvas(np.array(p2, float))
        self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], width=width, fill="#222222")

    def _draw_circle(self, center, radius, outline="#999999"):
        c = self._world_to_canvas(np.array(center, float))
        # approximate radius in canvas pixels using local scale in x
        c2 = self._world_to_canvas(np.array([center[0] + radius, center[1]], float))
        rp = abs(c2[0] - c[0])
        self.canvas.create_oval(c[0]-rp, c[1]-rp, c[0]+rp, c[1]+rp, outline=outline)

    def _draw_status(self, P=None, extra=""):
        # Compute Jacobian info
        J, p_now = jacobian_num(self.q, branch=self.branch)
        detJ = None
        cond = None
        if J is not None:
            detJ = float(np.linalg.det(J))
            s = np.linalg.svd(J, compute_uv=False)
            if s[-1] > 1e-12:
                cond = float(s[0] / s[-1])

        th1_deg = np.rad2deg(self.q[0])
        th2_deg = np.rad2deg(self.q[1])

        if P is None:
            B1, B2, P = fk(self.q, branch=self.branch)

        if P is not None:
            e = self.pd - P
            err = float(np.linalg.norm(e))
        else:
            err = float("nan")

        lines = []
        lines.append(f"mode={self.mode.get()}   running={self.running}   branch={self.branch}")
        lines.append(f"theta1={th1_deg: .2f} deg   theta2={th2_deg: .2f} deg")
        if P is not None:
            lines.append(f"P = [{P[0]: .4f}, {P[1]: .4f}] m")
        lines.append(f"Pd= [{self.pd[0]: .4f}, {self.pd[1]: .4f}] m   |e|={err: .5f} m")
        if detJ is not None and cond is not None:
            lines.append(f"det(J)={detJ: .4e}   cond(J)={cond: .3f}   Kp={self.Kp}   lam={self.lam}")
        else:
            lines.append(f"J invalid   Kp={self.Kp}   lam={self.lam}")
        if extra:
            lines.append(extra)

        # write to status box (replace content)
        self.status.delete("1.0", "end")
        self.status.insert("end", "\n".join(lines))

    def _sync_joint_vars(self):
        # update sliders without causing unwanted jumps
        self.th1_var.set(float(np.rad2deg(self.q[0])))
        self.th2_var.set(float(np.rad2deg(self.q[1])))

    def _log(self, msg):
        # simple log into status footer (keep it short)
        # here we just print to console for debugging
        print(msg)

    def _log_once(self, msg):
        # minimal: print once per redraw is noisy; keep silent or print occasionally
        pass

if __name__ == "__main__":
    app = FiveBarUI()
    app.mainloop()
