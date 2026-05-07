from __future__ import annotations

import math
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fivebar_solver import compute_link_transforms, compute_supported_link_transforms, load_stl_bounds, make_step, make_tree_step, solve_passive_pair, solver_status

matplotlib.use("TkAgg")

Vec3 = np.ndarray

JOINT_ANGLE_NAMES = (
    "pad_joint_right",
    "thigh_joint_right_1",
    "calf_joint_right_1",
    "thigh_joint_right_2",
    "calf_joint_right_2",
    "wheel_joint_right",
    "pad_joint_left",
    "thigh_joint_left_1",
    "calf_joint_left_1",
    "thigh_joint_left_2",
    "calf_joint_left_2",
    "wheel_joint_left",
)
JOINT_ANGLE_INDEX = {name: index for index, name in enumerate(JOINT_ANGLE_NAMES)}

@dataclass
class RobotState:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    left_wheel: float = 0.0
    right_wheel: float = 0.0

@dataclass
class MeshVisual:
    path: Path
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    color: tuple[float, float, float, float]
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray

@dataclass
class URDFLink:
    name: str
    visuals: list[MeshVisual]

@dataclass
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None

@dataclass
class URDFModel:
    name: str
    root_link: str
    links: dict[str, URDFLink]
    joints: list[URDFJoint]
    child_joints: dict[str, list[URDFJoint]]
    mesh_count: int
    triangle_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def transform(points: np.ndarray, origin: Vec3, rotation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + origin

def rpy_matrix(rpy: Vec3) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def rotation_between_vectors(source: Vec3, target: Vec3) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-12 or target_norm < 1e-12:
        return np.eye(3)
    a = source / source_norm
    b = target / target_norm
    axis = np.cross(a, b)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if axis_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis_norm = float(np.linalg.norm(axis))
    axis = axis / axis_norm
    return axis_angle_matrix(axis, math.atan2(axis_norm, dot))

def axis_angle_matrix(axis: Vec3, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / norm
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )

def parse_vec(text: str | None, fallback: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(fallback, dtype=float)
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return np.asarray(fallback, dtype=float)
    return np.asarray([float(part) for part in parts], dtype=float)

def box_faces(center: Vec3, size: tuple[float, float, float], rotation: np.ndarray | None = None) -> list[np.ndarray]:
    sx, sy, sz = size
    x, y, z = sx / 2, sy / 2, sz / 2
    pts = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=float,
    )
    if rotation is None:
        rotation = np.eye(3)
    pts = transform(pts, np.asarray(center, dtype=float), rotation)
    return [
        pts[[0, 1, 2, 3]],
        pts[[4, 5, 6, 7]],
        pts[[0, 1, 5, 4]],
        pts[[1, 2, 6, 5]],
        pts[[2, 3, 7, 6]],
        pts[[3, 0, 4, 7]],
    ]

def box_faces_from_bounds(bounds_min: Vec3, bounds_max: Vec3, origin: Vec3, rotation: np.ndarray, scale: float) -> list[np.ndarray]:
    center = (np.asarray(bounds_min, dtype=float) + np.asarray(bounds_max, dtype=float)) * 0.5
    size = np.maximum((np.asarray(bounds_max, dtype=float) - np.asarray(bounds_min, dtype=float)) * scale, 1e-4)
    world_center = origin + (center * scale) @ rotation.T
    return box_faces(world_center, (float(size[0]), float(size[1]), float(size[2])), rotation)

def cylinder_mesh(
    center: Vec3,
    radius: float,
    length: float,
    axis: str = "y",
    rotation: np.ndarray | None = None,
    segments: int = 28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * math.pi, segments)
    h = np.linspace(-length / 2, length / 2, 2)
    theta, h = np.meshgrid(theta, h)
    if axis == "x":
        pts = np.stack([h, radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
    elif axis == "y":
        pts = np.stack([radius * np.cos(theta), h, radius * np.sin(theta)], axis=-1)
    else:
        pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), h], axis=-1)
    if rotation is None:
        rotation = np.eye(3)
    pts = transform(pts.reshape(-1, 3), np.asarray(center, dtype=float), rotation).reshape(pts.shape)
    return pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

def load_urdf_model(path: Path) -> URDFModel:
    tree = ET.parse(path)
    root = tree.getroot()
    package_dir = path.parents[1]
    links: dict[str, URDFLink] = {}
    joints: list[URDFJoint] = []
    all_min = np.array([math.inf, math.inf, math.inf])
    all_max = np.array([-math.inf, -math.inf, -math.inf])
    triangle_count = 0
    mesh_count = 0

    for link_el in root.findall("link"):
        link_name = link_el.attrib.get("name", "")
        visuals: list[MeshVisual] = []
        for visual_el in link_el.findall("visual"):
            mesh_el = visual_el.find("./geometry/mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename", "")
            if not filename.lower().endswith(".stl"):
                continue
            mesh_path = (path.parent / filename).resolve()
            if not mesh_path.exists():
                mesh_path = (package_dir / filename.replace("package://robot_urdf/", "")).resolve()
            if not mesh_path.exists():
                continue
            origin_el = visual_el.find("origin")
            origin_xyz = parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None)
            origin_rpy = parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None)
            color_el = visual_el.find("./material/color")
            rgba = (0.70, 0.74, 0.82)
            alpha = 1.0
            if color_el is not None and color_el.attrib.get("rgba"):
                rgba_parts = [float(part) for part in color_el.attrib["rgba"].split()]
                if len(rgba_parts) >= 3:
                    rgba = tuple(rgba_parts[:3])
                alpha = rgba_parts[3] if len(rgba_parts) >= 4 else 1.0
            bounds_min, bounds_max, source_triangles = load_stl_bounds(mesh_path)
            visuals.append(MeshVisual(mesh_path, bounds_min, bounds_max, tuple(rgba) + (alpha,), origin_xyz, origin_rpy))
            all_min = np.minimum(all_min, bounds_min)
            all_max = np.maximum(all_max, bounds_max)
            triangle_count += source_triangles
            mesh_count += 1
        links[link_name] = URDFLink(link_name, visuals)

    seen_children: set[str] = set()
    for joint_el in root.findall("joint"):
        name = joint_el.attrib.get("name", "")
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        child = child_el.attrib.get("link", "")
        if child in seen_children:
            continue
        seen_children.add(child)
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        limit_el = joint_el.find("limit")
        lower = float(limit_el.attrib["lower"]) if limit_el is not None and "lower" in limit_el.attrib else None
        upper = float(limit_el.attrib["upper"]) if limit_el is not None and "upper" in limit_el.attrib else None
        joints.append(
            URDFJoint(
                name=name,
                joint_type=joint_el.attrib.get("type", "fixed"),
                parent=parent_el.attrib.get("link", ""),
                child=child,
                origin_xyz=parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None),
                origin_rpy=parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None),
                axis=parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
                lower=lower,
                upper=upper,
            )
        )

    child_joints: dict[str, list[URDFJoint]] = {}
    child_links = {joint.child for joint in joints}
    for joint in joints:
        child_joints.setdefault(joint.parent, []).append(joint)
    root_link = next((name for name in links if name not in child_links), next(iter(links), "base_link"))
    if not np.isfinite(all_min).all():
        all_min = np.zeros(3)
        all_max = np.ones(3)
    return URDFModel(root.attrib.get("name", path.stem), root_link, links, joints, child_joints, mesh_count, triangle_count, all_min, all_max)

class RobotViewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bao 2-Legged Robot 3D View")
        self.root.geometry("1360x840")
        self.root.minsize(1120, 720)

        self.state = RobotState()
        self.last_text_update = 0.0
        self.redraw_after_id: str | None = None
        self.camera_initialized = False
        self.camera_view: tuple[float, float, float] | None = None
        self.solve_dirty_sides: set[str] = {"left", "right"}
        self.urdf: URDFModel | None = None
        self.urdf_error = ""
        self.solver_chains: dict[str, np.ndarray] = {}
        self.link_order: list[str] = []
        self.link_index: dict[str, int] = {}
        self.tree_steps = np.empty((0, 13), dtype=np.float64)
        self.last_closure_error: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.end_effector_positions: dict[str, np.ndarray] = {}
        self.backend_warning = ""
        self.balance_tilt_deg = 0.0
        self.balance_roll_deg = 0.0
        self.balance_pitch_deg = 0.0
        self.camera_drag_start: tuple[float, float, float, float, float] | None = None
        self.selected_item = ""
        self.selectable_points: dict[str, np.ndarray] = {}
        
        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._load_urdf_model()
        self._build_ui()
        self._apply_style()
        self.draw_scene(reset_camera=True)

    def _build_vars(self) -> None:
        defaults = {
            "view_span": 4.0,
            "render_style": "Linkage",
            "urdf_scale": 4.0,
            "show_joints": True,
            "left_thigh_a_deg": 0.0,
            "left_thigh_b_deg": 0.0,
            "left_calf_a_deg": 0.0,
            "left_calf_b_deg": 0.0,
            "right_thigh_a_deg": 0.0,
            "right_thigh_b_deg": 0.0,
            "right_calf_a_deg": 0.0,
            "right_calf_b_deg": 0.0,
            "mesh_edges": True,
            "camera": "ISO",

        }
        for key, value in defaults.items():
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, str):
                self.vars[key] = tk.StringVar(value=value)
            else:
                self.vars[key] = tk.DoubleVar(value=value)

    def _load_urdf_model(self) -> None:
        base_dir = Path(__file__).resolve().parent
        urdf_path = base_dir / "robot_urdf" / "urdf" / "robot.SLDASM.urdf"
        if not urdf_path.exists():
            self.urdf_error = "robot_urdf/urdf/robot.SLDASM.urdf not found"
            return
        try:
            self.urdf = load_urdf_model(urdf_path)
            self.solver_chains.clear()
            self._prepare_c_kinematics()
            self.urdf_error = ""
        except Exception as exc:
            self.urdf = None
            self.urdf_error = f"URDF load failed: {exc}"

    def _prepare_c_kinematics(self) -> None:
        if self.urdf is None:
            self.link_order = []
            self.link_index = {}
            self.tree_steps = np.empty((0, 13), dtype=np.float64)
            return

        order = [self.urdf.root_link]
        ordered_joints: list[URDFJoint] = []
        seen = {self.urdf.root_link}
        stack = [self.urdf.root_link]
        while stack:
            parent = stack.pop(0)
            for joint in self.urdf.child_joints.get(parent, []):
                ordered_joints.append(joint)
                if joint.child not in seen:
                    order.append(joint.child)
                    seen.add(joint.child)
                    stack.append(joint.child)
        for name in self.urdf.links:
            if name not in seen:
                order.append(name)
                seen.add(name)

        self.link_order = order
        self.link_index = {name: index for index, name in enumerate(order)}
        steps = []
        for joint in ordered_joints:
            parent_index = self.link_index[joint.parent]
            child_index = self.link_index[joint.child]
            movable = joint.joint_type in {"revolute", "continuous"}
            steps.append(
                make_tree_step(
                    parent_index,
                    child_index,
                    joint.origin_xyz,
                    joint.origin_rpy,
                    joint.axis,
                    JOINT_ANGLE_INDEX.get(joint.name, -1),
                    movable,
                )
            )
        self.tree_steps = np.vstack(steps) if steps else np.empty((0, 13), dtype=np.float64)

    def _apply_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#eef1f5")
        style.configure("Panel.TFrame", background="#f7f8fb", relief="solid", borderwidth=1)
        style.configure("Title.TLabel", background="#f7f8fb", foreground="#1f2937", font=("Segoe UI Semibold", 11))
        style.configure("TLabel", background="#f7f8fb", foreground="#273244")
        style.configure("TButton", padding=(10, 6))
        style.configure("Run.TButton", font=("Segoe UI Semibold", 10))
        style.configure("Status.TLabel", background="#26313f", foreground="#f6f8fb", padding=(10, 6))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        view_frame = ttk.Frame(self.root)
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.rowconfigure(0, weight=1)
        view_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(9, 7), dpi=100, facecolor="#eef1f5")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_proj_type("ortho")
        self.canvas = FigureCanvasTkAgg(self.fig, master=view_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, view_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")
        self.toolbar = toolbar
        self._configure_solidworks_navigation()

        side_container = ttk.Frame(self.root, style="Panel.TFrame", padding=0)
        side_container.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)
        side_container.rowconfigure(0, weight=1)
        side_container.columnconfigure(0, weight=1)

        side_canvas = tk.Canvas(side_container, width=410, highlightthickness=0, bg="#f7f8fb")
        side_scrollbar = ttk.Scrollbar(side_container, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_canvas.grid(row=0, column=0, sticky="ns")
        side_scrollbar.grid(row=0, column=1, sticky="ns")

        side = ttk.Frame(side_canvas, style="Panel.TFrame", padding=12)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.columnconfigure(0, weight=1)

        def update_scroll_region(_event: tk.Event) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))

        def update_side_width(event: tk.Event) -> None:
            side_canvas.itemconfigure(side_window, width=event.width)

        def scroll_side(event: tk.Event) -> str:
            side_canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        side.bind("<Configure>", update_scroll_region)
        side_canvas.bind("<Configure>", update_side_width)
        side_canvas.bind("<MouseWheel>", scroll_side)
        side.bind("<MouseWheel>", scroll_side)
        self._bind_mousewheel_recursive(side, scroll_side)

        row = 0
        ttk.Label(side, text="Robot 3D Control", style="Title.TLabel").grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        buttons = ttk.Frame(side, style="Panel.TFrame")
        buttons.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Reset angles", command=self.reset).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        row += 1

        row = self._section(side, row, "Left Active Joints")
        row = self._slider(side, row, "A1 active deg", "left_thigh_a_deg", -90, 90, 0.1)
        row = self._slider(side, row, "A2 active deg", "left_thigh_b_deg", -90, 90, 0.1)

        row = self._section(side, row, "Right Active Joints")
        row = self._slider(side, row, "A1 active deg", "right_thigh_a_deg", -90, 90, 0.1)
        row = self._slider(side, row, "A2 active deg", "right_thigh_b_deg", -90, 90, 0.1)

        row = self._section(side, row, "3D Display")
        render_style = ttk.Combobox(
            side,
            textvariable=self.vars["render_style"],
            values=("Linkage", "Solid"),
            state="readonly",
        )
        render_style.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        render_style.bind("<<ComboboxSelected>>", lambda _event: self.queue_draw())
        row += 1
        row = self._slider(side, row, "URDF scale", "urdf_scale", 1.0, 8.0, 0.1)
        ttk.Checkbutton(side, text="Show URDF joint pivots", variable=self.vars["show_joints"], command=self.queue_draw).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        ttk.Checkbutton(side, text="Draw mesh edges", variable=self.vars["mesh_edges"], command=self.queue_draw).grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        row = self._section(side, row, "Camera")
        camera = ttk.Combobox(side, textvariable=self.vars["camera"], values=("ISO", "Front", "Side", "Top"), state="readonly")
        camera.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        camera.bind("<<ComboboxSelected>>", lambda _event: self.draw_scene(reset_camera=True))
        row += 1
        ttk.Button(side, text="Snapshot redraw", command=lambda: self.draw_scene(reset_camera=False)).grid(row=row, column=0, sticky="ew", pady=4)
        row += 1

        self.readout = tk.Text(side, height=10, width=36, relief="solid", bd=1, bg="#ffffff", fg="#1f2937")
        self.readout.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        self.readout.configure(state="disabled")

        self.status = ttk.Label(self.root, text="", style="Status.TLabel")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._bind_mousewheel_recursive(side, scroll_side)

    def _section(self, parent: ttk.Frame, row: int, text: str) -> int:
        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=(10, 7))
        row += 1
        ttk.Label(parent, text=text, style="Title.TLabel").grid(row=row, column=0, sticky="ew", pady=(0, 4))
        return row + 1

    def _bind_mousewheel_recursive(self, widget: tk.Widget, callback) -> None:
        widget.bind("<MouseWheel>", callback)
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child, callback)

    def _configure_solidworks_navigation(self) -> None:
        try:
            self.ax.disable_mouse_rotation()
        except AttributeError:
            self.ax.mouse_init(rotate_btn=[], pan_btn=[], zoom_btn=[])
        self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _on_canvas_scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        step = getattr(event, "step", 0.0)
        if step == 0:
            return
        current_span = max(float(self.vars["view_span"].get()), 0.35)
        requested_factor = 0.90 ** abs(step) if step > 0 else (1.0 / 0.90) ** abs(step)
        new_span = float(np.clip(current_span * requested_factor, 0.35, 30.0))
        actual_factor = new_span / current_span

        current_center = self._current_camera_center()
        anchor = self._cursor_anchor_point(event, current_center)
        new_center = anchor + (current_center - anchor) * actual_factor
        self.vars["view_span"].set(new_span)
        self._apply_camera_limits(
            center_x=float(new_center[0]),
            center_y=float(new_center[1]),
            center_z=float(new_center[2]),
        )
        self.canvas.draw_idle()

    def _on_canvas_button_press(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        if getattr(event, "button", None) == 2:
            self._remember_camera()
            elev, azim, roll = self.camera_view or (24.0, -45.0, 0.0)
            self.camera_drag_start = (float(event.x), float(event.y), elev, azim, roll)
            return
        if getattr(event, "button", None) in (1, 3):
            self._select_nearest_item(event)

    def _on_canvas_button_release(self, event) -> None:
        if getattr(event, "button", None) == 2:
            self.camera_drag_start = None

    def _on_canvas_motion(self, event) -> None:
        if self.camera_drag_start is None or event.inaxes is not self.ax:
            return
        start_x, start_y, start_elev, start_azim, start_roll = self.camera_drag_start
        dx = float(event.x) - start_x
        dy = float(event.y) - start_y
        elev = float(np.clip(start_elev - dy * 0.35, -89.0, 89.0))
        azim = start_azim - dx * 0.35
        self.camera_view = (elev, azim, start_roll)
        try:
            self.ax.view_init(elev=elev, azim=azim, roll=start_roll)
        except TypeError:
            self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _apply_camera_limits_from_current_center(self) -> None:
        center = self._current_camera_center()
        self._apply_camera_limits(center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]))

    def _current_camera_center(self) -> np.ndarray:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        return np.array(
            [
                (float(xlim[0]) + float(xlim[1])) * 0.5,
                (float(ylim[0]) + float(ylim[1])) * 0.5,
                (float(zlim[0]) + float(zlim[1])) * 0.5,
            ],
            dtype=float,
        )

    def _cursor_anchor_point(self, event, fallback: np.ndarray) -> np.ndarray:
        try:
            proj_x, proj_y = self.ax.transData.inverted().transform((event.x, event.y))
            inv_projection = np.linalg.inv(self.ax.get_proj())
            near = np.asarray(proj3d.inv_transform(proj_x, proj_y, -1.0, inv_projection), dtype=float).reshape(3)
            far = np.asarray(proj3d.inv_transform(proj_x, proj_y, 1.0, inv_projection), dtype=float).reshape(3)
        except (ValueError, np.linalg.LinAlgError):
            return fallback

        ray = far - near
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1e-9 or not np.isfinite(ray_norm):
            return fallback
        view_normal = ray / ray_norm
        denom = float(np.dot(ray, view_normal))
        if abs(denom) < 1e-9:
            return fallback
        t = float(np.dot(fallback - near, view_normal) / denom)
        anchor = near + ray * t
        if not np.isfinite(anchor).all():
            return fallback
        return anchor

    def _select_nearest_item(self, event) -> None:
        if not self.selectable_points:
            return
        projected: list[tuple[float, str]] = []
        for name, point in self.selectable_points.items():
            x2, y2, _z2 = proj3d.proj_transform(point[0], point[1], point[2], self.ax.get_proj())
            screen_x, screen_y = self.ax.transData.transform((x2, y2))
            distance = math.hypot(float(event.x) - screen_x, float(event.y) - screen_y)
            projected.append((distance, name))
        distance, name = min(projected, default=(math.inf, ""))
        if distance > 42.0:
            return
        self.selected_item = name
        self._update_text()
        self.draw_scene(reset_camera=False)

    def _slider(self, parent: ttk.Frame, row: int, label: str, key: str, start: float, end: float, step: float) -> int:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        value = ttk.Label(frame, textvariable=self.vars[key], width=7, anchor="e")
        value.grid(row=0, column=1, columnspan=2, sticky="e")
        scale = ttk.Scale(frame, from_=start, to=end, variable=self.vars[key], command=lambda _v: self.queue_draw(changed_key=key))
        scale.grid(row=1, column=0, columnspan=3, sticky="ew")
        spin = ttk.Spinbox(frame, from_=start, to=end, increment=step, textvariable=self.vars[key], width=8, command=lambda: self.queue_draw(changed_key=key))
        spin.grid(row=2, column=1, sticky="e", pady=(2, 0))
        spin.bind("<Return>", lambda _event: self.queue_draw(changed_key=key))
        spin.bind("<KP_Enter>", lambda _event: self.queue_draw(changed_key=key))
        ttk.Button(frame, text="Apply", command=lambda: self.queue_draw(changed_key=key)).grid(row=2, column=2, sticky="e", padx=(6, 0), pady=(2, 0))
        return row + 1

    def reset(self) -> None:
        self.state = RobotState()
        for key in (
            "left_thigh_a_deg",
            "left_thigh_b_deg",
            "left_calf_a_deg",
            "left_calf_b_deg",
            "right_thigh_a_deg",
            "right_thigh_b_deg",
            "right_calf_a_deg",
            "right_calf_b_deg",
        ):
            self.vars[key].set(0.0)
        self.solve_dirty_sides = {"left", "right"}
        self.draw_scene(reset_camera=True)

    def body_rotation(self) -> np.ndarray:
        return rot_z(self.state.yaw)

    def queue_draw(self, delay_ms: int = 1, changed_key: str | None = None, force_solve: bool = False) -> None:
        self._mark_solver_dirty(changed_key, force_solve)
        if self.redraw_after_id is not None:
            self.root.after_cancel(self.redraw_after_id)
        self.redraw_after_id = self.root.after(delay_ms, self._run_queued_draw)

    def _mark_solver_dirty(self, changed_key: str | None, force_solve: bool = False) -> None:
        if force_solve:
            self.solve_dirty_sides = {"left", "right"}
            return
        if changed_key is None:
            return
        if changed_key in {"left_thigh_a_deg", "left_thigh_b_deg"}:
            self.solve_dirty_sides.add("left")
        elif changed_key in {"right_thigh_a_deg", "right_thigh_b_deg"}:
            self.solve_dirty_sides.add("right")

    def _run_queued_draw(self) -> None:
        self.redraw_after_id = None
        self.draw_scene(reset_camera=False)

    def draw_scene(self, reset_camera: bool = False) -> None:
        if self.redraw_after_id is not None:
            self.root.after_cancel(self.redraw_after_id)
            self.redraw_after_id = None
        self._remember_camera()
        self.ax.clear()
        self.ax.set_facecolor("#eef1f5")
        self.fig.set_facecolor("#eef1f5")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self._draw_robot()
        self._set_camera(reset_camera=reset_camera)
        self._update_text()
        self.canvas.draw_idle()

    def _draw_robot(self) -> None:
        if self.urdf is not None:
            try:
                self._draw_urdf_robot()
            except RuntimeError as exc:
                self.backend_warning = str(exc)
                self.ax.text(0.0, 0.0, 0.0, str(exc), color="#b91c1c")
            return
        self.ax.text(0.0, 0.0, 0.0, self.urdf_error or "URDF model is not loaded", color="#b91c1c")

    def _draw_urdf_robot(self) -> None:
        if self.urdf is None:
            return
        body_R = self.body_rotation()
        scale = float(self.vars["urdf_scale"].get())
        body_origin = np.array([self.state.x, self.state.y, 0.0])
        root_R = body_R @ rot_z(math.radians(90.0))
        joint_angles = self._urdf_joint_angles()
        transforms = self._urdf_supported_link_transforms(body_origin, root_R, scale, joint_angles)
        self._cache_selectable_points(transforms)
        self._update_end_effector_positions(transforms)

        if str(self.vars["render_style"].get()) == "Linkage":
            self._draw_linkage_diagram(transforms, scale)
            return

        for link_name, (origin, rotation) in transforms.items():
            link = self.urdf.links.get(link_name)
            if link is None:
                continue
            for visual in link.visuals:
                visual_R = rotation @ rpy_matrix(visual.origin_rpy)
                visual_origin = origin + (visual.origin_xyz * scale) @ rotation.T
                self._draw_link_proxy(link_name, visual, visual_origin, visual_R, scale)

        if bool(self.vars["show_joints"].get()):
            points = np.asarray([value[0] for name, value in transforms.items() if name != self.urdf.root_link])
            if len(points):
                self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=18, color="#1d4ed8", edgecolors="#ffffff", linewidths=0.45, depthshade=False)
        self._draw_selected_marker(scale)

    def _cache_selectable_points(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.selectable_points = {
            name: origin.copy()
            for name, (origin, _rotation) in transforms.items()
            if self.urdf is None or name != self.urdf.root_link
        }
        if self.selected_item and self.selected_item not in self.selectable_points:
            self.selected_item = ""

    def _draw_selected_marker(self, scale: float) -> None:
        if not self.selected_item:
            return
        point = self.selectable_points.get(self.selected_item)
        if point is None:
            return
        size = max(90.0, 18.0 * scale)
        self.ax.scatter(
            [point[0]],
            [point[1]],
            [point[2]],
            s=size,
            color="#f59e0b",
            edgecolors="#111827",
            linewidths=1.1,
            depthshade=False,
        )
        self.ax.text(point[0], point[1], point[2] + 0.045 * scale, self.selected_item, color="#92400e", fontsize=8, ha="center")

    def _update_end_effector_positions(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.end_effector_positions = {
            side: transforms[link_name][0].copy()
            for side, link_name in (("left", "wheel_link_left"), ("right", "wheel_link_right"))
            if link_name in transforms
        }

    def _support_aligned_root_rotation(
        self,
        root_rotation: np.ndarray,
        scale: float,
        joint_angles: dict[str, float],
    ) -> np.ndarray:
        transforms = self._urdf_link_transforms(np.zeros(3), root_rotation, scale, joint_angles)
        if "wheel_link_left" not in transforms or "wheel_link_right" not in transforms:
            return root_rotation
        wheel_line = transforms["wheel_link_left"][0] - transforms["wheel_link_right"][0]
        horizontal_line = wheel_line.copy()
        horizontal_line[2] = 0.0
        if np.linalg.norm(horizontal_line) < 1e-9:
            return root_rotation
        correction = rotation_between_vectors(wheel_line, horizontal_line)
        return correction @ root_rotation

    def _urdf_supported_link_transforms(
        self,
        support_origin: Vec3,
        requested_root_rotation: np.ndarray,
        scale: float,
        joint_angles: dict[str, float],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.urdf is None or not self.link_order:
            return {}
        if "wheel_link_left" not in self.link_index or "wheel_link_right" not in self.link_index:
            root_R = self._support_aligned_root_rotation(requested_root_rotation, scale, joint_angles)
            root_origin = self._support_anchored_root(support_origin, root_R, scale, joint_angles)
            return self._urdf_link_transforms(root_origin, root_R, scale, joint_angles)

        angles = np.asarray([joint_angles[name] for name in JOINT_ANGLE_NAMES], dtype=np.float64)
        wheel_radius = max(
            self._link_radius("wheel_link_left", scale, fallback=0.045 * scale),
            self._link_radius("wheel_link_right", scale, fallback=0.045 * scale),
        )
        try:
            origins, rotations = compute_supported_link_transforms(
                self.tree_steps,
                angles,
                support_origin,
                requested_root_rotation,
                scale,
                len(self.link_order),
                self.link_index["wheel_link_left"],
                self.link_index["wheel_link_right"],
                wheel_radius,
            )
            self.backend_warning = ""
        except RuntimeError as exc:
            self.backend_warning = f"{exc} | display fallback active"
            root_R = self._support_aligned_root_rotation(requested_root_rotation, scale, joint_angles)
            root_origin = self._support_anchored_root(support_origin, root_R, scale, joint_angles)
            return self._urdf_link_transforms_python(root_origin, root_R, scale, joint_angles)

        return {
            name: (origins[index], rotations[index])
            for index, name in enumerate(self.link_order)
        }

    def _support_anchored_root(
        self,
        support_origin: Vec3,
        root_rotation: np.ndarray,
        scale: float,
        joint_angles: dict[str, float],
    ) -> np.ndarray:
        raw_root = np.zeros(3, dtype=float)
        transforms = self._urdf_link_transforms(raw_root, root_rotation, scale, joint_angles)
        wheel_names = [name for name in ("wheel_link_left", "wheel_link_right") if name in transforms]
        if not wheel_names:
            bounds_min, bounds_max = self._urdf_world_bounds(transforms, scale)
            root = raw_root.copy()
            root[:2] += support_origin[:2] - ((bounds_min[:2] + bounds_max[:2]) * 0.5)
            root[2] += 0.015 - bounds_min[2]
            return root

        wheel_centers = np.asarray([transforms[name][0] for name in wheel_names])
        wheel_radius = max(self._link_radius(name, scale, fallback=0.045 * scale) for name in wheel_names)
        support_center = wheel_centers.mean(axis=0)
        root = raw_root.copy()
        root[:2] += support_origin[:2] - support_center[:2]
        root[2] += wheel_radius - support_center[2]
        return root

    def _draw_linkage_diagram(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], scale: float) -> None:
        self.ax.set_facecolor("#ffffff")
        self.fig.set_facecolor("#ffffff")
        self._draw_linkage_reference_grid(transforms)
        self._draw_balance_plane(transforms, scale)

        for side_name, side_color, target_color in (
            ("right", "#222222", "#0066ff"),
            ("left", "#5b6472", "#0ea5e9"),
        ):
            self._draw_fivebar_side(transforms, side_name, side_color, target_color, scale)

        if bool(self.vars["show_joints"].get()):
            all_points = np.asarray([value[0] for name, value in transforms.items() if name != self.urdf.root_link])
            if len(all_points):
                self.ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], s=8, color="#9ca3af", depthshade=False, alpha=0.55)
        self._draw_selected_marker(scale)

    def _draw_balance_plane(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], scale: float) -> None:
        anchors = [
            transforms[name][0]
            for name in ("thigh_right_1", "thigh_right_2", "thigh_left_1", "thigh_left_2")
            if name in transforms
        ]
        if len(anchors) < 3:
            return
        anchors = [np.asarray(point, dtype=float) for point in anchors]
        right_center = (transforms["thigh_right_1"][0] + transforms["thigh_right_2"][0]) * 0.5
        left_center = (transforms["thigh_left_1"][0] + transforms["thigh_left_2"][0]) * 0.5
        x_axis = (
            (transforms["thigh_right_2"][0] - transforms["thigh_right_1"][0])
            + (transforms["thigh_left_2"][0] - transforms["thigh_left_1"][0])
        ) * 0.5
        y_axis = left_center - right_center
        x_norm = float(np.linalg.norm(x_axis))
        y_norm = float(np.linalg.norm(y_axis))
        if x_norm < 1e-9 or y_norm < 1e-9:
            return
        x_axis = x_axis / x_norm
        y_axis = y_axis / y_norm
        normal = np.cross(x_axis, y_axis)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-9:
            return
        normal = normal / normal_norm
        if normal[2] < 0.0:
            normal = -normal
            y_axis = -y_axis
        y_axis = np.cross(normal, x_axis)

        center = np.mean(np.asarray(anchors), axis=0) + normal * (0.018 * scale)
        half_x = max(x_norm * 0.95, 0.14 * scale)
        half_y = max(y_norm * 0.72, 0.10 * scale)
        corners = np.array(
            [
                center - x_axis * half_x - y_axis * half_y,
                center + x_axis * half_x - y_axis * half_y,
                center + x_axis * half_x + y_axis * half_y,
                center - x_axis * half_x + y_axis * half_y,
            ]
        )
        plane = Poly3DCollection([corners], facecolor="#60a5fa", edgecolor="#1d4ed8", linewidth=0.75, alpha=0.24)
        plane.set_zsort("average")
        self.ax.add_collection3d(plane)
        for anchor in anchors:
            projected = center + x_axis * np.dot(anchor - center, x_axis) + y_axis * np.dot(anchor - center, y_axis)
            self._plot_link(anchor, projected, "#93c5fd", linewidth=1.2)

        up = np.array([0.0, 0.0, 1.0])
        self.balance_tilt_deg = math.degrees(math.acos(float(np.clip(np.dot(normal, up), -1.0, 1.0))))
        self.balance_roll_deg = math.degrees(math.atan2(normal[1], normal[2]))
        self.balance_pitch_deg = math.degrees(math.atan2(-normal[0], math.sqrt(normal[1] * normal[1] + normal[2] * normal[2])))
        self.ax.text(
            center[0],
            center[1],
            center[2] + 0.018 * scale,
            f"BALANCE PLANE\npitch={self.balance_pitch_deg:+.1f} roll={self.balance_roll_deg:+.1f}",
            color="#1d4ed8",
            fontsize=8,
            ha="center",
        )

    def _draw_fivebar_side(
        self,
        transforms: dict[str, tuple[np.ndarray, np.ndarray]],
        side: str,
        link_color: str,
        point_color: str,
        scale: float,
    ) -> None:
        required = {
            "A1": f"thigh_{side}_1",
            "A2": f"thigh_{side}_2",
            "B1": f"calf_{side}_link_1",
            "B2": f"calf_{side}_link_2",
            "P": f"wheel_link_{side}",
        }
        if any(link not in transforms for link in required.values()):
            return

        A1 = transforms[required["A1"]][0]
        A2 = transforms[required["A2"]][0]
        B1 = transforms[required["B1"]][0]
        B2 = transforms[required["B2"]][0]
        P = transforms[required["P"]][0]

        active_color = "#111827"
        passive_color = link_color
        base_color = "#d1d5db"
        self._plot_link(A1, A2, base_color, linewidth=1.5)
        self._plot_link(A1, B1, link_color, linewidth=3.0)
        self._plot_link(A2, B2, link_color, linewidth=3.0)
        self._plot_link(B1, P, passive_color, linewidth=3.0)
        self._plot_link(B2, P, passive_color, linewidth=3.0)

        wheel_radius = self._link_radius(required["P"], scale, fallback=0.045 * scale)
        self._draw_cylinder(P, wheel_radius, 0.018 * scale, "y", transforms[required["P"]][1], "#2d3340", edge_color="#151922", segments=32)

        self._plot_point(A1, "A1", fill=active_color, size=46)
        self._plot_point(A2, "A2", fill=active_color, size=46)
        self._plot_point(B1, "B1", fill="#6b7280", size=34)
        self._plot_point(B2, "B2", fill="#6b7280", size=34)
        self._plot_point(P, "P", fill=point_color, size=52)

        label_anchor = (A1 + A2) * 0.5
        self.ax.text(label_anchor[0], label_anchor[1], label_anchor[2] + 0.05 * scale, side.upper(), color=active_color, fontsize=9, ha="center")

    def _plot_link(self, p0: Vec3, p1: Vec3, color: str, linewidth: float = 2.0) -> None:
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=linewidth, solid_capstyle="round")

    def _plot_point(self, p: Vec3, label: str, fill: str = "#000000", size: float = 30.0) -> None:
        self.ax.scatter([p[0]], [p[1]], [p[2]], s=size, color=fill, edgecolors="#ffffff", linewidths=0.7, depthshade=False)
        self.ax.text(p[0], p[1], p[2] + 0.025, label, color="#333333", fontsize=8)

    def _link_radius(self, link_name: str, scale: float, fallback: float) -> float:
        if self.urdf is None:
            return fallback
        link = self.urdf.links.get(link_name)
        if link is None or not link.visuals:
            return fallback
        visual = link.visuals[0]
        extents = np.maximum((visual.bounds_max - visual.bounds_min) * scale, 1e-5)
        return max(float(extents[0]), float(extents[2])) * 0.5

    def _draw_linkage_reference_grid(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        points = np.asarray([origin for origin, _rotation in transforms.values()])
        if len(points) == 0:
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        z = mins[2] - 0.015
        step = max(0.05, float(np.linalg.norm(maxs[:2] - mins[:2])) / 8.0)
        x0, x1 = mins[0] - step, maxs[0] + step
        y0, y1 = mins[1] - step, maxs[1] + step
        xs = np.arange(x0, x1 + step, step)
        ys = np.arange(y0, y1 + step, step)
        for x in xs:
            self.ax.plot([x, x], [y0, y1], [z, z], color="#f0f0f0", linewidth=0.7)
        for y in ys:
            self.ax.plot([x0, x1], [y, y], [z, z], color="#f0f0f0", linewidth=0.7)

    def _draw_link_proxy(self, link_name: str, visual: MeshVisual, origin: Vec3, rotation: np.ndarray, scale: float) -> None:
        base_color, edge_color = self._solidworks_link_style(link_name)
        bounds_min = visual.bounds_min
        bounds_max = visual.bounds_max
        extents = np.maximum((bounds_max - bounds_min) * scale, 1e-4)
        center_local = (bounds_min + bounds_max) * 0.5
        center = origin + (center_local * scale) @ rotation.T

        if "wheel" in link_name:
            axis = "y" if extents[1] <= max(extents[0], extents[2]) else "x"
            radius = max(float(extents[0]), float(extents[2])) * 0.5
            length = max(float(extents[1]), 0.025 * scale)
            self._draw_cylinder(center, radius, length, axis, rotation, base_color, edge_color=edge_color, segments=40)
            self._draw_cylinder(center, radius * 0.42, length * 1.12, axis, rotation, "#d7dee9", edge_color=edge_color, segments=32)
            return

        faces = box_faces_from_bounds(bounds_min, bounds_max, origin, rotation, scale)
        self._add_faces(faces, base_color, alpha=min(0.98, visual.color[3]), edge=edge_color)

    def _solidworks_link_style(self, link_name: str) -> tuple[str, str]:
        if link_name == "base_link":
            return "#d8dde7", "#7b8494"
        if "pad" in link_name:
            return "#b7c0d0", "#6f7887"
        if "hip" in link_name:
            return "#cfd7e6", "#7b8494"
        if "thigh" in link_name:
            return "#8fb6dc", "#386b96"
        if "calf" in link_name:
            return "#9aa7b8", "#596678"
        if "wheel" in link_name:
            return "#2d3340", "#151922"
        return "#c4ccd8", "#6f7887"

    def _urdf_link_transforms(
        self,
        root_origin: Vec3,
        root_rotation: np.ndarray,
        scale: float,
        joint_angles: dict[str, float] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.urdf is None or not self.link_order:
            return {}
        if joint_angles is None:
            joint_angles = self._urdf_joint_angles()
        angles = np.asarray([joint_angles[name] for name in JOINT_ANGLE_NAMES], dtype=np.float64)
        try:
            origins, rotations = compute_link_transforms(
                self.tree_steps,
                angles,
                root_origin,
                root_rotation,
                scale,
                len(self.link_order),
            )
            self.backend_warning = ""
        except RuntimeError as exc:
            self.backend_warning = f"{exc} | display fallback active"
            return self._urdf_link_transforms_python(root_origin, root_rotation, scale, joint_angles)
        return {
            name: (origins[index], rotations[index])
            for index, name in enumerate(self.link_order)
        }

    def _urdf_link_transforms_python(
        self,
        root_origin: Vec3,
        root_rotation: np.ndarray,
        scale: float,
        joint_angles: dict[str, float],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.urdf is None:
            return {}
        transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {self.urdf.root_link: (root_origin, root_rotation)}
        stack = [self.urdf.root_link]
        while stack:
            parent = stack.pop()
            parent_origin, parent_R = transforms[parent]
            for joint in self.urdf.child_joints.get(parent, []):
                joint_origin = parent_origin + (joint.origin_xyz * scale) @ parent_R.T
                joint_R = parent_R @ rpy_matrix(joint.origin_rpy)
                angle = joint_angles.get(joint.name, 0.0)
                if joint.joint_type in {"revolute", "continuous"}:
                    child_R = joint_R @ axis_angle_matrix(joint.axis, angle)
                else:
                    child_R = joint_R
                transforms[joint.child] = (joint_origin, child_R)
                stack.append(joint.child)
        return transforms

    def _urdf_world_bounds(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], scale: float) -> tuple[np.ndarray, np.ndarray]:
        if self.urdf is None:
            return np.zeros(3), np.ones(3)
        bounds_min = np.array([math.inf, math.inf, math.inf])
        bounds_max = np.array([-math.inf, -math.inf, -math.inf])
        for link_name, (origin, rotation) in transforms.items():
            link = self.urdf.links.get(link_name)
            if link is None:
                continue
            for visual in link.visuals:
                visual_R = rotation @ rpy_matrix(visual.origin_rpy)
                visual_origin = origin + (visual.origin_xyz * scale) @ rotation.T
                world_points = self._visual_bounds_points(visual, visual_origin, visual_R, scale)
                bounds_min = np.minimum(bounds_min, world_points.min(axis=0))
                bounds_max = np.maximum(bounds_max, world_points.max(axis=0))
        if not np.isfinite(bounds_min).all():
            return np.zeros(3), np.ones(3)
        return bounds_min, bounds_max

    def _visual_bounds_points(self, visual: MeshVisual, origin: Vec3, rotation: np.ndarray, scale: float) -> np.ndarray:
        lo = visual.bounds_min
        hi = visual.bounds_max
        points = np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]],
            ],
            dtype=float,
        )
        return points * scale @ rotation.T + origin

    def _urdf_joint_angles(self) -> dict[str, float]:
        angles = {
            "pad_joint_right": 0.0,
            "thigh_joint_right_1": math.radians(float(self.vars["right_thigh_a_deg"].get())),
            "calf_joint_right_1": math.radians(float(self.vars["right_calf_a_deg"].get())),
            "thigh_joint_right_2": math.radians(float(self.vars["right_thigh_b_deg"].get())),
            "calf_joint_right_2": math.radians(float(self.vars["right_calf_b_deg"].get())),
            "wheel_joint_right": 0.0,
            "pad_joint_left": 0.0,
            "thigh_joint_left_1": math.radians(float(self.vars["left_thigh_a_deg"].get())),
            "calf_joint_left_1": math.radians(float(self.vars["left_calf_a_deg"].get())),
            "thigh_joint_left_2": math.radians(float(self.vars["left_thigh_b_deg"].get())),
            "calf_joint_left_2": math.radians(float(self.vars["left_calf_b_deg"].get())),
            "wheel_joint_left": 0.0,
        }
        status = solver_status()
        if status.backend != "C":
            self.backend_warning = f"C backend unavailable: {status.message} | passive joints paused"
            return angles
        dirty = set(self.solve_dirty_sides)
        if "right" in dirty:
            right_calf_a, right_calf_b, right_error = self._solve_parallel_side(
                angles,
                calf_a="calf_joint_right_1",
                calf_b="calf_joint_right_2",
                wheel_link="wheel_link_right",
                branch_b_link="calf_right_link_2",
                loop_origin=np.array([-0.18, -0.03, 0.0]),
            )
            angles["calf_joint_right_1"] = right_calf_a
            angles["calf_joint_right_2"] = right_calf_b
            self.vars["right_calf_a_deg"].set(round(math.degrees(right_calf_a), 3))
            self.vars["right_calf_b_deg"].set(round(math.degrees(right_calf_b), 3))
            self.last_closure_error["right"] = right_error

        if "left" in dirty:
            left_calf_a, left_calf_b, left_error = self._solve_parallel_side(
                angles,
                calf_a="calf_joint_left_1",
                calf_b="calf_joint_left_2",
                wheel_link="wheel_link_left",
                branch_b_link="calf_left_link_2",
                loop_origin=np.array([-0.18, 0.03, 0.0]),
            )
            angles["calf_joint_left_1"] = left_calf_a
            angles["calf_joint_left_2"] = left_calf_b
            self.vars["left_calf_a_deg"].set(round(math.degrees(left_calf_a), 3))
            self.vars["left_calf_b_deg"].set(round(math.degrees(left_calf_b), 3))
            self.last_closure_error["left"] = left_error

        self.solve_dirty_sides.difference_update(dirty)
        return angles

    def _solve_parallel_side(
        self,
        base_angles: dict[str, float],
        calf_a: str,
        calf_b: str,
        wheel_link: str,
        branch_b_link: str,
        loop_origin: np.ndarray,
    ) -> tuple[float, float, float]:
        angles = np.asarray([base_angles[name] for name in JOINT_ANGLE_NAMES], dtype=np.float64)
        return solve_passive_pair(
            self._solver_chain(wheel_link),
            self._solver_chain(branch_b_link),
            angles,
            passive_a_index=JOINT_ANGLE_INDEX[calf_a],
            passive_b_index=JOINT_ANGLE_INDEX[calf_b],
            loop_origin=loop_origin,
            initial_a=base_angles[calf_a],
            initial_b=base_angles[calf_b],
            lower=math.radians(-90.0),
            upper=math.radians(90.0),
        )

    def _solver_chain(self, target_link: str) -> np.ndarray:
        if target_link in self.solver_chains:
            return self.solver_chains[target_link]
        if self.urdf is None:
            return np.empty((0, 16), dtype=np.float64)

        by_child = {joint.child: joint for joint in self.urdf.joints}
        joints: list[URDFJoint] = []
        link = target_link
        while link in by_child:
            joint = by_child[link]
            joints.append(joint)
            link = joint.parent
            if link == self.urdf.root_link:
                break
        joints.reverse()
        steps = [
            make_step(
                joint.origin_xyz,
                rpy_matrix(joint.origin_rpy),
                joint.axis,
                JOINT_ANGLE_INDEX.get(joint.name, -1) if joint.joint_type in {"revolute", "continuous"} else -1,
            )
            for joint in joints
        ]
        chain = np.vstack(steps) if steps else np.empty((0, 16), dtype=np.float64)
        self.solver_chains[target_link] = chain
        return chain

    def _draw_cylinder(
        self,
        center: Vec3,
        radius: float,
        length: float,
        axis: str,
        R: np.ndarray,
        color: str,
        edge_color: str = "#1d2430",
        segments: int = 28,
    ) -> None:
        x, y, z = cylinder_mesh(center, radius, length, axis=axis, rotation=R, segments=segments)
        self.ax.plot_surface(x, y, z, color=color, edgecolor=edge_color, linewidth=0.25, shade=True, alpha=0.98)

    def _add_faces(self, faces: list[np.ndarray], color: str, alpha: float = 1.0, edge: str = "#1f2633") -> None:
        collection = Poly3DCollection(faces, facecolor=color, edgecolor=edge, linewidth=0.6, alpha=alpha)
        self.ax.add_collection3d(collection)

    def _remember_camera(self) -> None:
        if not hasattr(self, "ax"):
            return
        elev = getattr(self.ax, "elev", None)
        azim = getattr(self.ax, "azim", None)
        roll = getattr(self.ax, "roll", 0.0)
        if elev is None or azim is None:
            return
        self.camera_view = (float(elev), float(azim), float(roll))

    def _set_camera(self, reset_camera: bool = False) -> None:
        self._apply_camera_limits(center_x=self.state.x, center_y=self.state.y)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_axis_off()

        if reset_camera or not self.camera_initialized or self.camera_view is None:
            camera = str(self.vars["camera"].get())
            if camera == "Front":
                self.camera_view = (8.0, -90.0, 0.0)
            elif camera == "Side":
                self.camera_view = (12.0, 0.0, 0.0)
            elif camera == "Top":
                self.camera_view = (89.0, -90.0, 0.0)
            else:
                self.camera_view = (24.0, -45.0, 0.0)
            self.camera_initialized = True

        elev, azim, roll = self.camera_view
        try:
            self.ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:
            self.ax.view_init(elev=elev, azim=azim)

    def _apply_camera_limits(
        self,
        center_x: float | None = None,
        center_y: float | None = None,
        center_z: float | None = None,
    ) -> None:
        span = max(float(self.vars["view_span"].get()), 0.35)
        if center_x is None:
            center_x = self.state.x
        if center_y is None:
            center_y = self.state.y
        if center_z is None:
            center_z = self._default_camera_z_center()

        z_half_span = max(span * 0.45, 0.08)
        self.ax.set_xlim(center_x - span, center_x + span)
        self.ax.set_ylim(center_y - span, center_y + span)
        self.ax.set_zlim(center_z - z_half_span, center_z + z_half_span)
        self.ax.set_box_aspect((1, 1, 0.45))

    def _default_camera_z_center(self) -> float:
        if self.urdf is None:
            return 1.8
        urdf_scale = float(self.vars["urdf_scale"].get())
        top = max(1.2, float(self.urdf.bounds_max[2] - self.urdf.bounds_min[2]) * urdf_scale) + 0.55
        return top * 0.5

    def _update_text(self) -> None:
        now = time.perf_counter()
        self.last_text_update = now
        status = (
            "STATIC | "
            f"solver={solver_status().backend} | "
            f"left closure={self.last_closure_error['left']:.5f} m | "
            f"right closure={self.last_closure_error['right']:.5f} m"
        )
        if self.backend_warning:
            status = f"{status} | {self.backend_warning}"
        self.status.configure(text=status)
        lines = [
            "Linkage telemetry",
            f"Render style: {self.vars['render_style'].get()}",
            f"Active A1/A2 L: {float(self.vars['left_thigh_a_deg'].get()):+.2f} / {float(self.vars['left_thigh_b_deg'].get()):+.2f} deg",
            f"Active A1/A2 R: {float(self.vars['right_thigh_a_deg'].get()):+.2f} / {float(self.vars['right_thigh_b_deg'].get()):+.2f} deg",
            f"Balance pitch/roll: {self.balance_pitch_deg:+.2f} / {self.balance_roll_deg:+.2f} deg",
            f"Balance tilt: {self.balance_tilt_deg:.2f} deg",
            f"Solver: {solver_status().backend}",
            f"Closure L/R: {self.last_closure_error['left']:.5f} / {self.last_closure_error['right']:.5f} m",
            f"Selected: {self.selected_item or '-'}",
            f"Passive B1/B2 L: {float(self.vars['left_calf_a_deg'].get()):+.2f} / {float(self.vars['left_calf_b_deg'].get()):+.2f} deg",
            f"Passive B1/B2 R: {float(self.vars['right_calf_a_deg'].get()):+.2f} / {float(self.vars['right_calf_b_deg'].get()):+.2f} deg",
        ]
        for side in ("left", "right"):
            point = self.end_effector_positions.get(side)
            if point is not None:
                lines.append(f"End effector {side[0].upper()}: x={point[0]:+.3f} y={point[1]:+.3f} z={point[2]:+.3f} m")
        if self.backend_warning:
            lines.extend(["", self.backend_warning])
        if self.urdf is not None:
            lines.extend(
                [
                    "",
                    f"URDF: {self.urdf.name}",
                    f"Root link: {self.urdf.root_link}",
                    f"Links/Joints: {len(self.urdf.links)} / {len(self.urdf.joints)}",
                    f"Bounded meshes: {self.urdf.mesh_count}",
                    f"Source facets read once: {self.urdf.triangle_count:,}",
                    "Rendered as lightweight solids",
                ]
            )
        elif self.urdf_error:
            lines.extend(["", self.urdf_error])
        self.readout.configure(state="normal")
        self.readout.delete("1.0", "end")
        self.readout.insert("1.0", "\n".join(lines))
        self.readout.configure(state="disabled")

def main() -> None:
    root = tk.Tk()
    app = RobotViewApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
