from __future__ import annotations

import math
import struct
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

matplotlib.use("TkAgg")

Vec3 = np.ndarray

@dataclass
class RobotState:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    left_wheel: float = 0.0
    right_wheel: float = 0.0

@dataclass
class STLMesh:
    name: str
    triangles: np.ndarray
    normals: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    triangle_count: int
    scale: float
    local_min: np.ndarray
    local_max: np.ndarray

@dataclass
class MeshVisual:
    path: Path
    triangles: np.ndarray
    normals: np.ndarray
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

def normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / n

def transform(points: np.ndarray, origin: Vec3, rotation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + origin

def rpy_matrix(rpy: Vec3) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def axis_angle_matrix(axis: Vec3, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=float))
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
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

def oriented_bar_faces(p0: Vec3, p1: Vec3, width: float, thickness: float) -> list[np.ndarray]:
    p0, p1 = np.asarray(p0), np.asarray(p1)
    axis = normalize(p1 - p0)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.92:
        helper = np.array([1.0, 0.0, 0.0])
    side = normalize(np.cross(axis, helper))
    up = normalize(np.cross(side, axis))
    rotation = np.column_stack((axis, side, up))
    length = float(np.linalg.norm(p1 - p0))
    center = (p0 + p1) / 2
    return box_faces(center, (length, width, thickness), rotation)

def load_stl_mesh(path: Path, max_triangles: int = 30000) -> STLMesh:
    size = path.stat().st_size
    with path.open("rb") as fh:
        header = fh.read(80)
        count_bytes = fh.read(4)
    if len(count_bytes) != 4:
        raise ValueError(f"{path.name} is too small to be an STL file")

    binary_count = struct.unpack("<I", count_bytes)[0]
    looks_binary = 84 + binary_count * 50 == size
    if looks_binary:
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attr", "<u2"),
            ]
        )
        records = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(binary_count,))
        all_vertices = records["vertices"].reshape(-1, 3)
        bounds_min = np.asarray(all_vertices.min(axis=0), dtype=float)
        bounds_max = np.asarray(all_vertices.max(axis=0), dtype=float)
        sample_count = min(max_triangles, binary_count)
        indices = np.linspace(0, binary_count - 1, sample_count, dtype=np.int64)
        triangles = np.asarray(records["vertices"][indices], dtype=float)
        del records
    else:
        vertices: list[list[float]] = []
        bounds_min = np.array([math.inf, math.inf, math.inf])
        bounds_max = np.array([-math.inf, -math.inf, -math.inf])
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped.startswith("vertex"):
                    continue
                _tag, x, y, z = stripped.split()[:4]
                vertex = [float(x), float(y), float(z)]
                vertices.append(vertex)
                bounds_min = np.minimum(bounds_min, vertex)
                bounds_max = np.maximum(bounds_max, vertex)
        if len(vertices) < 3:
            raise ValueError(f"{path.name} does not contain STL triangles")
        all_triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
        triangle_count = len(all_triangles)
        sample_count = min(max_triangles, triangle_count)
        indices = np.linspace(0, triangle_count - 1, sample_count, dtype=np.int64)
        triangles = all_triangles[indices]
        binary_count = triangle_count

    center = (bounds_min + bounds_max) / 2
    extents = bounds_max - bounds_min
    largest_xy = max(float(extents[0]), float(extents[1]), 1e-9)
    scale = 2.75 / largest_xy
    local = (triangles - center) * scale
    local[:, :, 2] -= local[:, :, 2].min() + 1.70
    v1 = local[:, 1] - local[:, 0]
    v2 = local[:, 2] - local[:, 0]
    normals = np.cross(v1, v2)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-9
    normals[valid] /= lengths[valid, None]
    local_min = local.reshape(-1, 3).min(axis=0)
    local_max = local.reshape(-1, 3).max(axis=0)
    return STLMesh(
        name=path.name,
        triangles=local,
        normals=normals,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        triangle_count=int(binary_count),
        scale=scale,
        local_min=local_min,
        local_max=local_max,
    )

def load_stl_triangles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        _header = fh.read(80)
        count_bytes = fh.read(4)
    if len(count_bytes) != 4:
        raise ValueError(f"{path.name} is too small to be an STL file")

    binary_count = struct.unpack("<I", count_bytes)[0]
    looks_binary = 84 + binary_count * 50 == size
    if looks_binary:
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attr", "<u2"),
            ]
        )
        records = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(binary_count,))
        triangles = np.asarray(records["vertices"], dtype=float)
        del records
    else:
        vertices: list[list[float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped.startswith("vertex"):
                    continue
                _tag, x, y, z = stripped.split()[:4]
                vertices.append([float(x), float(y), float(z)])
        if len(vertices) < 3:
            raise ValueError(f"{path.name} does not contain STL triangles")
        triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)

    v1 = triangles[:, 1] - triangles[:, 0]
    v2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(v1, v2)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-9
    normals[valid] /= lengths[valid, None]
    bounds_min = triangles.reshape(-1, 3).min(axis=0)
    bounds_max = triangles.reshape(-1, 3).max(axis=0)
    return triangles, normals, bounds_min, bounds_max

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
            triangles, normals, bounds_min, bounds_max = load_stl_triangles(mesh_path)
            visuals.append(MeshVisual(mesh_path, triangles, normals, tuple(rgba) + (alpha,), origin_xyz, origin_rpy))
            all_min = np.minimum(all_min, bounds_min)
            all_max = np.maximum(all_max, bounds_max)
            triangle_count += len(triangles)
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
        self.running = False
        self.last_tick = time.perf_counter()
        self.last_render = 0.0
        self.last_text_update = 0.0
        self.trail: list[tuple[float, float]] = []
        self.mesh: STLMesh | None = None
        self.mesh_error = ""
        self.urdf: URDFModel | None = None
        self.urdf_error = ""

        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._load_robot_mesh()
        self._load_urdf_model()
        self._build_ui()
        self._apply_style()
        self.draw_scene()
        self._tick()

    def _build_vars(self) -> None:
        defaults = {
            "speed": 0.75,
            "turn": 0.0,
            "wheel_radius": 0.06,
            "wheel_track": 0.42,
            "body_h": 1.75,
            "terrain": 7.0,
            "grid": 0.5,
            "obstacle": True,
            "ghost": True,
            "show_axes": False,
            "use_stl": True,
            "display_mode": "URDF",
            "urdf_scale": 4.0,
            "urdf_detail": 12000.0,
            "show_joints": True,
            "mesh_detail": 5000.0,
            "live_detail": 800.0,
            "fps": 8.0,
            "mesh_edges": True,
            "camera": "ISO",
            "pitch": 0.0,
            "roll": 0.0,
        }
        for key, value in defaults.items():
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, str):
                self.vars[key] = tk.StringVar(value=value)
            else:
                self.vars[key] = tk.DoubleVar(value=value)

    def _load_robot_mesh(self) -> None:
        base_dir = Path(__file__).resolve().parent
        candidates = [base_dir / "robot.stl", base_dir / "robot.STL", base_dir / "ver2_full.STL"]
        stl_path = next((path for path in candidates if path.exists()), None)
        if stl_path is None:
            self.mesh_error = "robot.stl not found"
            return
        try:
            self.mesh = load_stl_mesh(stl_path, max_triangles=50000)
            self.mesh_error = ""
        except Exception as exc:
            self.mesh = None
            self.mesh_error = f"STL load failed: {exc}"

    def _load_urdf_model(self) -> None:
        base_dir = Path(__file__).resolve().parent
        urdf_path = base_dir / "robot_urdf" / "urdf" / "robot.SLDASM.urdf"
        if not urdf_path.exists():
            self.urdf_error = "robot_urdf/urdf/robot.SLDASM.urdf not found"
            return
        try:
            self.urdf = load_urdf_model(urdf_path)
            self.urdf_error = ""
        except Exception as exc:
            self.urdf = None
            self.urdf_error = f"URDF load failed: {exc}"

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
        self.canvas = FigureCanvasTkAgg(self.fig, master=view_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, view_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

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
        for i in range(3):
            buttons.columnconfigure(i, weight=1)
        ttk.Button(buttons, text="Play", style="Run.TButton", command=self.play).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Pause", command=self.pause).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(buttons, text="Reset", command=self.reset).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        row += 1

        row = self._section(side, row, "Wheel Drive")
        row = self._slider(side, row, "Forward speed", "speed", -2.0, 2.0, 0.01)
        row = self._slider(side, row, "Turn rate", "turn", -1.8, 1.8, 0.01)
        row = self._slider(side, row, "Wheel radius", "wheel_radius", 0.03, 0.12, 0.001)
        row = self._slider(side, row, "Wheel track", "wheel_track", 0.25, 0.75, 0.01)

        row = self._section(side, row, "Body Pose")
        row = self._slider(side, row, "Body height", "body_h", 1.1, 2.6, 0.01)
        row = self._slider(side, row, "Pitch deg", "pitch", -18, 18, 0.1)
        row = self._slider(side, row, "Roll deg", "roll", -18, 18, 0.1)

        row = self._section(side, row, "Robot Mesh")
        display_mode = ttk.Combobox(
            side,
            textvariable=self.vars["display_mode"],
            values=("URDF", "STL", "Wheel Placeholder"),
            state="readonly",
        )
        display_mode.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        display_mode.bind("<<ComboboxSelected>>", lambda _event: self.draw_scene())
        row += 1
        ttk.Checkbutton(side, text="Use robot.stl fallback", variable=self.vars["use_stl"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        row = self._slider(side, row, "URDF scale", "urdf_scale", 1.0, 8.0, 0.1)
        row = self._slider(side, row, "URDF triangles", "urdf_detail", 1000, 50000, 500)
        row = self._slider(side, row, "Mesh triangles", "mesh_detail", 500, 50000, 500)
        row = self._slider(side, row, "Live triangles", "live_detail", 100, 8000, 100)
        row = self._slider(side, row, "Simulation FPS", "fps", 4, 30, 1)
        ttk.Checkbutton(side, text="Show URDF joint pivots", variable=self.vars["show_joints"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        ttk.Checkbutton(side, text="Draw mesh edges", variable=self.vars["mesh_edges"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        row = self._section(side, row, "Environment")
        row = self._slider(side, row, "Terrain size", "terrain", 5, 14, 0.5)
        row = self._slider(side, row, "Grid spacing", "grid", 0.25, 1.0, 0.05)
        ttk.Checkbutton(side, text="Show obstacles", variable=self.vars["obstacle"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        ttk.Checkbutton(side, text="Show trail", variable=self.vars["ghost"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1
        ttk.Checkbutton(side, text="Show axes", variable=self.vars["show_axes"], command=self.draw_scene).grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        row = self._section(side, row, "Camera")
        camera = ttk.Combobox(side, textvariable=self.vars["camera"], values=("ISO", "Front", "Side", "Top"), state="readonly")
        camera.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        camera.bind("<<ComboboxSelected>>", lambda _event: self.draw_scene())
        row += 1
        ttk.Button(side, text="Snapshot redraw", command=self.draw_scene).grid(row=row, column=0, sticky="ew", pady=4)
        row += 1

        self.readout = tk.Text(side, height=8, width=36, relief="solid", bd=1, bg="#ffffff", fg="#1f2937")
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

    def _slider(self, parent: ttk.Frame, row: int, label: str, key: str, start: float, end: float, step: float) -> int:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        value = ttk.Label(frame, textvariable=self.vars[key], width=7, anchor="e")
        value.grid(row=0, column=1, sticky="e")
        scale = ttk.Scale(frame, from_=start, to=end, variable=self.vars[key], command=lambda _v: self.draw_scene())
        scale.grid(row=1, column=0, columnspan=2, sticky="ew")
        spin = ttk.Spinbox(frame, from_=start, to=end, increment=step, textvariable=self.vars[key], width=8, command=self.draw_scene)
        spin.grid(row=2, column=1, sticky="e", pady=(2, 0))
        return row + 1

    def play(self) -> None:
        self.running = True
        self.last_tick = time.perf_counter()

    def pause(self) -> None:
        self.running = False
        self.draw_scene()

    def reset(self) -> None:
        self.running = False
        self.state = RobotState()
        self.trail.clear()
        self.draw_scene()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.05, now - self.last_tick)
        self.last_tick = now
        if self.running:
            self.state.t += dt
            self.state.yaw += float(self.vars["turn"].get()) * dt
            speed = float(self.vars["speed"].get())
            turn = float(self.vars["turn"].get())
            self.state.x += math.cos(self.state.yaw) * speed * dt
            self.state.y += math.sin(self.state.yaw) * speed * dt
            wheel_radius = max(0.001, float(self.vars["wheel_radius"].get()))
            wheel_track = float(self.vars["wheel_track"].get())
            self.state.left_wheel -= (speed - turn * wheel_track / 2.0) * dt / wheel_radius
            self.state.right_wheel -= (speed + turn * wheel_track / 2.0) * dt / wheel_radius
            self.trail.append((self.state.x, self.state.y))
            self.trail = self.trail[-140:]
            fps = max(1.0, float(self.vars["fps"].get()))
            if now - self.last_render >= 1.0 / fps:
                self.last_render = now
                self.draw_scene()
        self.root.after(16, self._tick)

    def body_rotation(self) -> np.ndarray:
        pitch = math.radians(float(self.vars["pitch"].get()))
        roll = math.radians(float(self.vars["roll"].get()))
        return rot_z(self.state.yaw) @ rot_y(pitch) @ rot_x(roll)

    def draw_scene(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#eef1f5")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self._draw_environment()
        self._draw_robot()
        self._set_camera()
        self._update_text()
        self.canvas.draw_idle()

    def _draw_environment(self) -> None:
        terrain = float(self.vars["terrain"].get())
        grid = float(self.vars["grid"].get())
        span = np.arange(-terrain, terrain + grid, grid)
        lines = []
        for p in span:
            lines.append([(-terrain, p, 0), (terrain, p, 0)])
            lines.append([(p, -terrain, 0), (p, terrain, 0)])
        self.ax.add_collection3d(Line3DCollection(lines, colors="#cbd3df", linewidths=0.55, alpha=0.85))
        axes = [[(-terrain, 0, 0.002), (terrain, 0, 0.002)], [(0, -terrain, 0.002), (0, terrain, 0.002)]]
        self.ax.add_collection3d(Line3DCollection(axes, colors="#64748b", linewidths=1.2))

        if bool(self.vars["obstacle"].get()):
            obstacles = [
                ((1.8, -1.15, 0.13), (0.85, 0.42, 0.26)),
                ((-1.5, 1.35, 0.08), (0.7, 0.7, 0.16)),
                ((2.8, 1.7, 0.22), (0.35, 1.15, 0.44)),
            ]
            for center, size in obstacles:
                self._add_faces(box_faces(np.array(center), size), "#a7b3c2", alpha=0.72, edge="#6b7280")

        if bool(self.vars["ghost"].get()) and len(self.trail) > 1:
            xs, ys = zip(*self.trail)
            self.ax.plot(xs, ys, [0.035] * len(xs), color="#2f80ed", linewidth=2.0, alpha=0.75)

    def _draw_robot(self) -> None:
        mode = str(self.vars["display_mode"].get())
        if mode == "URDF" and self.urdf is not None:
            self._draw_urdf_robot()
            return
        if mode == "STL" and self.mesh is not None and bool(self.vars["use_stl"].get()):
            self._draw_stl_robot()
            return

        self._draw_wheel_biped_placeholder()

    def _draw_wheel_biped_placeholder(self) -> None:
        R = self.body_rotation()
        wheel_radius = float(self.vars["wheel_radius"].get())
        wheel_track = float(self.vars["wheel_track"].get())
        body_h = max(float(self.vars["body_h"].get()), wheel_radius * 2.4)
        origin = np.array([self.state.x, self.state.y, body_h])

        body_color = "#5f697b"
        link_color = "#7d8798"
        wheel_color = "#242933"
        axle_color = "#2f3645"

        self._add_faces(box_faces(origin, (0.70, 0.46, 0.42), R), body_color, 0.98)
        self._add_faces(box_faces(transform(np.array([[0, 0, 0.30]]), origin, R)[0], (0.76, 0.52, 0.10), R), "#aab3c5", 0.98)

        for side, spin in ((-1.0, self.state.left_wheel), (1.0, self.state.right_wheel)):
            wheel_center = transform(np.array([[0.0, side * wheel_track / 2.0, wheel_radius]]), np.array([self.state.x, self.state.y, 0.0]), R)[0]
            shoulder = transform(np.array([[0.0, side * wheel_track * 0.36, -0.28]]), origin, R)[0]
            self._add_faces(oriented_bar_faces(shoulder, wheel_center, 0.045, 0.055), link_color, 0.96)
            self._draw_cylinder(wheel_center, wheel_radius, 0.055, "y", R, wheel_color)
            self._draw_cylinder(wheel_center, wheel_radius * 0.42, 0.075, "y", R, axle_color)
            self._draw_wheel_spokes(wheel_center, wheel_radius, spin, R)

    def _draw_urdf_robot(self) -> None:
        if self.urdf is None:
            return
        body_R = self.body_rotation()
        body_h = float(self.vars["body_h"].get())
        scale = float(self.vars["urdf_scale"].get())
        body_origin = np.array([self.state.x, self.state.y, body_h])
        root_R = body_R @ rot_z(math.radians(90.0))
        root_origin = body_origin.copy()
        transforms = self._urdf_link_transforms(root_origin, root_R, scale)
        bounds_min, bounds_max = self._urdf_world_bounds(transforms, scale)
        ground_clearance = 0.015
        bounds_center_xy = (bounds_min[:2] + bounds_max[:2]) / 2.0
        root_origin[:2] += body_origin[:2] - bounds_center_xy
        root_origin[2] += ground_clearance - bounds_min[2]
        transforms = self._urdf_link_transforms(root_origin, root_R, scale)
        total = max(1, self.urdf.triangle_count)
        requested = int(float(self.vars["urdf_detail"].get()))
        budget = max(1, min(requested, total))
        ratio = budget / total

        for link_name, (origin, rotation) in transforms.items():
            link = self.urdf.links.get(link_name)
            if link is None:
                continue
            for visual in link.visuals:
                visual_R = rotation @ rpy_matrix(visual.origin_rpy)
                visual_origin = origin + (visual.origin_xyz * scale) @ rotation.T
                triangles = visual.triangles * scale
                detail = max(1, min(len(triangles), int(len(triangles) * ratio)))
                indices = np.arange(len(triangles), dtype=np.int64) if detail >= len(triangles) else np.linspace(0, len(triangles) - 1, detail, dtype=np.int64)
                faces = triangles[indices].reshape(-1, 3) @ visual_R.T + visual_origin
                faces = faces.reshape(-1, 3, 3)
                face_colors = self._urdf_face_colors(visual, indices, visual_R)
                draw_edges = bool(self.vars["mesh_edges"].get())
                mesh = Poly3DCollection(
                    faces,
                    facecolor=face_colors,
                    edgecolor="#26303f" if draw_edges else "none",
                    linewidth=0.12 if draw_edges else 0.0,
                    alpha=visual.color[3],
                )
                mesh.set_zsort("average")
                self.ax.add_collection3d(mesh)

        if bool(self.vars["show_joints"].get()):
            points = np.asarray([value[0] for name, value in transforms.items() if name != self.urdf.root_link])
            if len(points):
                self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color="#2563eb", depthshade=False)

    def _urdf_link_transforms(self, root_origin: Vec3, root_rotation: np.ndarray, scale: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.urdf is None:
            return {}
        transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {self.urdf.root_link: (root_origin, root_rotation)}
        stack = [self.urdf.root_link]
        joint_angles = self._urdf_joint_angles()
        while stack:
            parent = stack.pop()
            parent_origin, parent_R = transforms[parent]
            for joint in self.urdf.child_joints.get(parent, []):
                joint_origin = parent_origin + (joint.origin_xyz * scale) @ parent_R.T
                joint_R = parent_R @ rpy_matrix(joint.origin_rpy)
                angle = joint_angles.get(joint.name, 0.0)
                if joint.joint_type in {"revolute", "continuous"}:
                    angle_R = axis_angle_matrix(joint.axis, angle)
                    child_R = joint_R @ angle_R
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
                points = visual.triangles.reshape(-1, 3) * scale
                world_points = points @ visual_R.T + visual_origin
                bounds_min = np.minimum(bounds_min, world_points.min(axis=0))
                bounds_max = np.maximum(bounds_max, world_points.max(axis=0))
        if not np.isfinite(bounds_min).all():
            return np.zeros(3), np.ones(3)
        return bounds_min, bounds_max

    def _urdf_joint_angles(self) -> dict[str, float]:
        # Wheel-bipedal drive: linkage pose stays assembled while wheel joints roll.
        return {
            "pad_joint_right": 0.0,
            "thigh_joint_right_1": 0.0,
            "calf_joint_right_1": 0.0,
            "thigh_joint_right_2": 0.0,
            "calf_joint_right_2": 0.0,
            "wheel_joint_right": self.state.right_wheel,
            "pad_joint_left": 0.0,
            "thigh_joint_left_1": 0.0,
            "calf_joint_left_1": 0.0,
            "thigh_joint_left_2": 0.0,
            "calf_joint_left_2": 0.0,
            "wheel_joint_left": self.state.left_wheel,
        }

    def _urdf_face_colors(self, visual: MeshVisual, indices: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        base = np.array(visual.color[:3])
        highlight = np.clip(base + 0.18, 0.0, 1.0)
        shadow = np.clip(base * 0.52, 0.0, 1.0)
        normals = visual.normals[indices]
        light = normalize(np.array([-0.35, -0.55, 0.76]))
        rotated_light = rotation @ light
        intensity = np.clip(normals @ rotated_light, -0.35, 1.0)
        pos = np.clip(intensity, 0.0, 1.0)[:, None]
        neg = np.clip(-intensity, 0.0, 0.35)[:, None]
        colors = base * (1 - pos * 0.42 - neg * 0.55) + highlight * (pos * 0.42) + shadow * (neg * 0.55)
        alpha = np.full((len(indices), 1), visual.color[3])
        return np.hstack((np.clip(colors, 0.0, 1.0), alpha))

    def _draw_stl_robot(self) -> None:
        if self.mesh is None:
            return
        R = self.body_rotation()
        body_h = float(self.vars["body_h"].get())
        origin = np.array([self.state.x, self.state.y, body_h])
        detail = self._effective_mesh_detail()
        detail = max(1, min(detail, len(self.mesh.triangles)))
        indices = self._mesh_lod_indices(detail)
        local = self.mesh.triangles[indices]
        world_faces = local.reshape(-1, 3) @ R.T + origin
        world_faces = world_faces.reshape(-1, 3, 3)
        draw_edges = bool(self.vars["mesh_edges"].get()) and not self.running
        face_colors = "#687184" if self.running else self._mesh_face_colors(indices, R)
        mesh = Poly3DCollection(
            world_faces,
            facecolor=face_colors,
            edgecolor="#242b38" if draw_edges else "none",
            linewidth=0.10 if draw_edges else 0.0,
            alpha=0.98,
        )
        mesh.set_zsort("average")
        self.ax.add_collection3d(mesh)

        self.ax.scatter([origin[0]], [origin[1]], [origin[2]], s=12, color="#2f80ed", depthshade=False)

    def _effective_mesh_detail(self) -> int:
        if self.mesh is None:
            return 0
        requested = int(float(self.vars["mesh_detail"].get()))
        if self.running:
            requested = min(requested, int(float(self.vars["live_detail"].get())))
        return max(1, min(requested, len(self.mesh.triangles)))

    def _mesh_lod_indices(self, detail: int) -> np.ndarray:
        if self.mesh is None:
            return np.empty((0,), dtype=np.int64)
        total = len(self.mesh.triangles)
        if detail >= total:
            return np.arange(total, dtype=np.int64)
        return np.linspace(0, total - 1, detail, dtype=np.int64)

    def _mesh_face_colors(self, indices: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        if self.mesh is None:
            return np.empty((0, 4))
        base = np.array(to_rgb("#6f788c"))
        highlight = np.array(to_rgb("#a8b1c2"))
        shadow = np.array(to_rgb("#434b5a"))
        normals = self.mesh.normals[indices]
        light = normalize(np.array([-0.35, -0.55, 0.76]))
        rotated_light = rotation @ light
        intensity = np.clip(normals @ rotated_light, -0.35, 1.0)
        pos = np.clip(intensity, 0.0, 1.0)[:, None]
        neg = np.clip(-intensity, 0.0, 0.35)[:, None]
        colors = base * (1 - pos * 0.45 - neg * 0.55) + highlight * (pos * 0.45) + shadow * (neg * 0.55)
        colors = np.clip(colors, 0.0, 1.0)
        alpha = np.full((len(indices), 1), 0.98)
        colors = np.hstack((colors, alpha))
        return colors

    def _draw_wheel_spokes(self, center: Vec3, radius: float, spin: float, R: np.ndarray) -> None:
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            local = np.array([[0, 0, 0], [radius * 0.72 * math.cos(angle + spin), 0, radius * 0.72 * math.sin(angle + spin)]])
            spoke = transform(local, center, R)
            self.ax.plot([spoke[0, 0], spoke[1, 0]], [spoke[0, 1], spoke[1, 1]], [spoke[0, 2], spoke[1, 2]], color="#aeb7c7", linewidth=1.0)

    def _draw_cylinder(self, center: Vec3, radius: float, length: float, axis: str, R: np.ndarray, color: str) -> None:
        x, y, z = cylinder_mesh(center, radius, length, axis=axis, rotation=R)
        self.ax.plot_surface(x, y, z, color=color, edgecolor="#1d2430", linewidth=0.25, shade=True, alpha=0.98)

    def _add_faces(self, faces: list[np.ndarray], color: str, alpha: float = 1.0, edge: str = "#1f2633") -> None:
        collection = Poly3DCollection(faces, facecolor=color, edgecolor=edge, linewidth=0.6, alpha=alpha)
        self.ax.add_collection3d(collection)

    def _set_camera(self) -> None:
        terrain = float(self.vars["terrain"].get())
        center_x, center_y = self.state.x, self.state.y
        span = max(terrain, 3.2)
        self.ax.set_xlim(center_x - span, center_x + span)
        self.ax.set_ylim(center_y - span, center_y + span)
        mode = str(self.vars["display_mode"].get())
        if mode == "URDF" and self.urdf is not None:
            urdf_scale = float(self.vars["urdf_scale"].get())
            top = float(self.vars["body_h"].get()) + max(1.2, float(self.urdf.bounds_max[2] - self.urdf.bounds_min[2]) * urdf_scale) + 0.35
            self.ax.set_zlim(0, max(2.4, top))
        elif mode == "STL" and self.mesh is not None and bool(self.vars["use_stl"].get()):
            top = float(self.vars["body_h"].get()) + float(self.mesh.local_max[2]) + 0.35
            self.ax.set_zlim(0, max(3.0, top))
        else:
            self.ax.set_zlim(0, 3.6)
        self.ax.set_box_aspect((1, 1, 0.45))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        if not bool(self.vars["show_axes"].get()):
            self.ax.set_axis_off()

        camera = str(self.vars["camera"].get())
        if camera == "Front":
            self.ax.view_init(elev=8, azim=-90)
        elif camera == "Side":
            self.ax.view_init(elev=12, azim=0)
        elif camera == "Top":
            self.ax.view_init(elev=89, azim=-90)
        else:
            self.ax.view_init(elev=24, azim=-45)

    def _update_text(self) -> None:
        now = time.perf_counter()
        if self.running and now - self.last_text_update < 0.25:
            return
        self.last_text_update = now
        speed = float(self.vars["speed"].get())
        turn = float(self.vars["turn"].get())
        wheel_radius = max(0.001, float(self.vars["wheel_radius"].get()))
        wheel_track = float(self.vars["wheel_track"].get())
        left_wheel_speed = -(speed - turn * wheel_track / 2.0) / wheel_radius
        right_wheel_speed = -(speed + turn * wheel_track / 2.0) / wheel_radius
        status = (
            f"{'RUNNING' if self.running else 'PAUSED'} | "
            f"x={self.state.x:+.2f} y={self.state.y:+.2f} yaw={math.degrees(self.state.yaw):+.1f} deg | "
            f"speed={float(self.vars['speed'].get()):+.2f} m/s"
        )
        self.status.configure(text=status)
        lines = [
            "Live telemetry",
            f"Display: {self.vars['display_mode'].get()}",
            f"Time: {self.state.t:6.2f} s",
            f"Body height: {float(self.vars['body_h'].get()):.2f} m",
            f"Pitch/Roll: {float(self.vars['pitch'].get()):+.1f} / {float(self.vars['roll'].get()):+.1f} deg",
            f"Wheel L/R: {self.state.left_wheel:+.1f} / {self.state.right_wheel:+.1f} rad",
            f"Wheel speed L/R: {left_wheel_speed:+.1f} / {right_wheel_speed:+.1f} rad/s",
            f"Trail points: {len(self.trail)}",
            "Mouse: rotate, pan, zoom in viewport",
        ]
        mode = str(self.vars["display_mode"].get())
        if mode == "URDF" and self.urdf is not None:
            lines.extend(
                [
                    "",
                    f"URDF: {self.urdf.name}",
                    f"Root link: {self.urdf.root_link}",
                    f"Links/Joints: {len(self.urdf.links)} / {len(self.urdf.joints)}",
                    f"Meshes: {self.urdf.mesh_count}",
                    f"Source triangles: {self.urdf.triangle_count:,}",
                ]
            )
        elif mode == "URDF" and self.urdf_error:
            lines.extend(["", self.urdf_error])
        elif mode == "STL" and self.mesh is not None:
            lines.extend(
                [
                    "",
                    f"STL: {self.mesh.name}",
                    f"Source triangles: {self.mesh.triangle_count:,}",
                    f"Rendered triangles: {self._effective_mesh_detail():,}",
                    f"Auto scale: {self.mesh.scale:.6g}",
                ]
            )
        elif mode == "STL" and self.mesh_error:
            lines.extend(["", self.mesh_error])
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
