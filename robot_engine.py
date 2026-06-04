from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fivebar_solver import (
    compute_balance_plane,
    compute_supported_link_transforms,
    load_stl_bounds,
    make_step,
    make_tree_step,
    solve_passive_pair,
    solver_status,
)

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
DEFAULT_CAMERA = {"mode": "ISO", "elev": 30.0, "azim": 135.0, "roll": 0.0, "span": 1.35}
SCENE_SCALE = 4.0
ROOT_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
ZERO_VEC3 = np.zeros(3, dtype=np.float64)
LOOP_OFFSETS = {
    "right": np.array([-0.18, -0.03, 0.0], dtype=np.float64),
    "left": np.array([-0.18, 0.03, 0.0], dtype=np.float64),
}

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
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def rpy_matrix(rpy: Vec3) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def parse_vec(text: str | None, fallback: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(fallback, dtype=float)
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return np.asarray(fallback, dtype=float)
    return np.asarray([float(part) for part in parts], dtype=float)

def serial_vec(point: Vec3) -> list[float]:
    return [float(point[0]), float(point[1]), float(point[2])]

def serial_axis(rotation: np.ndarray, axis_index: int) -> list[float]:
    return [float(rotation[0, axis_index]), float(rotation[1, axis_index]), float(rotation[2, axis_index])]

def wheel_contact_point(center: np.ndarray, rotation: np.ndarray, radius: float) -> np.ndarray:
    plane_x = rotation[:, 0]
    plane_z = rotation[:, 2]
    vertical = np.array([plane_x[2], plane_z[2]], dtype=np.float64)
    vertical_norm = float(np.linalg.norm(vertical))
    if vertical_norm < 1e-9:
        return center.copy()
    return center - radius * ((vertical[0] / vertical_norm) * plane_x + (vertical[1] / vertical_norm) * plane_z)

def load_urdf_model(path: Path) -> URDFModel:
    tree = ET.parse(path)
    root = tree.getroot()
    package_dir = path.parents[1]
    links: dict[str, URDFLink] = {}
    joints: list[URDFJoint] = []
    child_joints: dict[str, list[URDFJoint]] = {}
    all_min = np.array([math.inf, math.inf, math.inf])
    all_max = np.array([-math.inf, -math.inf, -math.inf])
    mesh_count = 0
    triangle_count = 0

    for link_el in root.findall("link"):
        link_name = link_el.attrib.get("name", "")
        visuals: list[MeshVisual] = []
        for visual_el in link_el.findall("visual"):
            mesh_el = visual_el.find("./geometry/mesh")
            filename = mesh_el.attrib.get("filename", "") if mesh_el is not None else ""
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
                parts = [float(part) for part in color_el.attrib["rgba"].split()]
                if len(parts) >= 3:
                    rgba = tuple(parts[:3])
                alpha = parts[3] if len(parts) >= 4 else 1.0
            bounds_min, bounds_max, source_triangles = load_stl_bounds(mesh_path)
            visuals.append(MeshVisual(mesh_path, bounds_min, bounds_max, tuple(rgba) + (alpha,), origin_xyz, origin_rpy))
            all_min = np.minimum(all_min, bounds_min)
            all_max = np.maximum(all_max, bounds_max)
            mesh_count += 1
            triangle_count += source_triangles
        links[link_name] = URDFLink(link_name, visuals)

    seen_children: set[str] = set()
    for joint_el in root.findall("joint"):
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        child = child_el.attrib.get("link", "")
        if child in seen_children:
            continue
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        joint = URDFJoint(
            name=joint_el.attrib.get("name", ""),
            joint_type=joint_el.attrib.get("type", "fixed"),
            parent=parent_el.attrib.get("link", ""),
            child=child,
            origin_xyz=parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None),
            origin_rpy=parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None),
            axis=parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
        )
        joints.append(joint)
        child_joints.setdefault(joint.parent, []).append(joint)
        seen_children.add(joint.child)

    root_link = next((name for name in links if name not in seen_children), next(iter(links), "base_link"))
    if not np.isfinite(all_min).all():
        all_min = np.zeros(3)
        all_max = np.ones(3)
    return URDFModel(root.attrib.get("name", path.stem), root_link, links, joints, child_joints, mesh_count, triangle_count, all_min, all_max)

class RobotEngine:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.urdf_path = self.base_dir / "robot_urdf" / "urdf" / "robot.SLDASM.urdf"
        self.urdf: URDFModel | None = None
        self.urdf_error = ""
        self.link_order: list[str] = []
        self.link_index: dict[str, int] = {}
        self.tree_steps = np.empty((0, 13), dtype=np.float64)
        self.solver_chains: dict[str, np.ndarray] = {}
        self.solver_status = solver_status()
        self.joint_angle_array = np.zeros(len(JOINT_ANGLE_NAMES), dtype=np.float64)
        self.link_radii: dict[str, float] = {}
        self.link_half_widths: dict[str, float] = {}
        self.angles_deg = {
            "left_thigh_a_deg": 0.0,
            "left_thigh_b_deg": 0.0,
            "left_calf_a_deg": 0.0,
            "left_calf_b_deg": 0.0,
            "right_thigh_a_deg": 0.0,
            "right_thigh_b_deg": 0.0,
            "right_calf_a_deg": 0.0,
            "right_calf_b_deg": 0.0,
        }
        self.solve_dirty_sides: set[str] = {"left", "right"}
        self.last_closure_error = {"left": 0.0, "right": 0.0}
        self.backend_warning = ""
        self.load()

    def load(self) -> None:
        if not self.urdf_path.exists():
            self.urdf_error = "robot_urdf/urdf/robot.SLDASM.urdf not found"
            return
        try:
            self.urdf = load_urdf_model(self.urdf_path)
            self._prepare_c_kinematics()
            self._prepare_static_geometry(SCENE_SCALE)
            self.urdf_error = ""
        except Exception as exc:
            self.urdf = None
            self.urdf_error = f"URDF load failed: {exc}"

    def update_angles(self, values: dict[str, float]) -> dict:
        for key, value in values.items():
            if key in self.angles_deg:
                self.angles_deg[key] = float(value)
                if key.startswith("left_thigh"):
                    self.solve_dirty_sides.add("left")
                elif key.startswith("right_thigh"):
                    self.solve_dirty_sides.add("right")
        return self.scene()

    def reset_angles(self) -> dict:
        for key in self.angles_deg:
            self.angles_deg[key] = 0.0
        self.solve_dirty_sides = {"left", "right"}
        return self.scene()

    def scene(self) -> dict:
        if self.urdf is None:
            return {"ok": False, "error": self.urdf_error, "camera": DEFAULT_CAMERA}
        joint_angles = self._joint_angles()
        transforms = self._supported_link_transforms(ZERO_VEC3, ROOT_ROTATION, SCENE_SCALE, joint_angles)
        return {
            "ok": True,
            "camera": DEFAULT_CAMERA,
            "solver": self.solver_status.__dict__,
            "warning": self.backend_warning,
            "angles": self.angles_deg,
            "closure": self.last_closure_error,
            "linkage": self._linkage_scene(transforms, SCENE_SCALE),
            "meta": {
                "urdf": self.urdf.name,
                "root": self.urdf.root_link,
                "links": len(self.urdf.links),
                "joints": len(self.urdf.joints),
                "meshes": self.urdf.mesh_count,
                "triangles": self.urdf.triangle_count,
            },
        }

    def _prepare_c_kinematics(self) -> None:
        if self.urdf is None:
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
        self.link_order = order
        self.link_index = {name: index for index, name in enumerate(order)}
        steps = [
            make_tree_step(
                self.link_index[joint.parent],
                self.link_index[joint.child],
                joint.origin_xyz,
                joint.origin_rpy,
                joint.axis,
                JOINT_ANGLE_INDEX.get(joint.name, -1),
                joint.joint_type in {"revolute", "continuous"},
            )
            for joint in ordered_joints
        ]
        self.tree_steps = np.vstack(steps) if steps else np.empty((0, 13), dtype=np.float64)
        self.solver_chains.clear()

    def _prepare_static_geometry(self, scale: float) -> None:
        if self.urdf is None:
            return
        self.link_radii = {
            name: self._compute_link_radius(name, scale, 0.045 * scale)
            for name in ("wheel_link_left", "wheel_link_right")
        }
        self.link_half_widths = {
            name: self._compute_link_half_width(name, scale, 0.015 * scale)
            for name in ("wheel_link_left", "wheel_link_right")
        }

    def _joint_angles(self) -> np.ndarray:
        angles = self.joint_angle_array
        angles.fill(0.0)
        angles[JOINT_ANGLE_INDEX["thigh_joint_right_1"]] = math.radians(self.angles_deg["right_thigh_a_deg"])
        angles[JOINT_ANGLE_INDEX["calf_joint_right_1"]] = math.radians(self.angles_deg["right_calf_a_deg"])
        angles[JOINT_ANGLE_INDEX["thigh_joint_right_2"]] = math.radians(self.angles_deg["right_thigh_b_deg"])
        angles[JOINT_ANGLE_INDEX["calf_joint_right_2"]] = math.radians(self.angles_deg["right_calf_b_deg"])
        angles[JOINT_ANGLE_INDEX["thigh_joint_left_1"]] = math.radians(self.angles_deg["left_thigh_a_deg"])
        angles[JOINT_ANGLE_INDEX["calf_joint_left_1"]] = math.radians(self.angles_deg["left_calf_a_deg"])
        angles[JOINT_ANGLE_INDEX["thigh_joint_left_2"]] = math.radians(self.angles_deg["left_thigh_b_deg"])
        angles[JOINT_ANGLE_INDEX["calf_joint_left_2"]] = math.radians(self.angles_deg["left_calf_b_deg"])

        if self.solver_status.backend != "C":
            self.backend_warning = f"C backend unavailable: {self.solver_status.message}"
            return angles
        self.backend_warning = ""
        dirty = set(self.solve_dirty_sides)
        if "right" in dirty:
            a, b, err = self._solve_side(angles, "right", LOOP_OFFSETS["right"])
            angles[JOINT_ANGLE_INDEX["calf_joint_right_1"]] = a
            angles[JOINT_ANGLE_INDEX["calf_joint_right_2"]] = b
            self.angles_deg["right_calf_a_deg"] = round(math.degrees(a), 3)
            self.angles_deg["right_calf_b_deg"] = round(math.degrees(b), 3)
            self.last_closure_error["right"] = err
        if "left" in dirty:
            a, b, err = self._solve_side(angles, "left", LOOP_OFFSETS["left"])
            angles[JOINT_ANGLE_INDEX["calf_joint_left_1"]] = a
            angles[JOINT_ANGLE_INDEX["calf_joint_left_2"]] = b
            self.angles_deg["left_calf_a_deg"] = round(math.degrees(a), 3)
            self.angles_deg["left_calf_b_deg"] = round(math.degrees(b), 3)
            self.last_closure_error["left"] = err
        self.solve_dirty_sides.difference_update(dirty)
        return angles

    def _solve_side(self, angles: np.ndarray, side: str, loop_origin: np.ndarray) -> tuple[float, float, float]:
        passive_a_index = JOINT_ANGLE_INDEX[f"calf_joint_{side}_1"]
        passive_b_index = JOINT_ANGLE_INDEX[f"calf_joint_{side}_2"]
        return solve_passive_pair(
            self._solver_chain(f"wheel_link_{side}"),
            self._solver_chain(f"calf_{side}_link_2"),
            angles,
            passive_a_index=passive_a_index,
            passive_b_index=passive_b_index,
            loop_origin=loop_origin,
            initial_a=float(angles[passive_a_index]),
            initial_b=float(angles[passive_b_index]),
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

    def _supported_link_transforms(
        self,
        support_origin: Vec3,
        requested_root_rotation: np.ndarray,
        scale: float,
        joint_angles: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self.urdf is None:
            return {}
        wheel_radius = max(
            self.link_radii.get("wheel_link_left", 0.045 * scale),
            self.link_radii.get("wheel_link_right", 0.045 * scale),
        )
        origins, rotations = compute_supported_link_transforms(
            self.tree_steps,
            joint_angles,
            support_origin,
            requested_root_rotation,
            scale,
            len(self.link_order),
            self.link_index["wheel_link_left"],
            self.link_index["wheel_link_right"],
            wheel_radius,
        )
        return {name: (origins[index], rotations[index]) for index, name in enumerate(self.link_order)}

    def _link_radius(self, link_name: str, scale: float, fallback: float) -> float:
        return self.link_radii.get(link_name) or self._compute_link_radius(link_name, scale, fallback)

    def _compute_link_radius(self, link_name: str, scale: float, fallback: float) -> float:
        if self.urdf is None:
            return fallback
        link = self.urdf.links.get(link_name)
        if link is None or not link.visuals:
            return fallback
        extents = np.maximum((link.visuals[0].bounds_max - link.visuals[0].bounds_min) * scale, 1e-5)
        return max(float(extents[0]), float(extents[2])) * 0.5

    def _link_half_width(self, link_name: str, scale: float, fallback: float) -> float:
        return self.link_half_widths.get(link_name) or self._compute_link_half_width(link_name, scale, fallback)

    def _compute_link_half_width(self, link_name: str, scale: float, fallback: float) -> float:
        if self.urdf is None:
            return fallback
        link = self.urdf.links.get(link_name)
        if link is None or not link.visuals:
            return fallback
        extents = np.maximum((link.visuals[0].bounds_max - link.visuals[0].bounds_min) * scale, 1e-5)
        return float(extents[1]) * 0.5

    def _linkage_scene(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], scale: float) -> dict:
        lines: list[dict] = []
        points: list[dict] = []
        wheels: list[dict] = []
        for side, color, point_color in (("right", "#c9d1d9", "#58a6ff"), ("left", "#8b949e", "#3fb950")):
            required = {
                "A1": f"thigh_{side}_1",
                "A2": f"thigh_{side}_2",
                "B1": f"calf_{side}_link_1",
                "B2": f"calf_{side}_link_2",
                "P": f"wheel_link_{side}",
            }
            if any(name not in transforms for name in required.values()):
                continue
            pts = {label: transforms[name][0] for label, name in required.items()}
            b2_tip = pts["B2"] + (LOOP_OFFSETS[side] * scale) @ transforms[required["B2"]][1].T
            wheel_rotation = transforms[required["P"]][1]
            wheel_radius = self._link_radius(required["P"], scale, 0.045 * scale)
            wheel_half_width = self._link_half_width(required["P"], scale, 0.015 * scale)
            for a, b, line_color, width in (
                ("A1", "A2", "#6e7681", 2.0),
                ("A1", "B1", color, 4.0),
                ("A2", "B2", color, 4.0),
                ("B1", "P", color, 4.0),
            ):
                lines.append({"a": serial_vec(pts[a]), "b": serial_vec(pts[b]), "color": line_color, "width": width})
            lines.append({"a": serial_vec(pts["B2"]), "b": serial_vec(b2_tip), "color": color, "width": 4.0})
            closure_gap = float(np.linalg.norm(pts["P"] - b2_tip))
            if closure_gap > 1e-5:
                lines.append({"a": serial_vec(b2_tip), "b": serial_vec(pts["P"]), "color": "#f85149", "width": 1.5})
            for label in ("A1", "A2", "B1", "B2", "P"):
                points.append({"p": serial_vec(pts[label]), "label": label, "side": side, "color": point_color if label == "P" else "#8b949e"})
            contact = wheel_contact_point(pts["P"], wheel_rotation, wheel_radius)
            wheels.append(
                {
                    "center": serial_vec(pts["P"]),
                    "contact": serial_vec(contact),
                    "radius": wheel_radius,
                    "halfWidth": wheel_half_width,
                    "axisX": serial_axis(wheel_rotation, 0),
                    "axisY": serial_axis(wheel_rotation, 1),
                    "axisZ": serial_axis(wheel_rotation, 2),
                    "color": "#c9d1d9",
                }
            )

        balance = self._balance_plane(transforms, scale)
        grid = self._reference_grid(transforms, wheels)
        return {"lines": lines, "points": points, "wheels": wheels, "balance": balance, "grid": grid}

    def _balance_plane(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], scale: float) -> dict | None:
        names = ("thigh_right_1", "thigh_right_2", "thigh_left_1", "thigh_left_2")
        if any(name not in transforms for name in names):
            return None
        solved = compute_balance_plane(np.asarray([transforms[name][0] for name in names], dtype=np.float64), scale)
        if solved is None:
            return None
        corners, center, pitch, roll, tilt = solved
        return {"corners": [serial_vec(p) for p in corners], "center": serial_vec(center), "pitch": pitch, "roll": roll, "tilt": tilt}

    def _reference_grid(self, transforms: dict[str, tuple[np.ndarray, np.ndarray]], wheels: list[dict]) -> list[dict]:
        points = np.asarray([origin for origin, _rotation in transforms.values()])
        if len(points) == 0:
            return []
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        contact_z_values = [float(wheel["contact"][2]) for wheel in wheels if "contact" in wheel]
        z = min(contact_z_values) if contact_z_values else float(mins[2] - 0.015)
        step = max(0.05, float(np.linalg.norm(maxs[:2] - mins[:2])) / 8.0)
        x0, x1 = float(mins[0] - step), float(maxs[0] + step)
        y0, y1 = float(mins[1] - step), float(maxs[1] + step)
        lines: list[dict] = []
        for x in np.arange(x0, x1 + step, step):
            lines.append({"a": [float(x), y0, z], "b": [float(x), y1, z]})
        for y in np.arange(y0, y1 + step, step):
            lines.append({"a": [x0, float(y), z], "b": [x1, float(y), z]})
        return lines
