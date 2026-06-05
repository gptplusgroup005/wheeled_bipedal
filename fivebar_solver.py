from __future__ import annotations

import ctypes
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP_WIDTH = 16
TREE_STEP_WIDTH = 13
DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
INT_PTR = ctypes.POINTER(ctypes.c_int)

@dataclass(frozen=True)
class SolverStatus:
    backend: str
    message: str

_STATUS = SolverStatus("unavailable", "C solver not loaded")
_LIB: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False

def solver_status() -> SolverStatus:
    _load_c_solver()
    return _STATUS

def make_step(origin_xyz: np.ndarray, origin_rotation: np.ndarray, axis: np.ndarray, angle_index: int) -> np.ndarray:
    return np.asarray(
        [
            float(origin_xyz[0]),
            float(origin_xyz[1]),
            float(origin_xyz[2]),
            *np.asarray(origin_rotation, dtype=np.float64).reshape(9),
            float(axis[0]),
            float(axis[1]),
            float(axis[2]),
            float(angle_index),
        ],
        dtype=np.float64,
    )

def make_tree_step(parent_index: int, child_index: int, origin_xyz: np.ndarray, origin_rpy: np.ndarray, axis: np.ndarray, angle_index: int, movable: bool) -> np.ndarray:
    return np.asarray(
        [
            float(parent_index),
            float(child_index),
            float(origin_xyz[0]),
            float(origin_xyz[1]),
            float(origin_xyz[2]),
            float(origin_rpy[0]),
            float(origin_rpy[1]),
            float(origin_rpy[2]),
            float(axis[0]),
            float(axis[1]),
            float(axis[2]),
            float(angle_index),
            1.0 if movable else 0.0,
        ],
        dtype=np.float64,
    )

def compute_link_transforms(
    tree_steps: np.ndarray,
    angles: np.ndarray,
    root_origin: np.ndarray,
    root_rotation: np.ndarray,
    scale: float,
    link_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree_steps = _as_tree_steps(tree_steps)
    angles = np.asarray(angles, dtype=np.float64)
    root_origin = np.asarray(root_origin, dtype=np.float64)
    root_rotation = np.ascontiguousarray(np.asarray(root_rotation, dtype=np.float64).reshape(9))
    origins = np.zeros((link_count, 3), dtype=np.float64)
    rotations = np.zeros((link_count, 3, 3), dtype=np.float64)

    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for FK transforms: {_STATUS.message}")

    ok = lib.compute_link_transforms_c(
        tree_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(tree_steps)),
        angles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(angles)),
        root_origin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        root_rotation.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(scale),
        ctypes.c_int(link_count),
        origins.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rotations.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not ok:
        raise RuntimeError("C backend failed to compute FK transforms")
    return origins, rotations

def compute_supported_link_transforms(
    tree_steps: np.ndarray,
    angles: np.ndarray,
    support_origin: np.ndarray,
    requested_root_rotation: np.ndarray,
    scale: float,
    link_count: int,
    wheel_left_index: int,
    wheel_right_index: int,
    wheel_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    tree_steps = _as_tree_steps(tree_steps)
    angles = np.asarray(angles, dtype=np.float64)
    support_origin = np.asarray(support_origin, dtype=np.float64)
    requested_root_rotation = np.ascontiguousarray(np.asarray(requested_root_rotation, dtype=np.float64).reshape(9))
    origins = np.zeros((link_count, 3), dtype=np.float64)
    rotations = np.zeros((link_count, 3, 3), dtype=np.float64)

    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for supported FK transforms: {_STATUS.message}")

    ok = lib.compute_supported_link_transforms_c(
        tree_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(tree_steps)),
        angles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(angles)),
        support_origin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        requested_root_rotation.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(scale),
        ctypes.c_int(link_count),
        ctypes.c_int(wheel_left_index),
        ctypes.c_int(wheel_right_index),
        ctypes.c_double(wheel_radius),
        origins.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rotations.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not ok:
        raise RuntimeError("C backend failed to compute supported FK transforms")
    return origins, rotations

def load_stl_bounds(path: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    path = Path(path)
    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for STL bounds: {_STATUS.message}")

    bounds_min = np.zeros(3, dtype=np.float64)
    bounds_max = np.zeros(3, dtype=np.float64)
    triangle_count = ctypes.c_int()
    ok = lib.load_stl_bounds_c(
        str(path).encode("utf-8"),
        bounds_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bounds_max.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(triangle_count),
    )
    if not ok:
        raise RuntimeError(f"C backend failed to read STL bounds: {path.name}")
    return bounds_min, bounds_max, int(triangle_count.value)

def solve_passive_pair(
    wheel_chain: np.ndarray,
    branch_chain: np.ndarray,
    angles: np.ndarray,
    passive_a_index: int,
    passive_b_index: int,
    loop_origin: np.ndarray,
    initial_a: float,
    initial_b: float,
    lower: float = -math.pi / 2.0,
    upper: float = math.pi / 2.0,
) -> tuple[float, float, float]:
    wheel_chain = _as_chain(wheel_chain)
    branch_chain = _as_chain(branch_chain)
    angles = np.asarray(angles, dtype=np.float64)
    loop_origin = np.asarray(loop_origin, dtype=np.float64)

    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for the five-bar solver: {_STATUS.message}")

    out = (ctypes.c_double * 3)()
    ok = lib.solve_passive_pair_c(
        wheel_chain.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(wheel_chain)),
        branch_chain.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(branch_chain)),
        angles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(angles)),
        ctypes.c_int(passive_a_index),
        ctypes.c_int(passive_b_index),
        loop_origin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(initial_a),
        ctypes.c_double(initial_b),
        ctypes.c_double(lower),
        ctypes.c_double(upper),
        out,
    )
    if not ok:
        raise RuntimeError("C backend failed to solve the passive five-bar pair")
    return float(out[0]), float(out[1]), float(out[2])

def compute_balance_plane(anchors: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    anchors = np.ascontiguousarray(np.asarray(anchors, dtype=np.float64).reshape(4, 3))
    out = np.zeros(18, dtype=np.float64)

    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for balance plane solve: {_STATUS.message}")

    ok = lib.compute_balance_plane_c(
        anchors.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(scale),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not ok:
        return None
    return out[:12].reshape(4, 3), out[12:15].copy(), float(out[15]), float(out[16]), float(out[17])

def solve_robot_state(
    tree_steps: np.ndarray,
    wheel_right_chain: np.ndarray,
    branch_right_chain: np.ndarray,
    wheel_left_chain: np.ndarray,
    branch_left_chain: np.ndarray,
    angles: np.ndarray,
    dirty_right: bool,
    dirty_left: bool,
    right_passive_a: int,
    right_passive_b: int,
    left_passive_a: int,
    left_passive_b: int,
    right_loop_origin: np.ndarray,
    left_loop_origin: np.ndarray,
    support_origin: np.ndarray,
    requested_root_rotation: np.ndarray,
    scale: float,
    link_count: int,
    wheel_left_index: int,
    wheel_right_index: int,
    wheel_radius: float,
    balance_indices: np.ndarray,
    lower: float = -math.pi / 2.0,
    upper: float = math.pi / 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[np.ndarray, np.ndarray, float, float, float] | None]:
    tree_steps = _as_tree_steps(tree_steps)
    wheel_right_chain = _as_chain(wheel_right_chain)
    branch_right_chain = _as_chain(branch_right_chain)
    wheel_left_chain = _as_chain(wheel_left_chain)
    branch_left_chain = _as_chain(branch_left_chain)
    angles = np.ascontiguousarray(np.asarray(angles, dtype=np.float64))
    right_loop_origin = np.asarray(right_loop_origin, dtype=np.float64)
    left_loop_origin = np.asarray(left_loop_origin, dtype=np.float64)
    support_origin = np.asarray(support_origin, dtype=np.float64)
    requested_root_rotation = np.ascontiguousarray(np.asarray(requested_root_rotation, dtype=np.float64).reshape(9))
    balance_indices = np.ascontiguousarray(np.asarray(balance_indices, dtype=np.int32).reshape(4))
    origins = np.zeros((link_count, 3), dtype=np.float64)
    rotations = np.zeros((link_count, 3, 3), dtype=np.float64)
    closure = np.zeros(2, dtype=np.float64)
    balance = np.zeros(18, dtype=np.float64)
    balance_ok = ctypes.c_int()

    lib = _load_c_solver()
    if lib is None:
        raise RuntimeError(f"C backend is required for robot state solve: {_STATUS.message}")

    ok = lib.solve_robot_state_c(
        tree_steps.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(tree_steps)),
        wheel_right_chain.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(wheel_right_chain)),
        branch_right_chain.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(branch_right_chain)),
        wheel_left_chain.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(wheel_left_chain)),
        branch_left_chain.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(branch_left_chain)),
        angles.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_int(len(angles)),
        ctypes.c_int(1 if dirty_right else 0),
        ctypes.c_int(1 if dirty_left else 0),
        ctypes.c_int(right_passive_a),
        ctypes.c_int(right_passive_b),
        ctypes.c_int(left_passive_a),
        ctypes.c_int(left_passive_b),
        right_loop_origin.ctypes.data_as(DOUBLE_PTR),
        left_loop_origin.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_double(lower),
        ctypes.c_double(upper),
        support_origin.ctypes.data_as(DOUBLE_PTR),
        requested_root_rotation.ctypes.data_as(DOUBLE_PTR),
        ctypes.c_double(scale),
        ctypes.c_int(link_count),
        ctypes.c_int(wheel_left_index),
        ctypes.c_int(wheel_right_index),
        ctypes.c_double(wheel_radius),
        balance_indices.ctypes.data_as(INT_PTR),
        origins.ctypes.data_as(DOUBLE_PTR),
        rotations.ctypes.data_as(DOUBLE_PTR),
        closure.ctypes.data_as(DOUBLE_PTR),
        balance.ctypes.data_as(DOUBLE_PTR),
        ctypes.byref(balance_ok),
    )
    if not ok:
        raise RuntimeError("C backend failed to solve robot state")
    balance_result = None
    if balance_ok.value:
        balance_result = (
            balance[:12].reshape(4, 3),
            balance[12:15].copy(),
            float(balance[15]),
            float(balance[16]),
            float(balance[17]),
        )
    return angles, origins, rotations, (float(closure[0]), float(closure[1])), balance_result

def _as_chain(chain: np.ndarray) -> np.ndarray:
    arr = np.asarray(chain, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, STEP_WIDTH), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1, STEP_WIDTH), dtype=np.float64)

def _as_tree_steps(steps: np.ndarray) -> np.ndarray:
    arr = np.asarray(steps, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, TREE_STEP_WIDTH), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1, TREE_STEP_WIDTH), dtype=np.float64)

def _load_c_solver() -> ctypes.CDLL | None:
    global _LIB, _STATUS, _LOAD_ATTEMPTED
    if _LIB is not None:
        return _LIB
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True

    base_dir = Path(__file__).resolve().parent
    source = base_dir / "fivebar_solver.c"
    build_dir = base_dir / "build"
    if not source.exists():
        _STATUS = SolverStatus("unavailable", "fivebar_solver.c not found")
        return None
    library = _library_for_source(source, build_dir)

    if not library.exists():
        _compile_c_solver(source, library)

    if not library.exists():
        return None

    try:
        lib = ctypes.CDLL(str(library))
        lib.solve_passive_pair_c.argtypes = [
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            DOUBLE_PTR,
        ]
        lib.solve_passive_pair_c.restype = ctypes.c_int
        lib.compute_link_transforms_c.argtypes = [
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            DOUBLE_PTR,
            ctypes.c_double,
            ctypes.c_int,
            DOUBLE_PTR,
            DOUBLE_PTR,
        ]
        lib.compute_link_transforms_c.restype = ctypes.c_int
        lib.compute_supported_link_transforms_c.argtypes = [
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            DOUBLE_PTR,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            DOUBLE_PTR,
            DOUBLE_PTR,
        ]
        lib.compute_supported_link_transforms_c.restype = ctypes.c_int
        lib.load_stl_bounds_c.argtypes = [
            ctypes.c_char_p,
            DOUBLE_PTR,
            DOUBLE_PTR,
            INT_PTR,
        ]
        lib.load_stl_bounds_c.restype = ctypes.c_int
        lib.compute_balance_plane_c.argtypes = [
            DOUBLE_PTR,
            ctypes.c_double,
            DOUBLE_PTR,
        ]
        lib.compute_balance_plane_c.restype = ctypes.c_int
        lib.solve_robot_state_c.argtypes = [
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            DOUBLE_PTR,
            DOUBLE_PTR,
            ctypes.c_double,
            ctypes.c_double,
            DOUBLE_PTR,
            DOUBLE_PTR,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            INT_PTR,
            DOUBLE_PTR,
            DOUBLE_PTR,
            DOUBLE_PTR,
            DOUBLE_PTR,
            INT_PTR,
        ]
        lib.solve_robot_state_c.restype = ctypes.c_int
        _LIB = lib
        _STATUS = SolverStatus("C", str(library))
        return _LIB
    except OSError as exc:
        _STATUS = SolverStatus("unavailable", f"C solver load failed: {exc}")
        return None

def _compile_c_solver(source: Path, library: Path) -> None:
    global _STATUS
    library.parent.mkdir(exist_ok=True)
    commands = []
    if _is_windows():
        msvc_command = _msvc_build_command(source, library)
        if msvc_command:
            commands.append(msvc_command)
        commands.extend(
            [
                ["gcc", "-O3", "-shared", "-o", str(library), str(source)],
                ["clang", "-O3", "-shared", "-o", str(library), str(source)],
                ["cl", "/O2", "/LD", str(source), f"/Fe:{library}", f"/Fo:{library.with_suffix('.obj')}"],
            ]
        )
    else:
        commands.extend(
            [
                ["cc", "-O3", "-fPIC", "-shared", "-o", str(library), str(source), "-lm"],
                ["gcc", "-O3", "-fPIC", "-shared", "-o", str(library), str(source), "-lm"],
                ["clang", "-O3", "-fPIC", "-shared", "-o", str(library), str(source), "-lm"],
            ]
        )

    last_error = ""
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=source.parent,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                shell=isinstance(command, str),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = str(exc)
            continue
        if result.returncode == 0 and library.exists():
            _STATUS = SolverStatus("C", str(library))
            return
        last_error = (result.stderr or result.stdout).strip()
    _STATUS = SolverStatus("unavailable", f"C compiler unavailable: {last_error}")

def _is_windows() -> bool:
    return hasattr(ctypes, "WinDLL")

def _library_for_source(source: Path, build_dir: Path) -> Path:
    version = source.stat().st_mtime_ns
    if _is_windows():
        return build_dir / f"fivebar_solver_{version}.dll"
    return build_dir / f"libfivebar_solver_{version}.so"

def _msvc_build_command(source: Path, library: Path) -> str | None:
    candidates = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    vcvars = next((path for path in candidates if path.exists()), None)
    if vcvars is None:
        return None

    import_lib = library.with_name(f"{library.stem}_import.lib")
    return (
        f'call "{vcvars}" && '
        f'cl /O2 /LD "{source}" /Fe"{library}" /Fo"{library.with_suffix(".obj")}" '
        f'/link /implib:"{import_lib}"'
    )
