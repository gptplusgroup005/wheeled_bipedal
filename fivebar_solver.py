from __future__ import annotations

import ctypes
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STEP_WIDTH = 16


@dataclass(frozen=True)
class SolverStatus:
    backend: str
    message: str


_STATUS = SolverStatus("python", "C solver not loaded")
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
    if lib is not None:
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
        if ok:
            return float(out[0]), float(out[1]), float(out[2])

    return _solve_python(
        wheel_chain,
        branch_chain,
        angles,
        passive_a_index,
        passive_b_index,
        loop_origin,
        initial_a,
        initial_b,
        lower,
        upper,
    )


def _as_chain(chain: np.ndarray) -> np.ndarray:
    arr = np.asarray(chain, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, STEP_WIDTH), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1, STEP_WIDTH), dtype=np.float64)


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
    library = build_dir / ("fivebar_solver.dll" if _is_windows() else "libfivebar_solver.so")
    if not source.exists():
        _STATUS = SolverStatus("python", "fivebar_solver.c not found")
        return None

    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        _compile_c_solver(source, library)

    if not library.exists():
        return None

    try:
        lib = ctypes.CDLL(str(library))
        lib.solve_passive_pair_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.solve_passive_pair_c.restype = ctypes.c_int
        _LIB = lib
        _STATUS = SolverStatus("C", str(library))
        return _LIB
    except OSError as exc:
        _STATUS = SolverStatus("python", f"C solver load failed: {exc}")
        return None


def _compile_c_solver(source: Path, library: Path) -> None:
    global _STATUS
    library.parent.mkdir(exist_ok=True)
    commands = []
    if _is_windows():
        commands.extend(
            [
                ["gcc", "-O3", "-shared", "-o", str(library), str(source)],
                ["clang", "-O3", "-shared", "-o", str(library), str(source)],
                ["cl", "/O2", "/LD", str(source), f"/Fe:{library}"],
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
            result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = str(exc)
            continue
        if result.returncode == 0 and library.exists():
            _STATUS = SolverStatus("C", str(library))
            return
        last_error = (result.stderr or result.stdout).strip()
    _STATUS = SolverStatus("python", f"C compiler unavailable: {last_error}")


def _is_windows() -> bool:
    return hasattr(ctypes, "WinDLL")


def _solve_python(
    wheel_chain: np.ndarray,
    branch_chain: np.ndarray,
    angles: np.ndarray,
    passive_a_index: int,
    passive_b_index: int,
    loop_origin: np.ndarray,
    initial_a: float,
    initial_b: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    best_a = initial_a
    best_b = initial_b
    best_error = math.inf
    center_a = initial_a
    center_b = initial_b
    search_radius = (upper - lower) / 2.0

    for samples in (13, 11, 9):
        a_values = np.linspace(max(lower, center_a - search_radius), min(upper, center_a + search_radius), samples)
        b_values = np.linspace(max(lower, center_b - search_radius), min(upper, center_b + search_radius), samples)
        for angle_a in a_values:
            for angle_b in b_values:
                error = _closure_error(wheel_chain, branch_chain, angles, passive_a_index, passive_b_index, loop_origin, float(angle_a), float(angle_b))
                if error < best_error:
                    best_error = error
                    best_a = float(angle_a)
                    best_b = float(angle_b)
        center_a = best_a
        center_b = best_b
        search_radius *= 0.28

    return best_a, best_b, best_error


def _closure_error(
    wheel_chain: np.ndarray,
    branch_chain: np.ndarray,
    angles: np.ndarray,
    passive_a_index: int,
    passive_b_index: int,
    loop_origin: np.ndarray,
    candidate_a: float,
    candidate_b: float,
) -> float:
    wheel_origin, _wheel_rotation = _fk(wheel_chain, angles, passive_a_index, passive_b_index, candidate_a, candidate_b)
    branch_origin, branch_rotation = _fk(branch_chain, angles, passive_a_index, passive_b_index, candidate_a, candidate_b)
    branch_tip = branch_origin + branch_rotation @ loop_origin
    return float(np.linalg.norm(wheel_origin - branch_tip))


def _fk(
    chain: np.ndarray,
    angles: np.ndarray,
    passive_a_index: int,
    passive_b_index: int,
    candidate_a: float,
    candidate_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    for step in chain:
        local_origin = step[:3]
        origin_rotation = step[3:12].reshape(3, 3)
        axis = step[12:15]
        angle_index = int(step[15])
        origin = origin + rotation @ local_origin
        rotation = rotation @ origin_rotation
        if angle_index >= 0:
            if angle_index == passive_a_index:
                angle = candidate_a
            elif angle_index == passive_b_index:
                angle = candidate_b
            else:
                angle = float(angles[angle_index])
            rotation = rotation @ _axis_angle_matrix(axis, angle)
    return origin, rotation


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
