# Wheeled Biped Robot 3D Viewer

Interactive desktop viewer for a wheeled biped robot exported from URDF. The app runs in a local `pywebview` window: HTML/CSS/JS handles the UI and canvas, Python exposes the app API, and C does the kinematics work.

## Features

- URDF-based robot rendering from `robot_urdf/urdf/robot.SLDASM.urdf`
- One-time URDF/STL bounds loading for each robot link
- Stick/linkage rendering instead of per-frame mesh rendering
- Event-driven redraws with no background simulation loop
- Desktop HTML/CSS/JS UI served from local files, not a web server
- Per-side 5-bar linkage controls for pad, motor, calf, and wheel joints
- C/ctypes kinematics, passive calf-joint solver, and balance-plane angle solve
- Adjustable camera and model scale
- Linkage view anchors the wheel contact support, so the body/balance plane moves as joints change
- Passive solve is locked to a local branch window and continuity-biased, so it does not auto-flip to another closure branch

## Requirements

- Python 3.10+
- Python packages listed in `requirements.txt`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python robot_sim.py
```

## Project Structure

```text
robot_sim.py              Desktop app launcher
robot_engine.py           URDF parsing, scene preparation, and Python/C bridge
app_ui/                   Local HTML/CSS/JS app UI
fivebar_solver.py         ctypes wrapper for the C backend
fivebar_solver.c          C routines for FK, 5-bar closure, STL bounds, and plane solve
robot_urdf/               Minimal URDF and visual STL mesh assets
requirements.txt          Python dependencies
```

## C Solver

`fivebar_solver.py` builds a source-versioned library in `build/`, for example `fivebar_solver_<source_time>.dll` on Windows. If GCC/Clang is on `PATH`, or Visual Studio 2022 is installed in its default location, the app builds from `fivebar_solver.c`. FK, wheel support anchoring, STL bounds, five-bar solving, and balance-plane solve all run through this library. Versioned DLL names avoid Windows overwrite locks when an older viewer process is still open.

Manual Windows build with GCC:

```powershell
mkdir build
gcc -O3 -shared -o build\fivebar_solver.dll fivebar_solver.c
```

## Notes

- The app expects the URDF at `robot_urdf/urdf/robot.SLDASM.urdf`.
- Visual mesh bounds are loaded from `robot_urdf/meshes/*.STL` once for wheel size and linkage display metadata.
- The viewer keeps URDF joint origins, axes, and link hierarchy intact; display scale is applied uniformly.
- In linkage mode, the wheel contact midpoint is treated as the fixed ground support and the translucent blue top plane is attached to the upper thigh-joint geometry.
- The current focus is static linkage inspection.
- ROS/Gazebo export files, collision meshes, USD exports, and build intermediates are not part of the working app folder. `2d_view.py` is kept as a standalone tool.
- Cache files and logs are ignored by Git.
