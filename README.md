# Wheeled Biped Robot 3D Viewer

Interactive Python/Tkinter viewer for a wheeled biped robot exported from URDF. The app reads the URDF once at startup, extracts joint geometry and link dimensions, then renders a lightweight CAD-style solid model for responsive 3D joint display.

## Features

- URDF-based robot rendering from `robot_urdf/urdf/robot.SLDASM.urdf`
- One-time URDF/STL bounds loading for each robot link
- Lightweight SolidWorks-style primitive rendering instead of per-frame mesh rendering
- Event-driven redraws with no background simulation loop
- Per-side 5-bar linkage controls for pad, motor, calf, and wheel joints
- C/ctypes kinematics and passive calf-joint solver; Python is used only for UI/display
- Adjustable camera and model scale
- Linkage view anchors the wheel contact support, so the body/balance plane moves as joints change
- Passive solve is locked to a local branch window and continuity-biased, so it does not auto-flip to another closure branch

## Requirements

- Python 3.10+
- Tkinter, usually included with Python on Windows
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
robot_sim.py              Main application
fivebar_solver.py         ctypes wrapper for C kinematics and solver
fivebar_solver.c          C backend for FK and 5-bar loop-closure solving
robot_urdf/               URDF package and mesh assets
requirements.txt          Python dependencies
```

## C Solver

`fivebar_solver.py` automatically builds a source-versioned library in `build/`, for example `fivebar_solver_<source_time>.dll` on Windows. If a C compiler is available in `PATH`, or Visual Studio 2022 is installed in its default location, the app will build it from `fivebar_solver.c`. The viewer requires this C backend for FK, supported-wheel anchoring, and five-bar solving; Python is reserved for UI and rendering. Versioned DLL names avoid Windows overwrite locks when an older viewer process is still open.

Manual Windows build with GCC:

```powershell
mkdir build
gcc -O3 -shared -o build\fivebar_solver.dll fivebar_solver.c
```

## Notes

- The app expects the URDF at `robot_urdf/urdf/robot.SLDASM.urdf`.
- Visual mesh bounds are loaded from `robot_urdf/meshes/*.STL` once and converted into simple boxes/cylinders for display.
- The viewer keeps URDF joint origins, axes, and link hierarchy intact; display scale is applied uniformly to the lightweight model.
- In linkage mode, the wheel contact midpoint is treated as the fixed ground support and the translucent blue top plane is attached to the upper thigh-joint geometry.
- The current focus is static linkage inspection. Map, obstacle, ground grid, and drive simulation logic have been removed for now.
- Generated cache files and export logs are intentionally ignored by Git.
