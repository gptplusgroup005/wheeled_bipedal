# Wheeled Biped Robot 3D Viewer

Interactive Python/Tkinter viewer for a wheeled biped robot exported from URDF. The app focuses on 3D joint display and provides controls for detailed 5-bar linkage pose inspection, body pose, camera view, mesh detail, and telemetry.

## Features

- URDF-based robot rendering from `robot_urdf/urdf/robot.SLDASM.urdf`
- STL mesh loading for each robot link
- Per-side 5-bar linkage controls for pad, motor, calf, and wheel joints
- Passive calf-joint auto-solver with a C/ctypes backend and Python fallback
- Adjustable body pose, camera, mesh scale, and mesh detail
- Optional STL fallback

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
fivebar_solver.py         ctypes wrapper and Python fallback for the solver
fivebar_solver.c          C backend for 5-bar loop-closure solving
robot_urdf/               URDF package and mesh assets
requirements.txt          Python dependencies
```

## C Solver

`fivebar_solver.py` automatically looks for `build/fivebar_solver.dll` on Windows or `build/libfivebar_solver.so` on Linux/macOS. If a C compiler is available in `PATH`, the app will build it from `fivebar_solver.c`; otherwise it keeps running with the Python fallback.

Manual Windows build with GCC:

```powershell
mkdir build
gcc -O3 -shared -o build\fivebar_solver.dll fivebar_solver.c
```

## Notes

- The app expects the URDF at `robot_urdf/urdf/robot.SLDASM.urdf`.
- Visual meshes are loaded from `robot_urdf/meshes/*.STL`.
- The viewer keeps URDF joint origins, axes, and link hierarchy intact; display scale is applied uniformly to the whole model.
- The current focus is static linkage inspection. Map, obstacle, ground grid, and drive simulation logic have been removed for now.
- Generated cache files and export logs are intentionally ignored by Git.
