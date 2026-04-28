# Wheeled Biped Robot 3D Viewer

Interactive Python/Tkinter viewer for a wheeled biped robot exported from URDF. The app renders the robot in 3D, shows a terrain grid, and provides controls for differential wheel motion, body pose, camera view, mesh detail, and telemetry.

## Features

- URDF-based robot rendering from `robot_urdf/urdf/robot.SLDASM.urdf`
- STL mesh loading for each robot link
- Differential wheel drive simulation
- Trail/path visualization
- Adjustable wheel radius, wheel track, body pose, terrain size, and camera
- Optional STL fallback and simple wheel-biped placeholder

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
robot_urdf/               URDF package and mesh assets
requirements.txt          Python dependencies
```

## Notes

- The app expects the URDF at `robot_urdf/urdf/robot.SLDASM.urdf`.
- Visual meshes are loaded from `robot_urdf/meshes/*.STL`.
- Generated cache files and export logs are intentionally ignored by Git.
