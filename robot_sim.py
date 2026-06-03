from __future__ import annotations

from pathlib import Path

import webview

from robot_engine import RobotEngine

class RobotApi:
    def __init__(self) -> None:
        self.engine = RobotEngine(Path(__file__).resolve().parent)

    def get_scene(self) -> dict:
        return self.engine.scene()

    def update_angles(self, values: dict[str, float]) -> dict:
        return self.engine.update_angles(values)

    def reset_angles(self) -> dict:
        return self.engine.reset_angles()

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "app_ui" / "index.html"
    api = RobotApi()
    webview.create_window(
        "Robot Viewer",
        html_path.as_uri(),
        js_api=api,
        width=1360,
        height=840,
        min_size=(1120, 720),
    )
    webview.start()

if __name__ == "__main__":
    main()
