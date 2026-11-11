"""Motion prompts shared by the image-to-video adapters."""

from __future__ import annotations

from shortgen.domain.models import CameraMove

_CAMERA_PHRASES: dict[CameraMove, str] = {
    CameraMove.STATIC: "the camera stays still",
    CameraMove.ZOOM_IN: "the camera slowly pushes in",
    CameraMove.ZOOM_OUT: "the camera slowly pulls back",
    CameraMove.PAN_LEFT: "the camera pans slowly to the left",
    CameraMove.PAN_RIGHT: "the camera pans slowly to the right",
    CameraMove.PAN_UP: "the camera tilts slowly upward",
    CameraMove.PAN_DOWN: "the camera tilts slowly downward",
}


def motion_prompt(prompt: str, camera: CameraMove) -> str:
    """Add the scene's camera move to its motion description."""
    return f"{prompt.strip().rstrip('.')}, {_CAMERA_PHRASES[camera]}"
