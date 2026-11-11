"""Frame-aware sensor queue reads, independent of the CARLA runtime."""
import queue
import time


def image_for_frame(images, frame, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            image = images.get(timeout=remaining)
        except queue.Empty:
            return None
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"Sensor advanced to {image.frame} before requested frame {frame}; another client may be ticking")
