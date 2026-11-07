import math


def quat_z(angle: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a rotation of `angle` about +z."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def quat_mul(a, b):
    """Quaternion product a·b: rotate by b first, then by a."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
