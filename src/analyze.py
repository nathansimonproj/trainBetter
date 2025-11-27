import numpy as np

def computeAngle(a, b, c) -> int:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    # avoid divide by zero
    if np.linalg.norm(ab) == 0 or np.linalg.norm(cb) == 0:
        return None

    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))