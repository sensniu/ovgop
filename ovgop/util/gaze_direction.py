"""Shared geometry for the eight gaze directions."""


DIRECTION_STEPS = (
    (1, 0),
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, -1),
    (-1, -1),
    (-1, 1),
    (1, 1),
)


def nearest_direction_angle(degree):
    angles = range(0, 360, 45)
    return min(angles, key=lambda angle: abs((degree - angle + 180) % 360 - 180))


def direction_pixels(center_x, center_y, width, height, direction_id):
    """Yield a ray from its center through the last pixel inside the image."""
    dx, dy = DIRECTION_STEPS[direction_id]
    x, y = int(center_x), int(center_y)
    while 0 <= x < width and 0 <= y < height:
        yield x, y
        x += dx
        y += dy
