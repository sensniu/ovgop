"""Gaze-specific training losses."""

import math

import torch


def gaze_box_energy_loss(boxes, heatmaps):
    """Encourage high average heatmap energy inside the target gaze box."""
    power = 0.0
    for box, heatmap in zip(boxes, heatmaps):
        xmin = math.floor((box[0] - box[2] / 2) * 64)
        ymin = math.floor((box[1] - box[3] / 2) * 64)
        xmax = math.ceil((box[0] + box[2] / 2) * 64)
        ymax = math.ceil((box[1] + box[3] / 2) * 64)
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1
        power = power + torch.sum(
            heatmap[ymin:min(ymax + 1, 64), xmin:min(xmax + 1, 64)]
        ) / (box_width * box_height)
    return 1 - power / len(heatmaps)
