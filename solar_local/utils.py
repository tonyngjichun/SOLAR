import cv2
import torch
import math
import numpy as np


def describe_opencv(model,
                    img,
                    kpts,
                    patch_size=32,
                    mag_factor=3,
                    use_gpu=True):
    """
        Rectifies patches around openCV keypoints, and returns patches tensor
    """
    patches = []
    for kp in kpts:
        x, y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / patch_size
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix(
            [[+s * cos, -s * sin, (-s * cos + s * sin) * patch_size / 2.0 + x],
             [+s * sin, +s * cos,
              (-s * sin - s * cos) * patch_size / 2.0 + y]])

        patch = cv2.warpAffine(
            img,
            M, (patch_size, patch_size),
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC +
            cv2.WARP_FILL_OUTLIERS)

        patches.append(patch)

    patches = torch.from_numpy(np.asarray(patches)).float()
    patches = torch.unsqueeze(patches, 1)
    if use_gpu:
        patches = patches.cuda()
    descrs = model(patches)
    return descrs.detach().cpu().numpy()
