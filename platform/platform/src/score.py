import numpy as np

def soft_jackidx_score(pred : np.ndarray, real : np.ndarray) -> float:

    intersection = np.min(np.stack([pred,real], axis=0), axis=0).sum(-1)
    union = np.max(np.stack([pred,real], axis=0), axis=0).sum(-1)

    jacc_idx = intersection / (union + 1e-9)

    return jacc_idx.mean().item()