import numpy as np

def se3_logmap(T: np.ndarray) -> np.ndarray:
    """
    Compute the logarithmic map of an SE(3) transformation matrix.
    
    Args:
        T: 4x4 SE(3) transformation matrix [R | t; 0 0 0 1]
        
    Returns:
        6D twist vector [omega, v] where omega is the rotation (axis * angle)
        and v is the linear velocity component.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute rotation angle from trace
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity rotation: omega ≈ 0, V ≈ I
        omega = np.zeros(3)
        v = t
    elif abs(theta - np.pi) < 1e-6:
        # Near 180 degree rotation: extract axis from R
        # Find the column of R + I with largest norm
        RpI = R + np.eye(3)
        col_norms = np.linalg.norm(RpI, axis=0)
        k = np.argmax(col_norms)
        axis = RpI[:, k] / col_norms[k]
        omega = axis * theta

        # V inverse for theta ≈ pi
        omega_hat = skew(omega)
        V_inv = np.eye(3) - 0.5 * omega_hat + (1 / theta**2) * (1 - theta / (2 * np.tan(theta / 2))) * omega_hat @ omega_hat
        v = V_inv @ t
    else:
        # General case
        omega_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
        omega = unskew(omega_hat)
        
        # Compute V inverse
        half_theta = theta / 2
        V_inv = (np.eye(3) 
                 - 0.5 * omega_hat 
                 + (1 / theta**2) * (1 - half_theta / np.tan(half_theta)) * omega_hat @ omega_hat)
        v = V_inv @ t
    
    return np.concatenate([omega, v])


def skew(v: np.ndarray) -> np.ndarray:
    """Convert 3-vector to skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def unskew(M: np.ndarray) -> np.ndarray:
    """Extract 3-vector from skew-symmetric matrix."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])