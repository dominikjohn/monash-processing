import numpy as np
import pandas as pd


def create_2d_stencil_matrix(nx=5, ny=5):
    """Create the coefficient matrix for 5-point stencil on 5x5 grid"""
    n = nx * ny
    # Set dtype to object to allow string entries
    matrix = np.full((n, n), '.', dtype=object)

    # Fill matrix with coefficients
    for i in range(nx):
        for j in range(ny):
            current = i * ny + j

            # Center point
            matrix[current, current] = 'B'  # h²g - 2a - 2a'

            # Left neighbor
            if j > 0:
                matrix[current, current - 1] = 'A'  # hd/2 + a

            # Right neighbor
            if j < ny - 1:
                matrix[current, current + 1] = 'C'  # -hd/2 + a

            # Top neighbor
            if i > 0:
                matrix[current, current - ny] = 'D'  # he/2 + a'

            # Bottom neighbor
            if i < nx - 1:
                matrix[current, current + ny] = 'E'  # -he/2 + a'

    return matrix


# Create and display matrix
matrix = create_2d_stencil_matrix(5, 5)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.DataFrame(matrix)
print("Matrix for 5x5 grid (25x25 matrix):")
print("where:")
print("B = h²g - 2a - 2a'  (center)")
print("A = hd/2 + a        (left neighbor)")
print("C = -hd/2 + a       (right neighbor)")
print("D = he/2 + a'       (top neighbor)")
print("E = -he/2 + a'      (bottom neighbor)")
print("\n")
print(df)