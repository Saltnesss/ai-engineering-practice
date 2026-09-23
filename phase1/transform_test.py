"""Phase 1 Lesson 3 - Matrix Transformations, from scratch."""
import math


def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]


def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]


def reflection_x():
    return [[1, 0], [0, -1]]


def reflection_y():
    return [[-1, 0], [0, 1]]


def mat_vec_mul(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector)))
            for i in range(len(matrix))]


def mat_mul(a, b):
    rows_a, cols_b, cols_a = len(a), len(b[0]), len(a[0])
    return [[sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
            for i in range(rows_a)]


def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)


def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        v = [1, 0] if abs(a - eigenvalue) < 1e-10 else [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


if __name__ == "__main__":
    print("--- Step 1: basic transformations ---")
    angle = math.pi / 4
    rotated = mat_vec_mul(rotation_2d(angle), [1.0, 0.0])
    print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")
    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")
    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")
    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")

    print("--- Step 2: composition ---")
    R = rotation_2d(math.pi / 2)
    S = scaling_2d(2, 0.5)
    rotate_then_scale = mat_mul(S, R)
    scale_then_rotate = mat_mul(R, S)
    r1 = mat_vec_mul(rotate_then_scale, [1.0, 0.0])
    r2 = mat_vec_mul(scale_then_rotate, [1.0, 0.0])
    print(f"Rotate 90 then scale: ({r1[0]:.2f}, {r1[1]:.2f})")
    print(f"Scale then rotate 90: ({r2[0]:.2f}, {r2[1]:.2f})")
    print(f"Same? {r1 == r2}")

    print("--- Step 3: eigenvalues from scratch ---")
    A = [[2, 1], [1, 2]]
    vals = eigenvalues_2x2(A)
    print(f"Matrix: {A}")
    print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")
    for val in vals:
        vec = eigenvector_2x2(A, val)
        result = mat_vec_mul(A, vec)
        scaled = [val * vec[0], val * vec[1]]
        print(f"  lambda={val:.1f}, v={[round(x, 4) for x in vec]}")
        print(f"    A@v = {[round(x, 4) for x in result]}")
        print(f"    l*v = {[round(x, 4) for x in scaled]}")
    print(f"rotation 90 deg: {eigenvalues_2x2([[0, -1], [1, 0]])}")

    print("--- Step 4: determinant ---")
    print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi / 4)):.4f}")
    print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
    print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
    print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")
    print(f"det(singular)    = {det_2x2([[1, 2], [2, 4]]):.1f}")
