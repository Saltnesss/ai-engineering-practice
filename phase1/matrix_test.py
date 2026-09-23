"""Phase 1 · Lesson 2 exercises — Vectors, Matrices & Operations.

The Matrix class is the one built in the lesson's "Build It" section.
"""


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    def minor(self, i, j):
        """The matrix left after deleting row i and column j."""
        return Matrix([
            [self.data[r][c] for c in range(self.cols) if c != j]
            for r in range(self.rows) if r != i
        ])

    def cofactor(self, i, j):
        """Signed determinant of the minor at (i, j)."""
        return ((-1) ** (i + j)) * self.minor(i, j).determinant()

    def adjugate(self):
        """Transpose of the cofactor matrix -- note the swapped indices."""
        return Matrix([
            [self.cofactor(j, i) for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return self.adjugate().scalar_multiply(1 / det)

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])


# --- Exercise 1: verify A @ A⁻¹ = I ---------------------------------------

tests = {
    "A": Matrix([[1, 2], [3, 4]]),
    "B": Matrix([[3, 7], [1, 5]]),
    "C": Matrix([[2, 1], [1, 3]]),
}

for name, M in tests.items():
    inv = M.inverse_2x2()
    product = M.matmul(inv)
    print(f"{name} = {M.data}")
    print(f"  det({name})    = {M.determinant()}")
    print(f"  {name}⁻¹       = {inv.data}")
    print(f"  {name} @ {name}⁻¹ = {product.data}")
    print(f"  exactly I?    = {product.data == Matrix.identity(2).data}")
    print()


# --- Exercise 2: 3x3 inverse via the adjugate method ----------------------

M = Matrix([[2, -1, 0],
            [-1, 2, -1],
            [0, -1, 2]])

print("M =", M.data)
print("det(M) =", M.determinant())

inv = M.inverse()
print("M-inverse =")
for row in inv.data:
    print("   ", row)

print("M @ M-inverse =")
for row in M.matmul(inv).data:
    print("   ", row)


# --- Exercise 2b: compare against NumPy ------------------------------------

import numpy as np

M_np = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
np_inv = np.linalg.inv(M_np)
ours = np.array(inv.data)

print("\nNumPy inverse:\n", np_inv)
print("max absolute difference:", np.abs(ours - np_inv).max())
print("element-for-element identical? ", np.array_equal(ours, np_inv))
print("close enough?                  ", np.allclose(ours, np_inv))


# --- Exercise 3: two-layer network, Matrix class only (no NumPy) -----------

import random
random.seed(0)


def randn_matrix(rows, cols):
    """A rows x cols matrix of small random numbers in [-1, 1)."""
    return Matrix([[random.uniform(-1, 1) for _ in range(cols)]
                   for _ in range(rows)])


def relu(m):
    """Element-wise max(0, x) -- the nonlinearity between the two layers."""
    return Matrix([[max(0.0, v) for v in row] for row in m.data])


x = Matrix([[0.5], [0.8], [0.2]])
W1, b1 = randn_matrix(4, 3), randn_matrix(4, 1)
W2, b2 = randn_matrix(2, 4), randn_matrix(2, 1)

z1 = W1.matmul(x) + b1
h = relu(z1)
z2 = W2.matmul(h) + b2
y = z2

print("\n--- two-layer forward pass: 3 -> 4 -> 2 ---")
for name, m in [("x ", x), ("W1", W1), ("b1", b1), ("z1", z1), ("h ", h),
                ("W2", W2), ("b2", b2), ("y ", y)]:
    print(f"  {name} shape {m.shape}")

print("\n  z1 (pre-activation):", [round(v[0], 4) for v in z1.data])
print("  h  (after ReLU)   :", [round(v[0], 4) for v in h.data])
print("  y  (output)       :", [round(v[0], 4) for v in y.data])

assert x.shape == (3, 1) and h.shape == (4, 1) and y.shape == (2, 1)
print("\n  all shapes correct")


# --- Why the nonlinearity matters: two linear layers collapse into one -----

W_eff = W2.matmul(W1)                 # (2,4) @ (4,3) -> (2,3)
b_eff = W2.matmul(b1) + b2            # (2,4) @ (4,1) -> (2,1)
y_collapsed = W_eff.matmul(x) + b_eff

print("\n--- collapsing the two layers into one ---")
print("  W_eff shape:", W_eff.shape, "  b_eff shape:", b_eff.shape)
print("  two-layer output :", [round(v[0], 10) for v in y.data])
print("  ONE-layer output :", [round(v[0], 10) for v in y_collapsed.data])
print("  identical?", [round(v[0], 10) for v in y.data]
      == [round(v[0], 10) for v in y_collapsed.data])

# Now force a negative pre-activation so ReLU actually clips something.
x_neg = Matrix([[-2.0], [3.0], [-1.5]])
z1n = W1.matmul(x_neg) + b1
hn = relu(z1n)
print("\n--- same network, an input where ReLU actually fires ---")
print("  z1:", [round(v[0], 4) for v in z1n.data])
print("  h :", [round(v[0], 4) for v in hn.data], " <- clipped entries")
y_real = W2.matmul(hn) + b2
y_lin = W_eff.matmul(x_neg) + b_eff
print("  with ReLU   :", [round(v[0], 4) for v in y_real.data])
print("  without it  :", [round(v[0], 4) for v in y_lin.data], " <- now they diverge")
