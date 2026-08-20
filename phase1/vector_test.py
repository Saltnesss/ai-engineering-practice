import random
import math


class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        cos_theta = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_radians = math.acos(cos_theta)
        return math.degrees(theta_radians)

    def __repr__(self):
        return f"Vector({self.components})"


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def rank(self):
        rows = [row[:] for row in self.rows]
        num_rows, num_cols = self.shape
        rank = 0
        for col in range(num_cols):
            pivot = None
            for row in range(rank, num_rows):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            scale = rows[rank][col]
            rows[rank] = [x / scale for x in rows[rank]]
            for row in range(num_rows):
                if row != rank and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [
                        rows[row][j] - factor * rows[rank][j] for j in range(num_cols)
                    ]
            rank += 1
        return rank


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


# ==================== 练习 3 ====================
# 用余弦相似度,在 5 个 50 维随机向量里找最相似的一对

print("========== 练习 3 ==========")
random.seed(42)
vectors = [Vector([random.gauss(0, 1) for _ in range(50)]) for _ in range(5)]

best_pair = None
best_similarity = -2  # 余弦相似度范围是 [-1,1],用 -2 保证第一次比较一定能刷新这个值

for i in range(5):
    for j in range(i + 1, 5):
        sim = vectors[i].cosine_similarity(vectors[j])
        print(f"vector {i} vs vector {j}: {sim:.4f}")
        if sim > best_similarity:
            best_similarity = sim
            best_pair = (i, j)

print(
    f"\n最相似的一对: vector {best_pair[0]} 和 vector {best_pair[1]}, 相似度 = {best_similarity:.4f}"
)


# ==================== 练习 4 ====================
# 验证 Gram-Schmidt 输出的向量组,是否真的正交归一

print("\n========== 练习 4 ==========")
v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])

print("检查模长:")
for i, u in enumerate(basis):
    mag = u.magnitude()
    is_unit = abs(mag - 1.0) < 1e-10
    print(f"  |u{i + 1}| = {mag:.10f}   是否为单位向量: {is_unit}")

print("\n检查两两正交:")
n = len(basis)
for i in range(n):
    for j in range(i + 1, n):
        dot_product = basis[i].dot(basis[j])
        is_orthogonal = abs(dot_product) < 1e-10
        print(f"  u{i + 1} · u{j + 1} = {dot_product:.10f}   是否正交: {is_orthogonal}")

# ==================== 练习 5 ====================
# 构造一个秩为 2 的 3x3 矩阵,验证并解释几何含义

print("\n========== 练习 5 ==========")
A = Matrix([[1, 0, 2], [0, 1, 3], [0, 0, 0]])
print(f"矩阵 A = {A.rows}")
print(f"A 的秩 = {A.rank()}")


# ==================== 练习 6 ====================
# 把 [1,2,3] 投影到 [1,1,1] 上,解释几何含义

print("\n========== 练习 6 ==========")
a = Vector([1, 2, 3])
b = Vector([1, 1, 1])
result = project(a, b)
print(f"a = {a}")
print(f"b = {b}")
print(f"proj_b(a) = {result}")
