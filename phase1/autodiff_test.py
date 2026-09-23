"""Phase 1 Lesson 5 — Chain Rule & Automatic Differentiation.
Build It steps 1-4: the Value class (micrograd-style autograd engine).
"""
import math


class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- step 2: ops that record how to send gradient backwards ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # --- step 4: everything else ---
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # --- step 3: the backward pass ---
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


if __name__ == "__main__":
    # sanity check: z = x*y + x,  with x=3, y=4
    # dz/dx = y + 1 = 5 ,  dz/dy = x = 3
    x = Value(3.0)
    y = Value(4.0)
    z = x * y + x
    z.backward()
    print("z      =", z)
    print("x.grad =", x.grad, "(expected 5.0)")
    print("y.grad =", y.grad, "(expected 3.0)")


# ---------------- Step 5: Neuron / Layer / MLP ----------------
import random


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def train_xor():
    """Step 5: train the MLP on XOR with pure-Python autograd."""
    random.seed(42)
    model = MLP([2, 4, 1])

    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]          # XOR, using -1/1 because the output is tanh

    for step in range(100):
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 20 == 0:
            print(f"step {step:3d}  loss = {loss.data:.4f}")

    print("\nPredictions after training:")
    for x, y in zip(xs, ys):
        print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")


# ---------------------------------------------------------------
# Step 6: gradient checking
# ---------------------------------------------------------------

def gradient_check(build_expr, x_val, h=1e-7):
    """Compare autodiff's gradient against a finite-difference estimate."""
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus  = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    rel  = diff / max(abs(autodiff_grad), abs(numerical_grad), 1e-12)
    return autodiff_grad, numerical_grad, diff, rel


def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()


if __name__ == "__main__":
    ad, num, diff, rel = gradient_check(expr, 0.5)
    print(f"Autodiff:   {ad:.10f}")
    print(f"Numerical:  {num:.10f}")
    print(f"Absolute:   {diff:.2e}")
    print(f"Relative:   {rel:.2e}")


# ---------------------------------------------------------------
# Step 7: verify against a hand-derived answer
# ---------------------------------------------------------------

def step7_manual_check():
    """y = relu(x1*x2 + 1). Hand-derived: dy/dx1 = x2, dy/dx2 = x1."""
    x1, x2 = Value(2.0), Value(3.0)
    y = (x1 * x2 + Value(1.0)).relu()
    y.backward()
    print(f"y = {y.data}   dy/dx1 = {x1.grad} (expect 3.0)   dy/dx2 = {x2.grad} (expect 2.0)")

    # dying ReLU: a negative pre-activation zeroes every gradient below it
    x1, x2 = Value(2.0), Value(-3.0)
    y = (x1 * x2 + Value(1.0)).relu()
    y.backward()
    print(f"y = {y.data}   dy/dx1 = {x1.grad}   dy/dx2 = {x2.grad}   <- relu shut off")


# ---------------------------------------------------------------
# Exercise 4: forward-mode autodiff with dual numbers
#
# A dual number is  a + a'·eps  with  eps**2 = 0.
#   .a  = the function value
#   .da = the derivative, carried along during the FORWARD pass
# No graph, no topological sort, no backward() -- that is the whole point.
# ---------------------------------------------------------------

class Dual:
    def __init__(self, a, da=0.0):
        self.a = a          # value
        self.da = da        # derivative

    def __repr__(self):
        return f"Dual(a={self.a:.6f}, da={self.da:.6f})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Dual) else Dual(x)

    def __add__(self, other):
        o = Dual._wrap(other)
        return Dual(self.a + o.a, self.da + o.da)

    def __mul__(self, other):
        o = Dual._wrap(other)
        return Dual(self.a * o.a, self.da * o.a + self.a * o.da)   # a'b + ab'

    def __neg__(self):
        return Dual(-self.a, -self.da)

    def __sub__(self, other):
        return self + (-Dual._wrap(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Dual._wrap(other) + (-self)

    def __pow__(self, n):
        return Dual(self.a ** n, n * self.a ** (n - 1) * self.da)  # chain rule

    def __truediv__(self, other):
        o = Dual._wrap(other)
        return Dual(self.a / o.a, (self.da * o.a - self.a * o.da) / (o.a ** 2))

    def exp(self):
        e = math.exp(self.a)
        return Dual(e, e * self.da)

    def log(self):
        return Dual(math.log(self.a), self.da / self.a)

    def tanh(self):
        t = math.tanh(self.a)
        return Dual(t, (1 - t ** 2) * self.da)

    def relu(self):
        return Dual(max(0.0, self.a), self.da if self.a > 0 else 0.0)


def forward_grad(f, x_val):
    """d f/dx at x_val, forward mode. Seed the INPUT with da=1.0."""
    out = f(Dual(x_val, 1.0))
    return out.a, out.da
