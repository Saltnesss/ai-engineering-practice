"""Phase 1 Lesson 4 - Calculus for ML, Build It."""

# --- Step 1: numerical derivative -------------------------------------
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

print("Step 1: numerical vs analytical derivative of x^2")
for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}"
          f"   diff={numerical - analytical:+.3e}")


# --- Step 2: partial derivatives and gradients ------------------------
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

print("\nStep 2: gradient of x^2 + 3xy + y^2 at (1, 2)")
grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"  numerical  : {[f'{g:.6f}' for g in grad]}")
print(f"  analytical : [2x+3y, 3x+2y] = [{2*1+3*2}, {3*1+2*2}]")


# --- Step 3: gradient descent on f(x) = x^2 ---------------------------
print("\nStep 3: gradient descent on x^2, start x=5, lr=0.1")
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    if step % 4 == 0 or step == 19:
        print(f"  step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}   0.8^{step+1}*5={5*0.8**(step+1):8.4f}")


# --- Step 4: gradient descent in 2D -----------------------------------
def f_2d(point):
    x, y = point
    return x**2 + y**2

print("\nStep 4: 2D gradient descent on x^2 + y^2, start (4, 3), lr=0.1")
point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    if step % 5 == 0 or step == 29:
        print(f"  step {step:2d}  point=({point[0]:8.5f}, {point[1]:8.5f})"
              f"  f={f_2d(point):.6f}   y/x={point[1]/point[0]:.4f}")


# --- Step 5: numerical vs analytical on five functions ----------------
import math

test_functions = [
    ("x^2",    lambda x: x**2,        lambda x: 2*x),
    ("x^3",    lambda x: x**3,        lambda x: 3*x**2),
    ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
    ("e^x",    lambda x: math.exp(x), lambda x: math.exp(x)),
    ("1/x",    lambda x: 1/x,         lambda x: -1/x**2),
]

print("\nStep 5: numerical vs analytical at x = 2, h = 1e-7")
x = 2.0
eps = 2.220446049250313e-16
h = 1e-7
print(f"{'Function':<10} {'Numerical':>13} {'Analytical':>13} {'Error':>11} {'|f(2)|':>8} {'predicted':>11}")
for name, fn, dfn in test_functions:
    num = numerical_derivative(fn, x)
    ana = dfn(x)
    err = abs(num - ana)
    bound = abs(fn(x)) * eps / (2 * h)
    print(f"{name:<10} {num:13.8f} {ana:13.8f} {err:11.2e} {abs(fn(x)):8.3f} {bound:11.2e}")


# --- Step 6: Hessian by finite differences ----------------------------
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h)
           - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x**2 - y**2

def bowl(x, y):
    return x**2 + y**2

print("\nStep 6: Hessian by finite differences at (0, 0), h=1e-5")
print(f"  saddle x^2 - y^2 : {hessian_2d(saddle, 0.0, 0.0)}")
print(f"  bowl   x^2 + y^2 : {hessian_2d(bowl, 0.0, 0.0)}")

print("\n  h-sweep on the bowl, at (0,0) then at (3,4)  [truth = 2.0]")
print(f"  {'h':>8} {'fxx at (0,0)':>16} {'fxx at (3,4)':>16} {'predicted err':>15}")
for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    at00 = hessian_2d(bowl, 0.0, 0.0, h)[0][0]
    at34 = hessian_2d(bowl, 3.0, 4.0, h)[0][0]
    pred = abs(bowl(3.0, 4.0)) * 2.220446049250313e-16 / h**2
    print(f"  {h:8.0e} {at00:16.8f} {at34:16.8f} {pred:15.2e}")


# --- Step 7: Taylor approximation, order 1 vs order 2 -----------------
def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

sin_dd = lambda x: -math.sin(x)

print("\nStep 7: Taylor around x0 = 0.0  (sin(0)=0, cos(0)=1, -sin(0)=0)")
x0 = 0.0
print(f"  {'h':>5} {'sin(h)':>10} {'order1':>10} {'order2':>10} {'error':>10} {'h^3/6':>10}")
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, sin_dd, x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, sin_dd, x0, h, order=2)
    print(f"  {h:5.1f} {true_val:10.4f} {t1:10.4f} {t2:10.4f} {abs(true_val-t2):10.4f} {h**3/6:10.4f}")

print("\n  same expansion around x0 = 1.0, where sin'' is NOT zero")
x0 = 1.0
print(f"  {'h':>5} {'sin(1+h)':>10} {'order1':>10} {'order2':>10} {'err1':>10} {'err2':>10}")
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(x0 + h)
    t1 = taylor_approx(math.sin, math.cos, sin_dd, x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, sin_dd, x0, h, order=2)
    print(f"  {h:5.1f} {true_val:10.4f} {t1:10.4f} {t2:10.4f} "
          f"{abs(true_val-t1):10.4f} {abs(true_val-t2):10.4f}")


# --- Step 8: a full training loop, learning y = 2x + 1 ----------------
import random

random.seed(42)
w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

print(f"\nStep 8: training y = w*x + b to fit y = 2x + 1")
print(f"  initial w={w:.4f}  b={b:.4f}   (both random draws from N(0,1))")
print(f"  {'epoch':>6} {'w':>9} {'b':>9} {'dw':>10} {'db':>10} {'loss':>11}")
for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"  {epoch:6d} {w:9.4f} {b:9.4f} {dw:10.4f} {db:10.4f} {total_loss:11.6f}")

print(f"\n  Learned: y = {w:.4f}x + {b:.4f}")
print(f"  Actual:  y = 2x + 1")


# --- Exercise 1: second derivative by nesting numerical_derivative ----
def numerical_second_derivative(f, x, h=1e-5):
    df = lambda t: numerical_derivative(f, t, h)
    return numerical_derivative(df, x, h)

def direct_second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

print("\nExercise 1: second derivative of x^3 at x=2   (true value = 6x = 12)")
cube = lambda t: t**3
print(f"  {'h':>8} {'nested':>14} {'direct':>14} {'err nested':>12} {'err direct':>12}")
for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    n = numerical_second_derivative(cube, 2.0, h)
    d = direct_second_derivative(cube, 2.0, h)
    print(f"  {h:8.0e} {n:14.8f} {d:14.8f} {abs(n-12):12.2e} {abs(d-12):12.2e}")


# --- Exercise 2: gradient descent to (3, -1) --------------------------
def f_ex2(point):
    x, y = point
    return (x - 3)**2 + (y + 1)**2

print("\nExercise 2: minimise (x-3)^2 + (y+1)^2 from (0,0)   target (3, -1)")
point = [0.0, 0.0]
lr = 0.1
print(f"  {'step':>5} {'x':>10} {'y':>10} {'f':>12} {'0.8^n * dist':>14}")
for step in range(60):
    grad = numerical_gradient(f_ex2, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    if step % 10 == 0 or step == 59:
        import math as _m
        dist = _m.hypot(point[0] - 3, point[1] + 1)
        print(f"  {step:5d} {point[0]:10.6f} {point[1]:10.6f} {f_ex2(point):12.8f}"
              f" {_m.hypot(3,1) * 0.8**(step+1):14.6f}")
print(f"  final: ({point[0]:.6f}, {point[1]:.6f})")


# --- Exercise 3: momentum vs plain gradient descent -------------------
def f_ex3(x):
    return x**4 - 3*x**2

def df_ex3(x):
    return 4*x**3 - 6*x

def gd_plain(x0, lr, steps):
    x = x0
    traj = [x]
    for _ in range(steps):
        x -= lr * df_ex3(x)
        traj.append(x)
    return traj

def gd_momentum(x0, lr, beta, steps):
    x = x0
    v = 0.0
    traj = [x]
    for _ in range(steps):
        v = beta * v + df_ex3(x)
        x -= lr * v
        traj.append(x)
    return traj

TARGET = 1.5 ** 0.5
print(f"\nExercise 3: f(x) = x^4 - 3x^2, minima at +-{TARGET:.4f}, start x=2.0, lr=0.01")
p = gd_plain(2.0, 0.01, 200)
m = gd_momentum(2.0, 0.01, 0.9, 200)
print(f"  {'step':>5} {'plain GD':>12} {'momentum':>12} {'|err| plain':>13} {'|err| mom':>12}")
for s in [0, 1, 2, 5, 10, 20, 40, 80, 200]:
    print(f"  {s:5d} {p[s]:12.6f} {m[s]:12.6f} {abs(p[s]-TARGET):13.2e} {abs(m[s]-TARGET):12.2e}")

def first_within(traj, tol):
    for i, x in enumerate(traj):
        if abs(x - TARGET) < tol:
            return i
    return None

print(f"\n  steps to get within 1e-4 of the minimum:")
print(f"    plain GD : {first_within(p, 1e-4)}")
print(f"    momentum : {first_within(m, 1e-4)}")
print(f"  did momentum overshoot below {TARGET:.4f}?  {min(m) < TARGET}   (min reached {min(m):.6f})")
