import torch
import time

size = 10000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.backends.mps.is_available():
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    # 热身:先跑一次,不计时,让 MPS 完成初始化开销
    _ = a_mps @ b_mps
    torch.mps.synchronize()

    # 正式计时
    start = time.time()
    c_mps = a_mps @ b_mps
    torch.mps.synchronize()
    mps_time = time.time() - start
    print(f"MPS (GPU): {mps_time:.3f}s")
    print(f"Speedup: {cpu_time / mps_time:.1f}x")
else:
    print("MPS not available on this machine.")