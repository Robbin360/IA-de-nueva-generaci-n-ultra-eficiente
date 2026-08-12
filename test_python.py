import time
import math
import random

print("========================================")
print(" 🐍 PYTHON LLM INFERENCE DEMO")
print("========================================")

def rms_norm(x):
    mean_sq = sum(i*i for i in x) / len(x)
    rsqrt = 1.0 / math.sqrt(mean_sq + 1e-6)
    return [i * rsqrt for i in x]

dim = 256
vec = [random.uniform(-1, 1) for _ in range(dim)]

start = time.time()
norm_vec = rms_norm(vec)
print(f"RMSNorm en {dim} dimensiones ejecutado en {(time.time() - start)*1000:.4f} ms")
print("Python 3 estándar está disponible en este entorno.")
