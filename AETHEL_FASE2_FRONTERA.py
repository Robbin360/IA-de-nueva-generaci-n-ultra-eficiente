import os
# Instalamos las librerías necesarias silenciosamente
os.system("pip install -q tiktoken datasets")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import tiktoken
from datasets import load_dataset

print("===================================================================")
print(" 🚀 AETHEL-QUANTUM FASE 2: ENTRENAMIENTO FRONTERA (TOKENS)")
print("===================================================================\n")

# 1. CONFIGURACIÓN OPTIMIZADA PARA NO CRASHEAR LA GPU T4 (15GB)
class AethelConfig:
    vocab_size = 50304  # Vocabulario de GPT-2/4 (múltiplo de 64 para eficiencia CUDA)
    dim = 384           # Aumentamos la inteligencia (ancho de red)
    n_layers = 6        # 6 capas de profundidad
    n_heads = 6         # 6 cabezas de atención
    n_experts = 4       # 4 Expertos MoE
    active_experts = 2  # 2 Expertos activos por token
    block_size = 256    # Ventana de contexto (tokens)
    batch_size = 16     # Lote pequeño para no saturar la VRAM
    grad_accum_steps = 4 # TRUCO PRO: Acumula gradientes para simular un batch_size de 64 sin gastar RAM
    learning_rate = 4e-4
    max_iters = 15000   # Iteraciones (Puede tardar unas 2-3 horas)

config = AethelConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Dispositivo de computo: {device.upper()}")

# --- ARQUITECTURA DEL NÚCLEO AETHEL ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class AethelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.n_head = config.n_heads
        self.dim = config.dim
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class AethelMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.dim, config.n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.dim * 4),
                nn.GELU(),
                nn.Linear(config.dim * 4, config.dim)
            ) for _ in range(config.n_experts)
        ])
        self.k = config.active_experts

    def forward(self, x):
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.k, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.k):
            expert_idx = top_indices[..., i]
            weight = top_weights[..., i].unsqueeze(-1)
            expert_out = sum(
                (expert_idx == j).unsqueeze(-1) * self.experts[j](x)
                for j in range(len(self.experts))
            )
            out += weight * expert_out
        return out

class AethelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AethelAttention(config)
        self.moe = AethelMoE(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class AethelQuantumModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.block_size, config.dim)
        self.layers = nn.ModuleList([AethelBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Compartir pesos entre embedding y salida (Ahorra un 30% de parámetros)
        self.token_emb.weight = self.output.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# --- PREPARACIÓN DE DATOS MASIVOS (MILLONES DE TOKENS) ---
print("[*] Iniciando Tokenizador Avanzado (BPE)...")
enc = tiktoken.get_encoding("gpt2")

print("[*] Descargando dataset TinyStories (Alta calidad de razonamiento)...")
# Usamos un streaming dataset para no colapsar la RAM normal
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Creamos un buffer de tokens
token_buffer = []
print("[*] Pre-procesando el primer millón de tokens (esto toma un minuto)...")
dataset_iter = iter(dataset)
while len(token_buffer) < 1_000_000:
    example = next(dataset_iter)
    tokens = enc.encode(example['text'])
    token_buffer.extend(tokens)
    token_buffer.append(enc.eot_token) # Fin de texto

data_tensor = torch.tensor(token_buffer, dtype=torch.long)
print(f"[+] Dataset listo. Total tokens cargados en RAM: {len(data_tensor):,}")

def get_batch():
    ix = torch.randint(len(data_tensor) - config.block_size, (config.batch_size,))
    x = torch.stack([data_tensor[i:i+config.block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- ENTRENAMIENTO DE FRONTERA ---
torch.cuda.empty_cache() # Limpiamos memoria VRAM por si acaso
model = AethelQuantumModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

print(f"\n[*] MODELO AETHEL-QUANTUM (SUBWORDS) EN VRAM.")
print(f"[*] Parámetros entrenables: {sum(p.numel() for p in model.parameters()):,}")
print(f"[*] Iniciando entrenamiento masivo ({config.max_iters} iteraciones).")
print("[*] Usando Accumulation Steps para maximizar el hardware...\n")

start_time = time.time()
for iter_num in range(config.max_iters):
    # Gradient Accumulation (Simula tener 4 GPUs juntas)
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(config.grad_accum_steps):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        loss = loss / config.grad_accum_steps
        loss.backward()
        loss_accum += loss.item()
        
    optimizer.step()
    
    if iter_num % 100 == 0:
        elapsed = (time.time() - start_time) / 60
        print(f"Paso {iter_num:5d}/{config.max_iters} | Pérdida (Loss): {loss_accum:.4f} | Tiempo: {elapsed:.2f} mins")
        
    # Guardado de seguridad cada 5000 iteraciones
    if iter_num > 0 and iter_num % 5000 == 0:
        torch.save(model.state_dict(), f'aethel_frontier_ckpt_{iter_num}.pth')
        print(f"[!] Checkpoint guardado: aethel_frontier_ckpt_{iter_num}.pth")

print("\n===================================================================")
print(" ✅ ENTRENAMIENTO FRONTERA COMPLETADO EXITOSAMENTE.")
print("===================================================================")
torch.save(model.state_dict(), 'aethel_frontier_FINAL.pth')
print("[*] Pesos finales guardados en 'aethel_frontier_FINAL.pth'")

# PRUEBA DE GENERACIÓN REAL
print("\n[*] El modelo reflexiona y escribe:")
model.eval()
context = torch.tensor([enc.encode("Once upon a time, there was a little girl named")], dtype=torch.long, device=device)
with torch.no_grad():
    for _ in range(100):
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_idx), dim=1)
        if context.size(1) > config.block_size:
            context = context[:, -config.block_size:]

print("\n" + "-"*50)
print(enc.decode(context[0].tolist()))
print("-"*50)
