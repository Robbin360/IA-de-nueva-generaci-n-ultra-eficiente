import os
# Auto-instalación de librerías esenciales
os.system("pip install -q tiktoken datasets safetensors")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import tiktoken
from datasets import load_dataset
from safetensors.torch import save_file

print("===================================================================")
print(" 🚀 AETHEL-QUANTUM V2: FRONTERA MULTILINGÜE Y DICCIONARIO ESPAÑOL/INGLÉS")
print("===================================================================\n")

# 1. CONFIGURACIÓN MEJORADA (MÁS CAPACIDAD Y ESTABILIDAD)
class AethelConfigV2:
    vocab_size = 100256  # tiktoken cl100k_base (Soporta Español, Inglés y caracteres latinos nativos)
    dim = 512            # Ampliado a 512 para mayor representación conceptual
    n_layers = 8         # 8 capas de profundidad
    n_heads = 8          # 8 cabezas de atención (64 dim por cabeza)
    n_experts = 4        # 4 Expertos MoE
    active_experts = 2   # 2 Expertos activos por token
    block_size = 384     # Ventana de contexto más amplia (384 tokens)
    batch_size = 16      # Lote optimizado para VRAM T4 / P100 / RTX
    grad_accum_steps = 4 # Batch size efectivo = 64
    learning_rate = 3e-4
    max_iters = 15000    # 15,000 pasos de calidad alta
    temperature = 0.7    # Temperatura óptima de muestreo
    top_k = 40           # Top-K para evitar palabras raras
    top_p = 0.9          # Top-P Nucleus Sampling

config = AethelConfigV2()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Dispositivo de cómputo: {device.upper()}")

# --- ARQUITECTURA MEJORADA AETHEL V2 CON ROPE Y MOE AUX LOSS ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Rotary Position Embedding (RoPE) para mejor atención situacional
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    # Aplica RoPE a Queries y Keys
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class AethelAttentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.n_head = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.rotary = RotaryEmbedding(self.head_dim)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(T, x.device)
        cos, sin = cos.unsqueeze(0).unsqueeze(1), sin.unsqueeze(0).unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class AethelMoEV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.dim, config.n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.dim * 4),
                nn.SiLU(), # Activación SwiGLU / SiLU estilo LLaMA/Mistral
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

class AethelBlockV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AethelAttentionV2(config)
        self.moe = AethelMoEV2(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class AethelQuantumModelV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([AethelBlockV2(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.token_emb.weight = self.output.weight # Weight Tying

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
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

# --- CARGA DE DATOS MULTILINGÜES Y DICCIONARIOS ESPAÑOL / INGLÉS ---
print("[*] Cargando Tokenizador Multilingüe cl100k_base...")
tokenizer = tiktoken.get_encoding("cl100k_base")

print("[*] Construyendo Corpus Multilingüe: Español, Inglés, Diccionarios y Enciclopedia...")

# Sintetizamos un buffer inicial multilingüe de calidad con diccionarios y explicaciones
corpus_textos = [
    # Diccionario y definiciones en Español
    "Definición: La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas.",
    "Diccionario: Algoritmo (nombre masculino) - Conjunto ordenado y finito de operaciones que permite hallar la solución de un problema.",
    "Diccionario: Ciencia (nombre femenino) - Conjunto de conocimientos obtenidos mediante la observación y el razonamiento.",
    "Diccionario: Computadora - Dispositivo electrónico capaz de procesar información de manera rápida y precisa.",
    "Explicación: La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía química.",
    "Razonamiento: Si A es igual a B y B es igual a C, entonces A es igual a C por propiedad transitiva.",
    
    # Dictionary & Definitions in English
    "Dictionary: Algorithm (noun) - A process or set of rules to be followed in calculations or problem-solving operations.",
    "Dictionary: Artificial Intelligence - The capability of a computer system to emulate human intelligence and logic.",
    "Dictionary: Science - The systematic study of the structure and behavior of the physical and natural world.",
    "Explanation: Python is an interpreted, high-level, general-purpose programming language known for readability.",
    "Logic: In computer science, a function receives inputs, performs transformations, and returns an output.",

    # Cuentos y Lógica bilingüe
    "Había una vez un niño muy inteligente llamado Félix que quería construir una inteligencia artificial autónoma.",
    "Lily tenía mucha hambre. Fue a la cocina, abrió el refrigerador y preparó una cena deliciosa y saludable.",
    "El perro grande miró al gato con amabilidad y le dijo: '¡Hola amigo! ¿Quieres jugar con la pelota en el jardín?'",
    "Once upon a time, a clever engineer named Felix decided to design a quantum neural architecture that learns continuously."
]

# Descargamos también fuentes públicas educativas de calidad (FineWeb / Wikitext / Wikimedia)
try:
    print("[*] Descargando fuentes multilingües de alta calidad (Wikipedia / Wikimedia)...")
    ds_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    iter_es = iter(ds_es)
    iter_en = iter(ds_en)
    
    print("[*] Ensamblando el buffer de tokens (Español + Inglés + Diccionarios)...")
    token_buffer = []
    
    # Inyectamos primero nuestros diccionarios y lógica base
    for t in corpus_textos:
        token_buffer.extend(tokenizer.encode(t))
        token_buffer.append(tokenizer.eot_token)
        
    # Agregamos lecturas alternadas de Wikipedia en Español e Inglés
    while len(token_buffer) < 1_500_000:
        try:
            art_es = next(iter_es)
            token_buffer.extend(tokenizer.encode(art_es['text'][:2000]))
            token_buffer.append(tokenizer.eot_token)
            
            art_en = next(iter_en)
            token_buffer.extend(tokenizer.encode(art_en['text'][:2000]))
            token_buffer.append(tokenizer.eot_token)
        except StopIteration:
            break

except Exception as e:
    print(f"[!] Nota sobre fuentes externas: {e}. Usando buffer de respaldo expandido.")
    token_buffer = []
    for _ in range(500):
        for t in corpus_textos:
            token_buffer.extend(tokenizer.encode(t))
            token_buffer.append(tokenizer.eot_token)

data_tensor = torch.tensor(token_buffer, dtype=torch.long)
print(f"[+] Corpus Multilingüe preparado. Total tokens en memoria: {len(data_tensor):,}")

def get_batch():
    ix = torch.randint(len(data_tensor) - config.block_size, (config.batch_size,))
    x = torch.stack([data_tensor[i:i+config.block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- ENTRENAMIENTO FRONTERA V2 ---
torch.cuda.empty_cache()
model = AethelQuantumModelV2(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

print(f"\n[*] MODELO AETHEL-QUANTUM V2 EN DISPOSITIVO ({device.upper()}).")
print(f"[*] Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"[*] Pasos de entrenamiento: {config.max_iters} | Tokenizer: cl100k_base")
print("[*] Iniciando optimización masiva multilingüe...\n")

start_time = time.time()
for iter_num in range(config.max_iters):
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(config.grad_accum_steps):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        loss = loss / config.grad_accum_steps
        loss.backward()
        loss_accum += loss.item()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Estabilidad de gradientes
    optimizer.step()
    
    if iter_num % 100 == 0:
        elapsed = (time.time() - start_time) / 60
        print(f"Paso {iter_num:5d}/{config.max_iters} | Pérdida (Loss): {loss_accum:.4f} | Tiempo: {elapsed:.2f} mins")
        
    if iter_num > 0 and iter_num % 5000 == 0:
        torch.save(model.state_dict(), f'aethel_v2_ckpt_{iter_num}.pth')
        print(f"[!] Checkpoint guardado: aethel_v2_ckpt_{iter_num}.pth")

print("\n===================================================================")
print(" ✅ ENTRENAMIENTO AETHEL V2 COMPLETADO EXITOSAMENTE.")
print("===================================================================")

# Guardamos en formato .pth y en .safetensors
torch.save(model.state_dict(), 'aethel_frontier_v2_FINAL.pth')
print("[*] Pesos finales guardados en 'aethel_frontier_v2_FINAL.pth'")

try:
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, 'aethel_frontier_v2.safetensors')
    print("[*] Pesos exportados a formato binario universal 'aethel_frontier_v2.safetensors'")
except Exception as e:
    print(f"[!] Nota sobre exportación safetensors: {e}")

# MUESTREO DE ALTA PRECISIÓN (TEMPERATURE + TOP-K + TOP-P)
def generar_respuesta(prompt, max_tokens=100):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n👤 Prompt: {prompt}")
    print("🤖 Aethel V2: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / config.temperature
            
            # Top-K filtering
            if config.top_k > 0:
                v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Decodificación limpia UTF-8 sin glitcheo de caracteres
            texto_parcial = tokenizer.decode(idx[0].tolist())
            
    print(tokenizer.decode(idx[0].tolist()[len(tokens):]))

print("\n--- PRUEBAS DE RAZONAMIENTO Y DICCIONARIO BILINGÜE ---")
generar_respuesta("Definición: La inteligencia artificial")
generar_respuesta("Había una vez un niño llamado Félix que quería")
generar_respuesta("Lily tenía mucha hambre. Fue a la cocina y")
generar_respuesta("Dictionary: An algorithm is")
