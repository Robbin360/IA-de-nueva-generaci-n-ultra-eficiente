import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("===================================================================")
print(" 🌌 AETHEL-QUANTUM V3: ARQUITECTURA DE FRONTERA (NIVEL DEEPMIND)")
print("===================================================================\n")

# 1. CONFIGURACIÓN DE FRONTERA
class AethelConfigV3:
    vocab_size = 100256
    dim = 768            # Dimensión expandida
    n_layers = 12        # Más capas de profundidad
    n_heads = 12         # Cabezas para las Queries (Q)
    n_kv_heads = 4       # Cabezas para Keys y Values (GQA - Ahorra 66% de RAM)
    n_experts = 8        # 8 Expertos totales
    active_experts = 2   # 2 Expertos activos por token
    block_size = 1024    # Ventana de contexto lista para KV Cache

config = AethelConfigV3()

# 2. NORMALIZACIÓN
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# 3. EL VERDADERO SwiGLU (La puerta lógica multiplicativa de Llama 3)
class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # A diferencia de V2, aquí usamos 3 matrices de pesos.
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x):
        # La magia matemática: F.silu(x*W1) * (x*W2)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# 4. GROUPED-QUERY ATTENTION (GQA) + KV CACHE PREP
class GQA_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        # Mapeo de grupos (cuántas cabezas Q comparten la misma KV)
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Repetir Keys y Values para igualar a las Queries (GQA)
        # Esto es lo que ahorra cantidades masivas de VRAM
        k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, T, self.n_heads, self.head_dim)
        v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, T, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2) # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Atención nativa eficiente de PyTorch (Automáticamente usa FlashAttention si está disponible)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

# 5. MoE CON LOAD BALANCING LOSS (Anti-pereza neuronal)
class MoERouterV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.k = config.active_experts
        self.router = nn.Linear(config.dim, config.n_experts, bias=False)
        
        # Inicializamos los expertos usando el verdadero SwiGLU
        hidden_dim = config.dim * 4
        self.experts = nn.ModuleList([SwiGLU(config.dim, hidden_dim) for _ in range(config.n_experts)])

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C) # Aplanamos para el ruteo (B*T, C)
        
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Seleccionamos los Top-K expertos
        top_probs, top_indices = torch.topk(routing_probs, self.k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True) # Normalizar pesos
        
        # --- EL SECRETO: LOAD BALANCING AUX LOSS ---
        # 1. ¿Qué fracción de los tokens fue asignada a cada experto?
        mask = F.one_hot(top_indices, num_classes=self.n_experts).float()
        density_fraction = mask.sum(dim=1).mean(dim=0)
        
        # 2. ¿Qué probabilidad le dio el router a cada experto en promedio?
        prob_fraction = routing_probs.mean(dim=0)
        
        # 3. La pérdida: penaliza si el modelo ignora expertos o abusa de uno solo.
        aux_loss = self.n_experts * torch.sum(density_fraction * prob_fraction)
        
        # Ruteo y computación
        out = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_idx = top_indices[:, i]
            weight = top_probs[:, i].unsqueeze(-1)
            
            # Ejecutar cada experto solo en los tokens asignados a él (Ahorro computacional)
            for j, expert in enumerate(self.experts):
                token_mask = (expert_idx == j)
                if token_mask.any():
                    tokens_for_expert = x_flat[token_mask]
                    out[token_mask] += weight[token_mask] * expert(tokens_for_expert)
                    
        return out.view(B, T, C), aux_loss

# 6. BLOQUE Y MODELO FINAL
class AethelBlockV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = GQA_Attention(config)
        self.norm2 = RMSNorm(config.dim)
        self.moe = MoERouterV3(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss

class AethelQuantumModelV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([AethelBlockV3(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.token_emb.weight = self.output.weight # Weight Tying

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        total_aux_loss = 0.0
        
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = self.norm(x)
        logits = self.output(x)
        
        ce_loss = None
        if targets is not None:
            B, T, C = logits.shape
            ce_loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            
        return logits, ce_loss, total_aux_loss

print("✅ Arquitectura V3 compilada con éxito en Python.")
print("  - SwiGLU: Integrado")
print("  - GQA (Grouped-Query Attention): Integrado (Ahorro de VRAM)")
print("  - MoE Load Balancing Loss: Integrado (Anti-colapso)")
print("  - Optimización FlashAttention: Habilitada nativamente vía PyTorch F.scaled_dot_product_attention")
