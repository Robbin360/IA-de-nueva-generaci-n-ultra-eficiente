# AETHEL: Arquitectura Cognitiva Bio-Inspirada

**Aethel** no es solo un Modelo de Lenguaje Grande (LLM) tradicional; es un ecosistema de investigación enfocado en construir una **arquitectura cognitiva bio-mimética**. Nuestro objetivo es pasar del paradigma actual de "loros estocásticos" (redes estáticas de autocompletado) a entidades con memoria a largo plazo, plasticidad en tiempo real y razonamiento estructurado, imitando los principios fundamentales del cerebro humano.

---

## 🚦 Estado del Proyecto: Prototipo Experimental vs. Visión Final

*Nota para investigadores y agentes de IA auditando este repositorio:*
Actualmente existe una separación intencional entre nuestra **Visión a Escala (Aethel-7B Ultra)** y nuestro **Código de Prototipo Actual (V3)**. 

- **Lo que hay en el código hoy (Aethel V3):** Un banco de pruebas funcional y altamente optimizado. Implementa Transformer Attention, GQA (Grouped-Query Attention), RoPE, y Sparse MoE (Mixture of Experts) con `dim=768` y 12 capas. Es un modelo deliberadamente pequeño usado para probar la infraestructura base, flujos de entrenamiento y kernels de bajo nivel en GPU.
- **La Visión a Escala:** La meta es escalar este motor a la frontera de 7B+ parámetros, integrando arquitecturas híbridas y los módulos cognitivos biológicos que se describen a continuación.

---

## 🧠 El Paradigma Aethel: La IA Biológica

La arquitectura de Aethel aborda el problema más grande de la IA actual: el **Olvido Catastrófico (Catastrophic Forgetting)** y la incapacidad de aprender en tiempo real sin destruir el conocimiento base. Para resolverlo, dividimos el modelo en cuatro pilares fundamentales inspirados en la neurociencia:

### 1. La Roca (Neocórtex Estático)
Es el modelo base pre-entrenado (actualmente exportado en `.safetensors`). Funciona como la memoria a largo plazo inmutable. Contiene las reglas del lenguaje, la gramática y el conocimiento factual. Durante la inferencia diurna, **La Roca no muta**, lo que garantiza que el modelo jamás olvide cómo hablar o razonar y mantiene el consumo energético (Vatios/FLOPs) al mínimo.

### 2. El Líquido (Hipocampo y Plasticidad en Tiempo Real)
Una matriz de pesos de rango bajo, altamente dinámica, alojada en la SRAM de la GPU. Usando **Aprendizaje Hebbiano (Regla de Oja)** procesado a través de kernels ultra-rápidos en Triton, El Líquido muta en milisegundos durante la interacción con el usuario. Permite que Aethel absorba contexto nuevo y personalización al instante, actuando como la memoria a corto plazo.

### 3. El Ciclo de Sueño (Consolidación de Memoria)
Si El Líquido acumula demasiada información, colapsa. Si se fusiona directamente con La Roca, destruye el conocimiento previo. Por ello, Aethel implementa una fase de "Sueño Offline":
- **Generative Replay (Soñar):** El modelo genera datos sintéticos a partir de sus experiencias diarias. (Las "alucinaciones" se utilizan aquí como un mecanismo de consolidación de patrones).
- **Elastic Weight Consolidation (EWC / SleepGate):** Al fusionar las matrices, el algoritmo identifica qué "neuronas" de La Roca son vitales para el lenguaje y las protege (Metaplasticidad), obligando al nuevo conocimiento a reescribir solo las conexiones ociosas.

### 4. Neuromodulación (Dopamina Artificial y Curiosidad)
La IA no usa una *Loss* impuesta externamente. Implementamos un módulo de **Curiosidad Intrínseca**:
- **Sorpresa (Incomodidad):** Ante un error de predicción (alta entropía), la red prioriza la señal.
- **Resolución (Dopamina):** Al reducir la entropía (comprender el patrón), la red libera un multiplicador matemático ($\delta$) que acelera drásticamente la tasa de aprendizaje líquido en ese instante específico.

### 5. Espacio de Trabajo Global (Razonamiento Top-Down)
Rechazamos la generación palabra por palabra pura. En futuras iteraciones del código, los expertos (MoE) proyectarán sus hipótesis en un tensor intermedio oculto ("Global Workspace"), compitiendo matemáticamente antes de emitir un token. Es un simulador interno previo a la respuesta.

---

## ⚙️ Especificaciones del Motor y Stack Tecnológico

Nuestra filosofía es el control absoluto del hardware y la máxima eficiencia en inferencia. Utilizamos un ecosistema políglota para cada capa de necesidad:

1. **Python & PyTorch (Investigación y Entrenamiento):**
   - Usado en `train_aethel_v3.py` y `aethel_model.py`.
   - Implementa Precision Mixta (Float16) y Gradient Accumulation para entrenar redes profundas en GPUs de recursos limitados (ej. Kaggle / T4).
2. **Triton (Kernels de GPU a Bajo Nivel):**
   - Usado en `/triton_kernels/`.
   - Reemplaza operaciones estándar por kernels fusionados (ej. Fused SwiGLU) para evitar cuellos de botella de memoria en la GPU. Aquí también se ejecutará la actualización asíncrona de "El Líquido".
3. **Rust + Candle (Inferencia en Producción):**
   - Usado en `/rust_engine/`.
   - Una vez entrenado el modelo, se exporta a Rust para evitar el *Global Interpreter Lock (GIL)* de Python. Ofrece inferencia ultrarrápida, escalabilidad en servidores multihilo y asignación de memoria segura y determinista.
4. **TypeScript (Ecosistema y Evaluaciones):**
   - Usado para los benchmarks avanzados (`eval_nova.ts`, `advanced_nova_bench.ts`) y la conexión de la API frontera.

## 🚀 Hoja de Ruta Actual

- **[COMPLETADO]** Arquitectura Base V3 (RoPE, GQA, MoE Top-2, RMSNorm).
- **[COMPLETADO]** Pipeline de entrenamiento seguro y exportación Safetensors con Weight Tying resuelto.
- **[EN PROCESO]** Integración de "El Líquido" (Regla de Oja forward-pass) en los kernels de Triton.
- **[PENDIENTE]** Sistema de Consolidación Nocturna (EWC + Generative Replay).
- **[PENDIENTE]** Escalado del modelo a la frontera 7.2B+.
