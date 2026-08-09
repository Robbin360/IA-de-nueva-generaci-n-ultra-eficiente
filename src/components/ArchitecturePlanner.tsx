import React from 'react';
import { ModelHyperparameters } from '../types';
import { Sliders, Cpu, Database, Zap, Layers, CpuIcon, CheckCircle2, ArrowRight } from 'lucide-react';

interface ArchitecturePlannerProps {
  params: ModelHyperparameters;
  setParams: React.Dispatch<React.SetStateAction<ModelHyperparameters>>;
  onGoToTraining: () => void;
  onGoToCode: () => void;
}

export const ArchitecturePlanner: React.FC<ArchitecturePlannerProps> = ({
  params,
  setParams,
  onGoToTraining,
  onGoToCode,
}) => {
  // Estimated size calculations
  const totalParamsMillion = Math.round(
    ((params.vocabSize * params.hiddenDim) +
      params.numLayers * (
        params.hiddenDim * params.stateDim * 2 + // SSM
        params.numExperts * (params.hiddenDim * params.hiddenDim * 2) // MoE
      )) / 1_000_000
  );

  const activeParamsMillion = Math.round(
    ((params.vocabSize * params.hiddenDim) +
      params.numLayers * (
        params.hiddenDim * params.stateDim * 2 +
        params.activeExpertsPerToken * (params.hiddenDim * params.hiddenDim * 2)
      )) / 1_000_000
  );

  const formatParamCount = (mCount: number) => {
    if (mCount >= 1_000_000) {
      return `${(mCount / 1_000_000).toFixed(2)} Trillones (T)`;
    } else if (mCount >= 1_000) {
      return `${(mCount / 1_000).toFixed(1)} Billones (B)`;
    }
    return `${mCount} Millones (M)`;
  };

  const bytesPerWeight = params.quantizationBits === '1.58b' ? 0.2 : params.quantizationBits === '4b' ? 0.5 : params.quantizationBits === '8b' ? 1 : 2;
  const memoryUsageMb = Math.round((totalParamsMillion * bytesPerWeight * 10) / 10);
  const memoryDisplay = memoryUsageMb >= 1024 * 1024 
    ? `${(memoryUsageMb / (1024 * 1024)).toFixed(2)} TB`
    : memoryUsageMb >= 1024 
    ? `${(memoryUsageMb / 1024).toFixed(1)} GB`
    : `${memoryUsageMb} MB`;

  const kvCacheSavingsPercent = 99.6; // O(1) state vs O(N^2)

  // Presets handler
  const applyPreset = (preset: 'micro_500m' | 'compact_7b' | 'frontier_70b' | 'ultra_405b' | 'hyper_1_8t' | 'titan_5_6t') => {
    if (preset === 'micro_500m') {
      setParams({
        ...params,
        modelName: 'Aethel-1 Micro (0.5B Edge)',
        hiddenDim: 1024,
        numLayers: 12,
        numExperts: 4,
        activeExpertsPerToken: 1,
        stateDim: 16,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 2,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    } else if (preset === 'compact_7b') {
      setParams({
        ...params,
        modelName: 'Aethel-1 Compact (7B MoE)',
        hiddenDim: 2048,
        numLayers: 24,
        numExperts: 8,
        activeExpertsPerToken: 2,
        stateDim: 32,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 6,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    } else if (preset === 'frontier_70b') {
      setParams({
        ...params,
        modelName: 'Aethel-1 Frontier (70B MoE)',
        hiddenDim: 4096,
        numLayers: 48,
        numExperts: 16,
        activeExpertsPerToken: 2,
        stateDim: 64,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 12,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    } else if (preset === 'ultra_405b') {
      setParams({
        ...params,
        modelName: 'Aethel-1 Ultra (405B MoE)',
        hiddenDim: 8192,
        numLayers: 96,
        numExperts: 32,
        activeExpertsPerToken: 4,
        stateDim: 128,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 20,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    } else if (preset === 'hyper_1_8t') {
      setParams({
        ...params,
        modelName: 'Aethel-Compact Distilled (7B MoE)',
        hiddenDim: 12288,
        numLayers: 96,
        numExperts: 64,
        activeExpertsPerToken: 4,
        stateDim: 128,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 24,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    } else if (preset === 'titan_5_6t') {
      setParams({
        ...params,
        modelName: 'Aethel-1 Titan (5.6T MoE Base)',
        hiddenDim: 16384,
        numLayers: 128,
        numExperts: 128,
        activeExpertsPerToken: 8,
        stateDim: 256,
        quantizationBits: '1.58b',
        testTimeReasoningDepth: 32,
        enableSelfAdaptiveRouting: true,
        autoReasoningDepth: true,
      });
    }
  };

  return (
    <div id="architecture-planner-container" className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Plan Blueprint Header */}
      <div id="planner-hero" className="bg-gradient-to-r from-slate-900 via-cyan-950 to-slate-900 p-6 md:p-8 rounded-2xl border border-cyan-500/30 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-72 h-72 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="inline-flex items-center space-x-2 bg-cyan-500/10 text-cyan-400 text-xs px-3 py-1 rounded-full font-semibold border border-cyan-500/20">
              <Sliders className="w-3.5 h-3.5" />
              <span>Paso 1: Planificación de Arquitectura No Convencional & Auto-Adaptativa</span>
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
              Diseño de Modelo Híbrido: <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400">{params.modelName}</span>
            </h2>
            <p className="text-slate-300 text-sm leading-relaxed">
              Configura y escala la capacidad computacional de tu LLM hasta cientos de miles de millones de parámetros. Incluye <strong>Auto-Adaptación Dinámica</strong>, <strong>Espacio de Estados O(1)</strong>, <strong>Mezcla de Expertos Dispersa (Sparse MoE)</strong>, <strong>BitNet 1.58b</strong> y <strong>Razonamiento Recursivo CoT</strong>.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              id="btn-go-to-training"
              onClick={onGoToTraining}
              className="flex items-center space-x-2 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold px-4 py-2.5 rounded-xl text-xs transition-all shadow-md shadow-amber-500/20"
            >
              <span>Ir a Educar / Entrenar LLM</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              id="btn-go-to-code"
              onClick={onGoToCode}
              className="flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold px-4 py-2.5 rounded-xl text-xs transition-all border border-slate-700"
            >
              <span>Ver Código PyTorch</span>
            </button>
          </div>
        </div>
      </div>

      {/* Scale Presets Selector Bar */}
      <div id="scale-presets-bar" className="bg-slate-900 p-4 rounded-2xl border border-slate-800 shadow-lg space-y-2">
        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
          ⚡ Escalas de Parámetros Preconfiguradas (Model Scaling Presets):
        </span>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          <button
            id="preset-micro"
            onClick={() => applyPreset('micro_500m')}
            className="p-2.5 rounded-xl border border-slate-800 bg-slate-950 hover:border-cyan-500/50 transition-all text-left space-y-1"
          >
            <span className="text-xs font-bold text-cyan-400 block">0.5B Micro</span>
            <span className="text-[10px] text-slate-400 block">1024d / 12L</span>
          </button>

          <button
            id="preset-compact"
            onClick={() => applyPreset('compact_7b')}
            className="p-2.5 rounded-xl border border-slate-800 bg-slate-950 hover:border-indigo-500/50 transition-all text-left space-y-1"
          >
            <span className="text-xs font-bold text-indigo-400 block">7B Compact</span>
            <span className="text-[10px] text-slate-400 block">2048d / 24L</span>
          </button>

          <button
            id="preset-frontier"
            onClick={() => applyPreset('frontier_70b')}
            className="p-2.5 rounded-xl border border-slate-800 bg-slate-950 hover:border-purple-500/50 transition-all text-left space-y-1"
          >
            <span className="text-xs font-bold text-purple-400 block">70B Frontier</span>
            <span className="text-[10px] text-slate-400 block">4096d / 48L</span>
          </button>

          <button
            id="preset-ultra"
            onClick={() => applyPreset('ultra_405b')}
            className="p-2.5 rounded-xl border border-slate-800 bg-slate-950 hover:border-rose-500/50 transition-all text-left space-y-1"
          >
            <span className="text-xs font-bold text-rose-400 block">405B Ultra</span>
            <span className="text-[10px] text-slate-400 block">8192d / 96L</span>
          </button>

          <button
            id="preset-fable5"
            onClick={() => applyPreset('hyper_1_8t')}
            className="p-2.5 rounded-xl border border-amber-500/30 bg-amber-950/20 hover:border-amber-400 transition-all text-left space-y-1"
          >
            <span className="text-xs font-bold text-amber-400 block">7B Compacto</span>
            <span className="text-[10px] text-slate-300 block">64 Exp / 12288d</span>
          </button>

          <button
            id="preset-gpt56sol"
            onClick={() => applyPreset('titan_5_6t')}
            className="p-2.5 rounded-xl border border-emerald-500/30 bg-emerald-950/20 hover:border-emerald-400 transition-all text-left space-y-1 ring-1 ring-emerald-500/50"
          >
            <span className="text-xs font-bold text-emerald-400 block">5.6T Titan Base</span>
            <span className="text-[10px] text-slate-300 block">128 Exp / 16384d</span>
          </button>
        </div>
      </div>

      {/* Live Parameter Controls & Estimated Model Blueprint */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Parameter Sliders */}
        <div id="parameter-controls-panel" className="lg:col-span-7 bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
          <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2 pb-3 border-b border-slate-800">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span>Ajustar Hiperparámetros del Motor</span>
          </h3>

          <div className="space-y-5">
            {/* Model Name Input */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 flex justify-between">
                <span>Nombre de la Arquitectura / Identificador</span>
              </label>
              <input
                id="input-model-name"
                type="text"
                value={params.modelName}
                onChange={(e) => setParams({ ...params, modelName: e.target.value })}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Hidden Dimension d_model */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-300">Dimensión Oculta ($d_{'{model}'}$)</span>
                <span className="font-mono text-cyan-400 font-bold">{params.hiddenDim} px</span>
              </div>
              <input
                id="slider-hidden-dim"
                type="range"
                min="512"
                max="32768"
                step="512"
                value={params.hiddenDim}
                onChange={(e) => setParams({ ...params, hiddenDim: Number(e.target.value) })}
                className="w-full accent-cyan-500 bg-slate-950 rounded-lg cursor-pointer"
              />
              <p className="text-[11px] text-slate-400">Controla la capacidad vectorial de representación conceptual por token.</p>
            </div>

            {/* Number of Layers */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-300">Número de Capas Profundas ($L$)</span>
                <span className="font-mono text-indigo-400 font-bold">{params.numLayers} capas</span>
              </div>
              <input
                id="slider-num-layers"
                type="range"
                min="4"
                max="160"
                step="4"
                value={params.numLayers}
                onChange={(e) => setParams({ ...params, numLayers: Number(e.target.value) })}
                className="w-full accent-indigo-500 bg-slate-950 rounded-lg cursor-pointer"
              />
              <p className="text-[11px] text-slate-400">Capas recurrentes híbridas SSM + MoE superpuestas.</p>
            </div>

            {/* MoE Experts & Active Experts */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-300 font-sans">Total de Expertos ($E$)</span>
                  <span className="font-mono text-amber-400 font-bold">{params.numExperts} exp</span>
                </div>
                <input
                  id="slider-num-experts"
                  type="range"
                  min="2"
                  max="2048"
                  step="2"
                  value={params.numExperts}
                  onChange={(e) => {
                    const numExperts = Number(e.target.value);
                    setParams({
                      ...params,
                      numExperts,
                      activeExpertsPerToken: Math.min(params.activeExpertsPerToken, numExperts),
                    });
                  }}
                  className="w-full accent-amber-500 bg-slate-950 rounded-lg cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-300">Expertos Activos / Token ($k$)</span>
                  <span className="font-mono text-emerald-400 font-bold">{params.activeExpertsPerToken} exp</span>
                </div>
                <input
                  id="slider-active-experts"
                  type="range"
                  min="1"
                  max={Math.min(params.numExperts, 128)}
                  step="1"
                  value={params.activeExpertsPerToken}
                  onChange={(e) => setParams({ ...params, activeExpertsPerToken: Number(e.target.value) })}
                  className="w-full accent-emerald-500 bg-slate-950 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* SSM State Dimension d_state */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-300">Dimensión de Estado SSM ($d_{'{state}'}$)</span>
                <span className="font-mono text-purple-400 font-bold">{params.stateDim} (Mamba Memory)</span>
              </div>
              <input
                id="slider-state-dim"
                type="range"
                min="8"
                max="512"
                step="8"
                value={params.stateDim}
                onChange={(e) => setParams({ ...params, stateDim: Number(e.target.value) })}
                className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer"
              />
              <p className="text-[11px] text-slate-400">Proyección de memoria matricial continua sin guardar tokens pasados.</p>
            </div>

            {/* Auto-Adaptive Routing Toggles */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">Enrutamiento y Razonamiento Auto-Adaptativo</span>
                <input
                  id="toggle-self-adaptive"
                  type="checkbox"
                  checked={params.enableSelfAdaptiveRouting ?? true}
                  onChange={(e) => setParams({ ...params, enableSelfAdaptiveRouting: e.target.checked })}
                  className="w-4 h-4 accent-cyan-500 cursor-pointer"
                />
              </div>
              <p className="text-[11px] text-slate-400">
                Ajusta dinámicamente la cantidad de expertos activados y los pasos de razonamiento según la complejidad sintáctica y semántica del prompt.
              </p>
            </div>

            {/* Quantization Bits */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 block">Formato de Pesos & Cuantización</label>
              <div className="grid grid-cols-4 gap-2">
                {(['1.58b', '4b', '8b', '16b'] as const).map((q) => (
                  <button
                    key={q}
                    id={`btn-quant-${q}`}
                    onClick={() => setParams({ ...params, quantizationBits: q })}
                    className={`py-2 px-2 rounded-xl text-xs font-mono font-bold transition-all border ${
                      params.quantizationBits === q
                        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500 shadow-md'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {q === '1.58b' ? 'BitNet 1.58b' : q.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Test-Time Reasoning Depth */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-300">Profundidad Búsqueda Test-Time (Razonamiento CoT)</span>
                <span className="font-mono text-rose-400 font-bold">{params.testTimeReasoningDepth} pasos</span>
              </div>
              <input
                id="slider-reasoning-depth"
                type="range"
                min="1"
                max="32"
                step="1"
                value={params.testTimeReasoningDepth}
                onChange={(e) => setParams({ ...params, testTimeReasoningDepth: Number(e.target.value) })}
                className="w-full accent-rose-500 bg-slate-950 rounded-lg cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Right Column: Live Model Blueprint Statistics */}
        <div id="blueprint-stats-panel" className="lg:col-span-5 space-y-6">
          {/* Estimated Metrics Summary Card */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-5">
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2 pb-3 border-b border-slate-800">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>Métricas de la Arquitectura Diseñada</span>
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 space-y-1">
                <span className="text-[11px] text-slate-400 block font-medium">Parámetros Totales</span>
                <span className="text-lg font-bold font-mono text-cyan-400">{formatParamCount(totalParamsMillion)}</span>
                <span className="text-[10px] text-slate-400 block">En disco / Cluster</span>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 space-y-1">
                <span className="text-[11px] text-slate-400 block font-medium">Parámetros Activos</span>
                <span className="text-lg font-bold font-mono text-emerald-400">{formatParamCount(activeParamsMillion)}</span>
                <span className="text-[10px] text-slate-400 block">Por Token (Sparse)</span>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 space-y-1">
                <span className="text-[11px] text-slate-400 block font-medium">Huella VRAM Estimada</span>
                <span className="text-lg font-bold font-mono text-amber-400">{memoryDisplay}</span>
                <span className="text-[10px] text-slate-400 block">Formato {params.quantizationBits}</span>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 space-y-1">
                <span className="text-[11px] text-slate-400 block font-medium">Ahorro Memoria KV</span>
                <span className="text-lg font-bold font-mono text-rose-400">{kvCacheSavingsPercent}%</span>
                <span className="text-[10px] text-slate-400 block">vs Attention O(N²)</span>
              </div>
            </div>

            {/* Architecture Stack Representation */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                Estructura de Capas Intercaladas:
              </span>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="flex items-center justify-between text-cyan-300 bg-cyan-950/40 p-2 rounded border border-cyan-500/20">
                  <span className="flex items-center space-x-2">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Bloque 1: State-Space SSM Kernel</span>
                  </span>
                  <span className="text-[10px] font-sans text-cyan-400">Memory $O(1)$</span>
                </div>

                <div className="flex items-center justify-between text-amber-300 bg-amber-950/40 p-2 rounded border border-amber-500/20">
                  <span className="flex items-center space-x-2">
                    <CpuIcon className="w-3.5 h-3.5" />
                    <span>Bloque 2: Router MoE Top-{params.activeExpertsPerToken} of {params.numExperts}</span>
                  </span>
                  <span className="text-[10px] font-sans text-amber-400">Sparse MoE</span>
                </div>

                <div className="flex items-center justify-between text-emerald-300 bg-emerald-950/40 p-2 rounded border border-emerald-500/20">
                  <span className="flex items-center space-x-2">
                    <Database className="w-3.5 h-3.5" />
                    <span>Bloque 3: Multiplicación Ternaria BitNet</span>
                  </span>
                  <span className="text-[10px] font-sans text-emerald-400">{"{-1, 0, 1}"}</span>
                </div>

                <div className="flex items-center justify-between text-rose-300 bg-rose-950/40 p-2 rounded border border-rose-500/20">
                  <span className="flex items-center space-x-2">
                    <Zap className="w-3.5 h-3.5" />
                    <span>Bloque 4: Árbol Test-Time ({params.testTimeReasoningDepth} Pasos)</span>
                  </span>
                  <span className="text-[10px] font-sans text-rose-400">MCTS Search</span>
                </div>
              </div>
            </div>
          </div>

          {/* Key Architectural Advancements */}
          <div className="bg-slate-900/80 p-5 rounded-2xl border border-slate-800 space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              ¿Por qué esta combinación supera a los Transformers?
            </h4>

            <ul className="space-y-2 text-xs text-slate-300">
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>No requiere KV-Cache masiva:</strong> Procesa secuencias de 1M+ palabras manteniendo la VRAM constante en pocos megabytes.</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Suma de Enteros en CPU/NPU:</strong> BitNet convierte multiplicaciones flotantes de coma fija en simples sumas y restas de números enteros.</span>
              </li>
              <li className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span><strong>Especialización Modular:</strong> Los sub-expertos solo se encienden para el tipo de problema específico (Código, Filosofía, Matemáticas).</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
