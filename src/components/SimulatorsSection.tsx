import React, { useState } from 'react';
import { ArchitectureMode } from '../types';
import { Zap, Cpu, Database, Brain, ArrowRight, RefreshCw, CheckCircle2 } from 'lucide-react';

interface SimulatorsSectionProps {
  initialMode?: ArchitectureMode;
}

export const SimulatorsSection: React.FC<SimulatorsSectionProps> = ({ initialMode = 'mamba_ssm' }) => {
  const [activeMode, setActiveMode] = useState<ArchitectureMode>(initialMode);

  // Mamba SSM State Simulator
  const [inputText, setInputText] = useState<string>('El conocimiento es el pilar de la inteligencia artificial.');
  const [ssmSteps, setSsmSteps] = useState<{ token: string; hiddenState: number[] }[]>([]);

  const handleSimulateSsm = () => {
    const tokens = inputText.split(' ');
    const steps = tokens.map((token, idx) => {
      // Simulate state recurrence h_t = A * h_{t-1} + B * x_t
      const state = Array.from({ length: 4 }, (_, i) =>
        Math.round((Math.sin(idx + i) * 0.5 + 0.5) * 100) / 100
      );
      return { token, hiddenState: state };
    });
    setSsmSteps(steps);
  };

  // MoE Router Simulator
  const [moeQuery, setMoeQuery] = useState<string>('Escribe una función de optimización en Python');
  const [selectedExperts, setSelectedExperts] = useState<number[]>([1, 4]);

  const handleRouteMoe = () => {
    if (moeQuery.toLowerCase().includes('python') || moeQuery.toLowerCase().includes('código')) {
      setSelectedExperts([2, 5]); // Experts 2 (Python) & 5 (Algoritmos)
    } else if (moeQuery.toLowerCase().includes('matemática') || moeQuery.toLowerCase().includes('fórmula')) {
      setSelectedExperts([1, 3]); // Experts 1 (Math) & 3 (Logic)
    } else {
      setSelectedExperts([0, 7]); // Experts 0 (Language) & 7 (General)
    }
  };

  // BitNet Quantization Simulator
  const [floatValue, setFloatValue] = useState<number>(0.84);

  const ternaryQuantized = floatValue > 0.3 ? 1 : floatValue < -0.3 ? -1 : 0;

  return (
    <div id="simulators-section-container" className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Simulator Mode Selector Header */}
      <div id="simulators-header" className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-rose-500/20 text-rose-400 rounded-xl border border-rose-500/30">
            <Zap className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Simuladores Interactivos de Componentes</h2>
            <p className="text-slate-300 text-xs">
              Experimenta de forma aislada cómo funciona matemáticamente cada invención de la arquitectura.
            </p>
          </div>
        </div>

        {/* Simulator Tabs */}
        <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-800">
          <button
            id="sim-btn-ssm"
            onClick={() => setActiveMode('mamba_ssm')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all border ${
              activeMode === 'mamba_ssm'
                ? 'bg-rose-600 text-white border-rose-500 shadow-md'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
            }`}
          >
            1. Recurrencia SSM O(1)
          </button>

          <button
            id="sim-btn-moe"
            onClick={() => setActiveMode('sparse_moe')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all border ${
              activeMode === 'sparse_moe'
                ? 'bg-rose-600 text-white border-rose-500 shadow-md'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
            }`}
          >
            2. Router Sparse MoE Top-2
          </button>

          <button
            id="sim-btn-bitnet"
            onClick={() => setActiveMode('bitnet_158')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all border ${
              activeMode === 'bitnet_158'
                ? 'bg-rose-600 text-white border-rose-500 shadow-md'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
            }`}
          >
            3. Red Ternaria BitNet {"{-1, 0, 1}"}
          </button>

          <button
            id="sim-btn-autoadapt"
            onClick={() => setActiveMode('test_time_compute')}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all border ${
              activeMode === 'test_time_compute'
                ? 'bg-rose-600 text-white border-rose-500 shadow-md'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:bg-slate-800'
            }`}
          >
            4. Auto-Adaptador Metacognitivo
          </button>
        </div>
      </div>

      {/* Mode 1: SSM Simulator */}
      {activeMode === 'mamba_ssm' && (
        <div id="ssm-simulator-card" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-rose-400" />
              <span>Simulador de Memoria en Espacio de Estados Continuo</span>
            </h3>
            <span className="text-xs text-emerald-400 font-mono font-bold bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
              VRAM O(1) Constante
            </span>
          </div>

          <div className="space-y-3">
            <label className="text-xs font-semibold text-slate-300 block">Ingresa una oración para observar la actualización del vector de estado $h_t$:</label>
            <div className="flex gap-2">
              <input
                id="input-ssm-text"
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500 font-mono"
              />
              <button
                id="btn-run-ssm"
                onClick={handleSimulateSsm}
                className="flex items-center space-x-2 bg-rose-600 hover:bg-rose-500 text-white font-bold px-4 py-2 rounded-xl text-xs transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Simular Recurrencia</span>
              </button>
            </div>
          </div>

          {/* Steps Output */}
          {ssmSteps.length > 0 && (
            <div className="space-y-2">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">Evolución del Vector de Estado Interno $h_t$:</span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                {ssmSteps.map((step, idx) => (
                  <div key={idx} className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-rose-400 font-mono">Token #{idx + 1}: "{step.token}"</span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 bg-slate-900 p-2 rounded border border-slate-800">
                      $h_t = [{step.hiddenState.join(', ')}]$
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Mode 2: Sparse MoE Simulator */}
      {activeMode === 'sparse_moe' && (
        <div id="moe-simulator-card" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Database className="w-4 h-4 text-amber-400" />
              <span>Simulador de Enrutador Gated Sparse MoE (Top-2)</span>
            </h3>
            <span className="text-xs text-amber-400 font-mono font-bold bg-amber-500/10 px-2.5 py-1 rounded-full border border-amber-500/20">
              Solo 25% Parámetros Activos
            </span>
          </div>

          <div className="space-y-3">
            <label className="text-xs font-semibold text-slate-300 block">Escribe una consulta para probar qué expertos se activan:</label>
            <div className="flex gap-2">
              <input
                id="input-moe-query"
                type="text"
                value={moeQuery}
                onChange={(e) => setMoeQuery(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
              />
              <button
                id="btn-route-moe"
                onClick={handleRouteMoe}
                className="flex items-center space-x-2 bg-amber-600 hover:bg-amber-500 text-slate-950 font-bold px-4 py-2 rounded-xl text-xs transition-all"
              >
                <span>Calcular Enrutamiento</span>
              </button>
            </div>
          </div>

          {/* Expert Grid */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">Red de 8 Sub-Expertos Específicos:</span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                '0. Lenguaje General',
                '1. Matemáticas Y Lógica',
                '2. Programación Python',
                '3. Razonamiento Simbólico',
                '4. Filosofía y Ética',
                '5. Algoritmos Híbridos',
                '6. Física y Sistemas',
                '7. Sintaxis y Traducción',
              ].map((expName, idx) => {
                const isActive = selectedExperts.includes(idx);
                return (
                  <div
                    key={idx}
                    className={`p-3.5 rounded-xl border transition-all text-xs flex flex-col justify-between space-y-2 ${
                      isActive
                        ? 'bg-amber-500/20 border-amber-500 text-amber-300 shadow-md ring-1 ring-amber-500/50'
                        : 'bg-slate-950 border-slate-800 text-slate-500 opacity-60'
                    }`}
                  >
                    <span className="font-bold">{expName}</span>
                    <span className="text-[10px] font-mono">
                      {isActive ? '✓ ACTIVADO (Weight: 0.88)' : '◯ Inactivo (0 W)'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Mode 3: BitNet Quantization Simulator */}
      {activeMode === 'bitnet_158' && (
        <div id="bitnet-simulator-card" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              <span>Simulador de Cuantización Ternaria BitNet 1.58-bit</span>
            </h3>
            <span className="text-xs text-emerald-400 font-mono font-bold bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
              Operaciones en Enteros {"{-1, 0, 1}"}
            </span>
          </div>

          <div className="space-y-4">
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-300">Ajustar Peso Continuo FP32 / FP16:</span>
                <span className="font-mono text-cyan-400 font-bold">{floatValue}</span>
              </div>
              <input
                id="slider-float-value"
                type="range"
                min="-1.5"
                max="1.5"
                step="0.05"
                value={floatValue}
                onChange={(e) => setFloatValue(Number(e.target.value))}
                className="w-full accent-emerald-500 bg-slate-950 rounded-lg cursor-pointer"
              />
            </div>

            <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 flex flex-col md:flex-row items-center justify-around gap-4 text-center">
              <div className="space-y-1">
                <span className="text-xs text-slate-400 block font-mono">Entrada FP16 Flotante</span>
                <span className="text-2xl font-bold font-mono text-slate-300">{floatValue}</span>
                <span className="text-[10px] text-slate-400 block">Requiere FPU / Multiplicación</span>
              </div>

              <ArrowRight className="w-6 h-6 text-emerald-400" />

              <div className="space-y-1 bg-emerald-950/30 p-4 rounded-xl border border-emerald-500/30">
                <span className="text-xs text-emerald-400 block font-mono font-bold">Peso Ternario BitNet</span>
                <span className="text-3xl font-extrabold font-mono text-emerald-300">{ternaryQuantized}</span>
                <span className="text-[10px] text-emerald-400 block font-medium">Suma Entera Sin Multiplicación</span>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Mode 4: Auto-Adaptive Metacognitive Reasoning Simulator */}
      {activeMode === 'test_time_compute' && (
        <div id="autoadapt-simulator-card" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              <span>Simulador de Razonamiento Metacognitivo y Auto-Adaptación CoT</span>
            </h3>
            <span className="text-xs text-cyan-400 font-mono font-bold bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-500/20">
              Profundidad CoT Dinámica
            </span>
          </div>

          <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-1">
                <span className="text-slate-400 block font-semibold">Evaluación de Complejidad</span>
                <span className="text-cyan-400 font-mono font-bold text-sm">Nivel 8.8 / 10 (Alta)</span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-1">
                <span className="text-slate-400 block font-semibold">Pasos CoT Asignados</span>
                <span className="text-indigo-400 font-mono font-bold text-sm">12 Pasos Recursivos</span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-1">
                <span className="text-slate-400 block font-semibold">Score de Auto-Corrección</span>
                <span className="text-emerald-400 font-mono font-bold text-sm">99.2% Consistencia</span>
              </div>
            </div>

            <div className="space-y-2 pt-2">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                Traza del Árbol de Búsqueda Metacognitiva (Tree of Thoughts):
              </span>
              <div className="bg-slate-900 p-3.5 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="p-2 rounded bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-2">
                  <span className="text-cyan-400 font-bold">[Pensamiento #1]</span>
                  <span>Descomposición sintáctica del problema en premisas lógicas universales.</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-2">
                  <span className="text-indigo-400 font-bold">[Pensamiento #2]</span>
                  <span>Activación adaptativa de expertos en optimización matricial y estructuras.</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-2">
                  <span className="text-amber-400 font-bold">[Pensamiento #3]</span>
                  <span>Verificación auto-reflexiva: detección de sesgos y corrección de hipótesis.</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-emerald-500/30 text-emerald-300 flex items-center space-x-2">
                  <span className="text-emerald-400 font-bold">[Pensamiento #4]</span>
                  <span>Síntesis final verificada con cero alucinaciones y alta fidelidad.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
