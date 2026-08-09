import React, { useState } from 'react';
import { ARCHITECTURE_CONCEPTS } from '../data/concepts';
import { ArchitectureMode } from '../types';
import { Lightbulb, CheckCircle2, AlertTriangle, ArrowRight, Zap, Database, Cpu, Brain, Layers } from 'lucide-react';

interface ExplainerSectionProps {
  onSelectSimulator: (mode: ArchitectureMode) => void;
}

export const ExplainerSection: React.FC<ExplainerSectionProps> = ({ onSelectSimulator }) => {
  const [selectedArch, setSelectedArch] = useState<ArchitectureMode>('mamba_ssm');

  const currentConcept = ARCHITECTURE_CONCEPTS.find((c) => c.id === selectedArch) || ARCHITECTURE_CONCEPTS[0];

  return (
    <div id="explainer-section" className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Hero Box explaining the paradigm shift */}
      <div id="paradigm-shift-hero" className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 p-6 md:p-8 rounded-2xl border border-indigo-500/30 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex items-start space-x-4">
          <div className="p-3 bg-indigo-500/20 text-indigo-400 rounded-xl border border-indigo-500/30">
            <Lightbulb className="w-8 h-8" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl md:text-2xl font-extrabold text-white">
              ¿Cómo crear un LLM Ultra-Eficiente e Inteligente saliendo de la caja?
            </h2>
            <p className="text-slate-300 text-sm md:text-base leading-relaxed max-w-4xl">
              Los Transformers tradicionales (GPT-4, Llama) requieren <strong className="text-amber-400 font-semibold">atención cuadrática O(N²)</strong> y activan el 100% de sus parámetros en cada palabra. Para romper este cuello de botella y crear un modelo ultra-eficiente sin perder inteligencia, se debe reemplazar el cálculo denso tradicional por 5 innovaciones revolucionarias:
            </p>
          </div>
        </div>

        {/* Key Pillars */}
        <div id="pillars-grid" className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-6 pt-6 border-t border-slate-800">
          <div className="bg-slate-900/80 p-3 rounded-xl border border-emerald-500/20 text-center">
            <Layers className="w-5 h-5 mx-auto text-emerald-400 mb-1" />
            <span className="block text-xs font-bold text-slate-200">1. Estado Continuo</span>
            <span className="text-[11px] text-slate-400">Atención Lineal O(1)</span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded-xl border border-indigo-500/20 text-center">
            <Cpu className="w-5 h-5 mx-auto text-indigo-400 mb-1" />
            <span className="block text-xs font-bold text-slate-200">2. Mezcla de Expertos</span>
            <span className="text-[11px] text-slate-400">Enrutamiento Disperso</span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded-xl border border-amber-500/20 text-center">
            <Database className="w-5 h-5 mx-auto text-amber-400 mb-1" />
            <span className="block text-xs font-bold text-slate-200">3. Pesos Ternarios</span>
            <span className="text-[11px] text-slate-400">BitNet {"{-1, 0, 1}"}</span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded-xl border border-rose-500/20 text-center">
            <Zap className="w-5 h-5 mx-auto text-rose-400 mb-1" />
            <span className="block text-xs font-bold text-slate-200">4. Pulso Neuromórfico</span>
            <span className="text-[11px] text-slate-400">Cómputo por Evento</span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded-xl border border-purple-500/20 text-center col-span-2 md:col-span-1">
            <Brain className="w-5 h-5 mx-auto text-purple-400 mb-1" />
            <span className="block text-xs font-bold text-slate-200">5. Test-Time Search</span>
            <span className="text-[11px] text-slate-400">Razonamiento MCTS</span>
          </div>
        </div>
      </div>

      {/* Architecture Selector Cards */}
      <div id="architecture-selector-section" className="space-y-4">
        <h3 className="text-base font-bold text-slate-200 flex items-center space-x-2">
          <span>Explora las 5 Arquitecturas "Fuera de la Caja":</span>
        </h3>

        <div id="architecture-tabs-grid" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {ARCHITECTURE_CONCEPTS.map((concept) => {
            const isSelected = selectedArch === concept.id;
            return (
              <button
                key={concept.id}
                id={`arch-btn-${concept.id}`}
                onClick={() => setSelectedArch(concept.id)}
                className={`p-4 rounded-xl text-left transition-all border relative overflow-hidden flex flex-col justify-between ${
                  isSelected
                    ? 'bg-slate-800 border-indigo-500 shadow-lg ring-1 ring-indigo-500/50'
                    : 'bg-slate-900/70 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] uppercase tracking-wider font-extrabold px-2 py-0.5 rounded bg-slate-950 text-indigo-400 border border-slate-800">
                      {concept.badge}
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-100 mb-1 line-clamp-2">
                    {concept.name}
                  </h4>
                  <p className="text-xs text-slate-400 line-clamp-2">
                    {concept.tagline}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-indigo-400 font-medium">
                  <span>Ver detalles</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Architecture Detailed Breakdown */}
      <div id="architecture-detail-card" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center space-x-3 mb-1">
              <h3 className="text-xl font-bold text-white">{currentConcept.name}</h3>
              <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {currentConcept.badge}
              </span>
            </div>
            <p className="text-sm text-slate-300">{currentConcept.tagline}</p>
          </div>

          <button
            id={`launch-simulator-${currentConcept.id}`}
            onClick={() => onSelectSimulator(currentConcept.id)}
            className="flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold px-4 py-2.5 rounded-xl text-xs transition-all shadow-md shadow-indigo-500/20 whitespace-nowrap self-start md:self-auto"
          >
            <Zap className="w-4 h-4" />
            <span>Probar Simulador Interactivo</span>
          </button>
        </div>

        {/* Math & Innovation Banner */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
              Eficiencia de Memoria (RAM / VRAM)
            </span>
            <div className="text-base font-bold text-emerald-400">
              {currentConcept.memoryEfficiency}
            </div>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
              Eficiencia de Cómputo (FLOPS)
            </span>
            <div className="text-base font-bold text-indigo-400">
              {currentConcept.computeEfficiency}
            </div>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
              Fórmula Clave de Funcionamiento
            </span>
            <div className="text-xs font-mono text-amber-300 bg-slate-900 p-2 rounded border border-slate-800 overflow-x-auto">
              {currentConcept.mathFormula}
            </div>
          </div>
        </div>

        {/* How It Works List */}
        <div className="space-y-3">
          <h4 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
            ¿Cómo funciona esta tecnología en la práctica?
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {currentConcept.howItWorks.map((step, idx) => (
              <div key={idx} className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 space-y-2">
                <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 text-xs font-bold border border-indigo-500/30">
                  {idx + 1}
                </span>
                <p className="text-xs text-slate-300 leading-relaxed">{step}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Pros and Cons */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="bg-emerald-950/20 border border-emerald-500/20 p-4 rounded-xl space-y-2">
            <h5 className="text-xs font-bold text-emerald-400 flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4" />
              <span>Ventajas Revolucionarias</span>
            </h5>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {currentConcept.pros.map((pro, idx) => (
                <li key={idx} className="flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">•</span>
                  <span>{pro}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-amber-950/20 border border-amber-500/20 p-4 rounded-xl space-y-2">
            <h5 className="text-xs font-bold text-amber-400 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4" />
              <span>Desafíos & Limitaciones Actuales</span>
            </h5>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {currentConcept.cons.map((con, idx) => (
                <li key={idx} className="flex items-start space-x-2">
                  <span className="text-amber-400 font-bold">•</span>
                  <span>{con}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
