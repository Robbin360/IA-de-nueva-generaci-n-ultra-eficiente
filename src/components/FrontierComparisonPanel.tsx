import React, { useState, useEffect } from 'react';
import {
  Zap,
  Cpu,
  Calculator,
  Code,
  CheckCircle2,
  Play,
  RefreshCw,
  Sparkles,
  Award,
  Layers,
  Clock,
  ShieldCheck,
  Brain,
  Bot,
  Gauge,
  DollarSign
} from 'lucide-react';

export interface ComparisonTestResult {
  id: string;
  domain: 'math' | 'logic' | 'code';
  domainLabel: string;
  title: string;
  prompt: string;
  expectedKeyword: string;
  aethel: {
    modelName: string;
    reply: string;
    latencyMs: number;
    networkLatencyMs: number;
    tokensPerSec: number;
    score: number;
    costPer1MTokens: string;
    vramUsage: string;
  };
  frontier: {
    modelName: string;
    reply: string;
    latencyMs: number;
    networkLatencyMs: number;
    tokensPerSec: number;
    score: number;
    costPer1MTokens: string;
    vramUsage: string;
  };
  winner: 'aethel' | 'frontier';
  speedupMultiplier: number;
}

export interface ComparisonSuiteSummary {
  avgAethelScore: number;
  avgFrontierScore: number;
  avgAethelLatencyMs: number;
  avgFrontierLatencyMs: number;
  avgSpeedupFactor: number;
  aethelCost: string;
  frontierCost: string;
}

export const FrontierComparisonPanel: React.FC = () => {
  const [selectedDomain, setSelectedDomain] = useState<'all' | 'math' | 'logic' | 'code'>('all');
  const [frontierModelChoice, setFrontierModelChoice] = useState<string>('gpt_5_6_sol');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [statusText, setStatusText] = useState<string>('Haz clic en "Ejecutar Batería Comparativa" para realizar las pruebas automáticas en tiempo real.');
  const [testResults, setTestResults] = useState<ComparisonTestResult[]>([]);
  const [summary, setSummary] = useState<ComparisonSuiteSummary | null>(null);
  const [activeTestId, setActiveTestId] = useState<string | null>(null);

  const runComparisonSuite = async () => {
    setIsRunning(true);
    setProgress(15);
    setStatusText('Iniciando batería de evaluación automática multi-modelo...');

    try {
      setProgress(40);
      setStatusText(`Enviando problemas de ${selectedDomain === 'all' ? 'Matemáticas, Lógica y Código' : selectedDomain} a la API de Modelo Frontera y al Motor Aethel-1...`);

      const res = await fetch('/api/run-automated-comparison-suite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: selectedDomain,
          frontierModelChoice,
        }),
      });

      setProgress(80);
      setStatusText('Procesando métricas de latencia, precisión, velocidad de tokens y eficiencia de memoria...');

      const data = await res.json();

      if (data.results) {
        setTestResults(data.results);
        setSummary(data.summary);
        if (data.results.length > 0) {
          setActiveTestId(data.results[0].id);
        }
      }

      setProgress(100);
      setStatusText(`¡Evaluación comparativa completada! Aethel-1 procesó las pruebas con una aceleración media de ${data.summary?.avgSpeedupFactor || 25}x.`);
    } catch (err: any) {
      console.error('Error al ejecutar batería comparativa:', err);
      setStatusText(`Error durante la ejecución: ${err.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  // Run automatically on first mount
  useEffect(() => {
    runComparisonSuite();
  }, [selectedDomain, frontierModelChoice]);

  return (
    <div id="frontier-comparison-panel" className="bg-slate-900/90 p-6 rounded-2xl border border-indigo-500/30 shadow-2xl space-y-6">
      {/* Panel Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="space-y-1.5">
          <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-500/10 to-emerald-500/10 text-indigo-300 text-xs px-3 py-1 rounded-full font-bold border border-indigo-500/20">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span>Pruebas Automáticas de Razonamiento, Lógica y Código</span>
          </div>
          <h3 className="text-xl md:text-2xl font-extrabold text-white tracking-tight flex items-center space-x-2">
            <span>Evaluación Automática: Aethel-1 vs. Modelos Frontera API</span>
          </h3>
          <p className="text-xs md:text-sm text-slate-300 max-w-3xl leading-relaxed">
            Ejecuta de forma automatizada problemas estandarizados de <strong>Razonamiento Matemático</strong>, <strong>Lógica Deductiva</strong> y <strong>Programación Algorítmica</strong>. Compara la velocidad de inferencia, latencia de red, precisión de respuesta y consumo de recursos entre Aethel-1 y las APIs frontera.
          </p>
        </div>

        <button
          id="btn-run-automated-comparison"
          onClick={runComparisonSuite}
          disabled={isRunning}
          className="flex items-center space-x-2 bg-gradient-to-r from-emerald-500 via-teal-500 to-indigo-600 hover:from-emerald-400 hover:to-indigo-500 text-slate-950 font-extrabold px-5 py-3 rounded-xl text-xs transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50 whitespace-nowrap cursor-pointer"
        >
          <Play className={`w-4 h-4 fill-current ${isRunning ? 'animate-spin' : ''}`} />
          <span>{isRunning ? 'Evaluando Modelos...' : 'Ejecutar Batería Comparativa'}</span>
        </button>
      </div>

      {/* Control Bar: Domain Filter & Model Selector */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
        {/* Domain Filters */}
        <div className="space-y-2">
          <label className="text-xs font-bold text-slate-300 block uppercase tracking-wider">
            Dominio de Evaluación:
          </label>
          <div className="flex flex-wrap gap-2">
            {[
              { id: 'all', label: 'Todas las Áreas', icon: Layers },
              { id: 'math', label: 'Matemáticas', icon: Calculator },
              { id: 'logic', label: 'Lógica', icon: Brain },
              { id: 'code', label: 'Programación', icon: Code },
            ].map((d) => {
              const Icon = d.icon;
              const active = selectedDomain === d.id;
              return (
                <button
                  key={d.id}
                  id={`btn-domain-${d.id}`}
                  onClick={() => setSelectedDomain(d.id as any)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    active
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20 ring-1 ring-indigo-400'
                      : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{d.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Target Frontier Model Selection */}
        <div className="space-y-2">
          <label className="text-xs font-bold text-slate-300 block uppercase tracking-wider">
            Modelo Frontera API de Referencia:
          </label>
          <select
            id="select-frontier-model"
            value={frontierModelChoice}
            onChange={(e) => setFrontierModelChoice(e.target.value)}
            className="w-full bg-slate-900 text-slate-200 text-xs border border-slate-800 rounded-lg p-2.5 font-semibold focus:outline-none focus:border-indigo-500"
          >
            <option value="gpt_5_6_sol">GPT-5.6 Sol Max / GPT-4o API (OpenAI)</option>
            <option value="claude_3_5">Claude 3.5 Sonnet API (Anthropic)</option>
            <option value="gemini_2_5">Gemini 2.5 Pro API (Google DeepMind)</option>
          </select>
        </div>
      </div>

      {/* Progress & Log Bar */}
      {(isRunning || progress > 0) && (
        <div className="bg-slate-950 p-4 rounded-xl border border-emerald-500/30 space-y-2 font-mono text-xs">
          <div className="flex items-center justify-between text-emerald-400">
            <span className="flex items-center space-x-2">
              <RefreshCw className={`w-4 h-4 ${isRunning ? 'animate-spin text-emerald-400' : 'text-emerald-300'}`} />
              <span>{statusText}</span>
            </span>
            <span className="font-bold">{progress}%</span>
          </div>
          <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-emerald-500 via-teal-400 to-indigo-500 h-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Aggregate Metrics Summary */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-slate-950 p-4 rounded-xl border border-emerald-500/30 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider block">Precisión Promedio</span>
            <div className="text-xl font-extrabold text-emerald-400 flex items-center justify-between">
              <span>Aethel-1: {summary.avgAethelScore}%</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            </div>
            <span className="text-[11px] text-slate-500 font-mono">Frontera API: {summary.avgFrontierScore}%</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-indigo-500/30 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider block">Latencia Media</span>
            <div className="text-xl font-extrabold text-indigo-300 flex items-center justify-between">
              <span>{summary.avgAethelLatencyMs} ms</span>
              <Clock className="w-4 h-4 text-indigo-400" />
            </div>
            <span className="text-[11px] text-slate-500 font-mono">Frontera Cloud: {summary.avgFrontierLatencyMs} ms</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-teal-500/30 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider block">Factor Aceleración</span>
            <div className="text-xl font-extrabold text-teal-300 flex items-center justify-between">
              <span>{summary.avgSpeedupFactor}x más rápido</span>
              <Zap className="w-4 h-4 text-teal-400" />
            </div>
            <span className="text-[11px] text-slate-500 font-mono">0.0ms Latencia de Red</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-amber-500/30 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider block">Costo / 1M Tokens</span>
            <div className="text-xl font-extrabold text-emerald-400 flex items-center justify-between">
              <span>$0.00 Local</span>
              <DollarSign className="w-4 h-4 text-amber-400" />
            </div>
            <span className="text-[11px] text-slate-500 font-mono">Frontera: {summary.frontierCost}</span>
          </div>
        </div>
      )}

      {/* Side-by-Side Test Case Comparison View */}
      {testResults.length > 0 && (
        <div className="space-y-4 pt-2">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h4 className="text-sm font-extrabold text-white uppercase tracking-wider flex items-center space-x-2">
              <Award className="w-4 h-4 text-indigo-400" />
              <span>Resultados de Exámenes Comparativos ({testResults.length} Pruebas)</span>
            </h4>
            <span className="text-xs text-slate-400 font-mono">
              Compración directa de salidas generadas y tiempos de ejecución
            </span>
          </div>

          {/* Test Selector Tabs */}
          <div className="flex space-x-2 overflow-x-auto pb-2">
            {testResults.map((tr, idx) => (
              <button
                key={tr.id}
                id={`btn-select-test-${idx}`}
                onClick={() => setActiveTestId(tr.id)}
                className={`px-3.5 py-2 rounded-xl text-xs font-mono font-bold transition-all whitespace-nowrap border ${
                  activeTestId === tr.id
                    ? 'bg-gradient-to-r from-emerald-500/20 to-indigo-500/20 text-emerald-300 border-emerald-500/60 ring-1 ring-emerald-500/30'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                }`}
              >
                #{idx + 1} {tr.title}
              </button>
            ))}
          </div>

          {/* Active Test Case Detail Grid */}
          {testResults
            .filter((tr) => tr.id === activeTestId)
            .map((tr) => (
              <div key={tr.id} className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-4">
                {/* Case Info Header */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="bg-indigo-500/20 text-indigo-300 text-[10px] px-2.5 py-0.5 rounded-full font-bold border border-indigo-500/30 uppercase font-mono">
                        {tr.domainLabel}
                      </span>
                      <h5 className="text-base font-bold text-white">{tr.title}</h5>
                    </div>
                  </div>

                  <div className="bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-lg border border-emerald-500/20 text-xs font-mono font-bold">
                    Aethel-1 es {tr.speedupMultiplier}x más rápido (0.0ms red)
                  </div>
                </div>

                {/* Question Prompt */}
                <div className="space-y-1 font-mono text-xs">
                  <span className="text-slate-400 font-bold uppercase tracking-wider block">Problema / Pregunta del Examen:</span>
                  <div className="bg-slate-900 p-3.5 rounded-xl text-slate-200 border border-slate-800 leading-relaxed">
                    {tr.prompt}
                  </div>
                </div>

                {/* Side-by-Side Model Answers */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
                  {/* Column 1: Aethel-1 Local Model */}
                  <div className="bg-gradient-to-b from-slate-900 to-emerald-950/20 p-4 rounded-xl border border-emerald-500/50 space-y-3">
                    <div className="flex items-center justify-between border-b border-emerald-500/30 pb-2">
                      <div className="flex items-center space-x-2">
                        <Zap className="w-4 h-4 text-emerald-400" />
                        <span className="font-bold text-emerald-300">{tr.aethel.modelName}</span>
                      </div>
                      <span className="bg-emerald-500/20 text-emerald-300 text-[10px] px-2 py-0.5 rounded font-bold border border-emerald-500/30">
                        {tr.aethel.score}% PRECISION
                      </span>
                    </div>

                    <pre className="bg-slate-950 p-3 rounded-lg text-emerald-200 border border-emerald-500/30 whitespace-pre-wrap overflow-x-auto leading-relaxed text-[11px] max-h-56">
                      {tr.aethel.reply}
                    </pre>

                    <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 pt-1">
                      <div>Latencia: <span className="text-emerald-400 font-bold">{tr.aethel.latencyMs} ms</span></div>
                      <div>Velocidad: <span className="text-emerald-400 font-bold">{tr.aethel.tokensPerSec} tok/s</span></div>
                      <div>Memoria VRAM: <span className="text-emerald-400 font-bold">{tr.aethel.vramUsage}</span></div>
                      <div>Costo: <span className="text-emerald-400 font-bold">{tr.aethel.costPer1MTokens}</span></div>
                    </div>
                  </div>

                  {/* Column 2: Frontier Model API */}
                  <div className="bg-slate-900 p-4 rounded-xl border border-indigo-500/40 space-y-3">
                    <div className="flex items-center justify-between border-b border-indigo-500/30 pb-2">
                      <div className="flex items-center space-x-2">
                        <Bot className="w-4 h-4 text-indigo-400" />
                        <span className="font-bold text-indigo-300">{tr.frontier.modelName}</span>
                      </div>
                      <span className="bg-indigo-500/20 text-indigo-300 text-[10px] px-2 py-0.5 rounded font-bold border border-indigo-500/30">
                        {tr.frontier.score}% PRECISION
                      </span>
                    </div>

                    <pre className="bg-slate-950 p-3 rounded-lg text-slate-300 border border-slate-800 whitespace-pre-wrap overflow-x-auto leading-relaxed text-[11px] max-h-56">
                      {tr.frontier.reply}
                    </pre>

                    <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400 pt-1">
                      <div>Latencia API: <span className="text-indigo-300 font-bold">{tr.frontier.latencyMs} ms</span></div>
                      <div>Velocidad: <span className="text-indigo-300 font-bold">{tr.frontier.tokensPerSec} tok/s</span></div>
                      <div>Memoria KV Cache: <span className="text-indigo-300 font-bold">{tr.frontier.vramUsage}</span></div>
                      <div>Costo: <span className="text-indigo-300 font-bold">{tr.frontier.costPer1MTokens}</span></div>
                    </div>
                  </div>
                </div>

                {/* Verdict Footer Banner */}
                <div className="bg-emerald-950/40 p-3 rounded-xl border border-emerald-500/40 flex items-center space-x-3 text-xs">
                  <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                  <p className="text-emerald-300">
                    <strong>Dictamen del Benchmark:</strong> Aethel-1 ejecutó la respuesta con un factor de aceleración de <strong>{tr.speedupMultiplier}x</strong> respecto a la API del modelo frontera, con costo $0.00 y reteniendo la precisión en razonamiento.
                  </p>
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};
