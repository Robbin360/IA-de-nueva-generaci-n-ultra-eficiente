import React, { useState, useEffect } from 'react';
import { BenchmarkData } from '../types';
import { Activity, Cpu, Zap, Database, Server, Layers, Award, BarChart3, CheckCircle2, Play, RefreshCw, Sparkles, HelpCircle, Code, Binary, GraduationCap, ShieldAlert } from 'lucide-react';
import { FrontierComparisonPanel } from './FrontierComparisonPanel';

interface FrontierModelEval {
  id: string;
  name: string;
  organization: string;
  isCustom: boolean;
  scores: {
    mmluPro: number;
    humanEval: number;
    gsm8kMath: number;
    gpqaDiamond: number;
    ifEval: number;
    chatbotArenaElo: number;
  };
  vramEfficiency: string;
  inferenceSpeedTokSec: number;
  strengths: string[];
}

interface BenchmarkTestCase {
  id: string;
  benchmark: string;
  prompt: string;
  expectedOutput: string;
  aethelStatus: string;
}

export const BenchmarksSection: React.FC = () => {
  const [seqLength, setSeqLength] = useState<number>(16384);
  const [paramsBillion, setParamsBillion] = useState<number>(7);
  const [data, setData] = useState<BenchmarkData | null>(null);

  // Frontier benchmarks state
  const [frontierModels, setFrontierModels] = useState<FrontierModelEval[]>([]);
  const [testCases, setTestCases] = useState<BenchmarkTestCase[]>([]);
  const [selectedBenchmarkKey, setSelectedBenchmarkKey] = useState<keyof FrontierModelEval['scores']>('humanEval');
  const [isRunningLiveSuite, setIsRunningLiveSuite] = useState<boolean>(false);
  const [suiteProgress, setSuiteProgress] = useState<number>(0);
  const [activeTestCaseIndex, setActiveTestCaseIndex] = useState<number>(0);
  const [testLogs, setTestLogs] = useState<string>('Haz clic en "Ejecutar Batería de Benchmarks" para evaluar a nuestro modelo en tiempo real.');

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch('/api/simulate-benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequenceLength: seqLength, modelSizeBillion: paramsBillion }),
      });
      const json = await res.json();
      if (json.results) {
        setData(json.results);
      }
    } catch (err) {
      console.error('Error fetching benchmarks:', err);
    }
  };

  const fetchFrontierEvaluations = async () => {
    try {
      const res = await fetch('/api/evaluate-frontier-benchmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelName: 'Aethel-1 SS-MoE', numExperts: 16, hiddenDim: 8192 }),
      });
      const json = await res.json();
      if (json.models) {
        setFrontierModels(json.models);
      }
      if (json.testCases) {
        setTestCases(json.testCases);
      }
    } catch (err) {
      console.error('Error fetching frontier evaluations:', err);
    }
  };

  useEffect(() => {
    fetchBenchmarks();
  }, [seqLength, paramsBillion]);

  useEffect(() => {
    fetchFrontierEvaluations();
  }, []);

  // Live official benchmark evaluation state
  const [liveOfficialResults, setLiveOfficialResults] = useState<Array<{
    id: string;
    benchmarkCategory: string;
    prompt: string;
    modelReply: string;
    latencyMs: number;
    passed: boolean;
    score: number;
    expectedCriteria: string;
  }>>([]);
  const [overallLiveScore, setOverallLiveScore] = useState<number | null>(null);
  const [liveLeaderboard, setLiveLeaderboard] = useState<Array<{
    name: string;
    score: number;
    isLiveConnectedModel: boolean;
    org: string;
  }>>([]);

  const handleRunSuite = async () => {
    setIsRunningLiveSuite(true);
    setSuiteProgress(10);
    setTestLogs('Iniciando evaluación oficial en vivo con nuestro modelo conectado...');

    try {
      setSuiteProgress(30);
      setTestLogs('Enviando exámenes oficiales (HumanEval, GSM8K, MMLU-Pro, GPQA, IFEval) a la API de Inferencia...');

      const res = await fetch('/api/run-live-official-benchmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });

      setSuiteProgress(80);
      setTestLogs('Evaluando respuestas generadas contra las soluciones y criterios oficiales...');

      const json = await res.json();

      if (json.results) {
        setLiveOfficialResults(json.results);
        setOverallLiveScore(json.overallOfficialScorePercent);
        if (json.frontierLeaderboardComparison) {
          setLiveLeaderboard(json.frontierLeaderboardComparison);
        }

        // Also update local test cases view with live real responses
        const formattedTestCases: BenchmarkTestCase[] = json.results.map((r: any) => ({
          id: r.id,
          benchmark: r.benchmarkCategory,
          prompt: r.prompt,
          expectedOutput: r.modelReply,
          aethelStatus: r.passed ? `APROBADO (${(r.latencyMs / 1000).toFixed(2)}s - ${r.score} pts)` : `PARCIAL (${(r.latencyMs / 1000).toFixed(2)}s - ${r.score} pts)`,
        }));
        setTestCases(formattedTestCases);
      }

      setSuiteProgress(100);
      setTestLogs(`¡Examen Oficial Completado! Puntuación final obtenida por Aethel-1 SS-MoE: ${json.overallOfficialScorePercent}% de precisión.`);
    } catch (err: any) {
      console.error('Error al ejecutar evaluación oficial:', err);
      setTestLogs(`Error durante la evaluación oficial: ${err.message}`);
    } finally {
      setIsRunningLiveSuite(false);
    }
  };

  const benchmarkLabels: Record<keyof FrontierModelEval['scores'], { name: string; category: string; desc: string }> = {
    humanEval: {
      name: 'HumanEval (Pass@1)',
      category: 'Programación Algorítmica Python',
      desc: '164 problemas sintácticos y algorítmicos complejos evaluados mediante ejecución real con tests unitarios.',
    },
    mmluPro: {
      name: 'MMLU-Pro (Precisión %)',
      category: 'Razonamiento Académico Multidisciplinario',
      desc: 'Evaluación rigurosa de opción múltiple en 14 dominios científicos, legales y filosóficos.',
    },
    gsm8kMath: {
      name: 'GSM8K & MATH (Precisión %)',
      category: 'Razonamiento Matemático Multipaso',
      desc: 'Problemas verbales de matemática escolar y álgebra de competición que requieren cadena de pensamiento.',
    },
    gpqaDiamond: {
      name: 'GPQA Diamond (Precisión %)',
      category: 'Ciencias Avanzadas Nivel Doctorado',
      desc: 'Preguntas extremadamente difíciles de física cuántica, biología sintética y química no encontrables en Google.',
    },
    ifEval: {
      name: 'IFEval (Strict Score %)',
      category: 'Seguimiento Estricto de Instrucciones',
      desc: 'Verificación de restricciones de formato, límite de palabras, JSON válido y sintaxis sin fallos.',
    },
    chatbotArenaElo: {
      name: 'LMSYS Chatbot Arena (Rating Elo)',
      category: 'Preferencia y Utilidad Humana',
      desc: 'Puntuación basada en votación ciega a ciegas entre miles de usuarios humanos reales.',
    },
  };

  return (
    <div id="benchmarks-section-container" className="space-y-8 max-w-7xl mx-auto py-4">
      {/* Benchmark Banner */}
      <div id="benchmarks-hero" className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 p-6 md:p-8 rounded-2xl border border-indigo-500/30 shadow-xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="inline-flex items-center space-x-2 bg-indigo-500/10 text-indigo-400 text-xs px-3 py-1 rounded-full font-semibold border border-indigo-500/20">
              <Award className="w-3.5 h-3.5" />
              <span>Paso 6: Comparativa Puntual de Benchmarks Frontera</span>
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
              Puntuación de Benchmarks Oficiales e Inferencia en Vivo
            </h2>
            <p className="text-slate-300 text-sm leading-relaxed">
              Compara de forma transparente cuántos puntos obtiene nuestro modelo <strong>Aethel-1 SS-MoE</strong> contra los gigantes de la industria (<strong>GPT-4o/5.6</strong>, <strong>Claude 3.5 Sonnet</strong>, <strong>DeepSeek V3</strong>, <strong>Gemini 1.5 Pro</strong>) en <strong>MMLU-Pro</strong>, <strong>HumanEval</strong>, <strong>GSM8K</strong>, <strong>GPQA</strong> e <strong>IFEval</strong>.
            </p>
          </div>

          <button
            id="btn-run-benchmark-suite"
            onClick={handleRunSuite}
            disabled={isRunningLiveSuite}
            className="flex items-center space-x-2 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-extrabold px-5 py-3 rounded-xl text-xs transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50 whitespace-nowrap cursor-pointer"
          >
            <Play className={`w-4 h-4 fill-current ${isRunningLiveSuite ? 'animate-spin' : ''}`} />
            <span>{isRunningLiveSuite ? 'Ejecutando Test Batería...' : 'Ejecutar Batería de Benchmarks'}</span>
          </button>
        </div>
      </div>

      {/* Automated Multi-Model Comparison Panel (Math, Logic, Code) */}
      <FrontierComparisonPanel />

      {/* Test Execution Progress Log */}
      {(isRunningLiveSuite || suiteProgress > 0) && (
        <div className="bg-slate-900 p-4 rounded-2xl border border-emerald-500/30 shadow-xl space-y-2 font-mono text-xs">
          <div className="flex items-center justify-between text-emerald-400">
            <span className="flex items-center space-x-2">
              <RefreshCw className={`w-4 h-4 ${isRunningLiveSuite ? 'animate-spin text-emerald-400' : 'text-emerald-300'}`} />
              <span>{testLogs}</span>
            </span>
            <span className="font-bold">{suiteProgress}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
            <div className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full transition-all duration-300" style={{ width: `${suiteProgress}%` }} />
          </div>
        </div>
      )}

      {/* Live Official Benchmark Scores Banner (When evaluated) */}
      {overallLiveScore !== null && liveLeaderboard.length > 0 && (
        <div id="live-official-results-banner" className="bg-gradient-to-br from-slate-900 via-emerald-950/40 to-slate-900 p-6 rounded-2xl border-2 border-emerald-500/60 shadow-2xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-emerald-500/30">
            <div className="space-y-1">
              <span className="inline-flex items-center space-x-1.5 text-xs font-extrabold text-emerald-400 uppercase tracking-wider bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Resultados en Vivo del Modelo Conectado</span>
              </span>
              <h3 className="text-xl font-bold text-white">
                Puntuación Oficial Promedio: <span className="text-emerald-400 text-2xl font-black">{overallLiveScore}%</span>
              </h3>
              <p className="text-xs text-slate-300">
                Respondiendo en tiempo real mediante la API de Inferencia configurada para Aethel-1 SS-MoE.
              </p>
            </div>

            <div className="bg-slate-950/80 p-3 rounded-xl border border-emerald-500/40 text-xs font-mono space-y-1">
              <div className="text-slate-400">Modelo Evaluado: <span className="text-emerald-300 font-bold">Aethel-1 SS-MoE</span></div>
              <div className="text-slate-400">Total Batería: <span className="text-white font-bold">5 Exámenes Oficiales</span></div>
              <div className="text-slate-400">Estado de API: <span className="text-emerald-400 font-bold">En Línea (Gemini 2.5)</span></div>
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Tabla Comparativa en Vivo (Puntos Obtenidos)</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {liveLeaderboard.map((item, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-xl border flex items-center justify-between transition-all ${
                    item.isLiveConnectedModel
                      ? 'bg-emerald-950/60 border-emerald-500 text-emerald-300 ring-1 ring-emerald-500/40 font-bold'
                      : 'bg-slate-950/80 border-slate-800 text-slate-300'
                  }`}
                >
                  <div className="space-y-0.5">
                    <div className="text-xs font-bold flex items-center space-x-1">
                      <span>{item.name}</span>
                      {item.isLiveConnectedModel && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 inline" />}
                    </div>
                    <div className="text-[10px] text-slate-500">{item.org}</div>
                  </div>
                  <div className="text-sm font-black font-mono">
                    {item.score}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Benchmark Category Selector Tabs */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            <span>Seleccionar Suite de Evaluación Frontera</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">Puntuaciones estandarizadas en 0-100% / Elo</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          {(Object.keys(benchmarkLabels) as Array<keyof FrontierModelEval['scores']>).map((key) => {
            const isSelected = selectedBenchmarkKey === key;
            return (
              <button
                key={key}
                id={`tab-benchmark-${key}`}
                onClick={() => setSelectedBenchmarkKey(key)}
                className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between space-y-1 ${
                  isSelected
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 font-bold ring-1 ring-indigo-500/30'
                    : 'bg-slate-950 border-slate-800 hover:border-slate-700 text-slate-400'
                }`}
              >
                <span className="text-xs font-bold">{benchmarkLabels[key].name.split(' ')[0]}</span>
                <span className="text-[10px] text-slate-500 line-clamp-1">{benchmarkLabels[key].category}</span>
              </button>
            );
          })}
        </div>

        {/* Selected Benchmark Detail Description */}
        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-1">
          <div className="text-xs font-bold text-slate-200">
            {benchmarkLabels[selectedBenchmarkKey].name} — {benchmarkLabels[selectedBenchmarkKey].category}
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            {benchmarkLabels[selectedBenchmarkKey].desc}
          </p>
        </div>

        {/* Model Points Leaderboard Bar Visualizer */}
        <div className="space-y-4">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Ranking y Puntaje por Modelo</h4>
          <div className="space-y-3">
            {frontierModels
              .slice()
              .sort((a, b) => b.scores[selectedBenchmarkKey] - a.scores[selectedBenchmarkKey])
              .map((m, idx) => {
                const score = m.scores[selectedBenchmarkKey];
                const isElo = selectedBenchmarkKey === 'chatbotArenaElo';
                const maxVal = isElo ? 1450 : 100;
                const minVal = isElo ? 1200 : 0;
                const percentageWidth = isElo
                  ? Math.max(10, ((score - minVal) / (maxVal - minVal)) * 100)
                  : score;

                return (
                  <div
                    key={m.id}
                    className={`p-3.5 rounded-xl border transition-all space-y-2 ${
                      m.isCustom
                        ? 'bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border-emerald-500/60 ring-1 ring-emerald-500/30'
                        : 'bg-slate-950 border-slate-800/80'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-slate-500 w-5">#{idx + 1}</span>
                        <span className={`font-bold ${m.isCustom ? 'text-emerald-300' : 'text-slate-200'}`}>
                          {m.name}
                        </span>
                        {m.isCustom && (
                          <span className="bg-emerald-500/20 text-emerald-300 text-[10px] px-2 py-0.5 rounded-full font-bold border border-emerald-500/30">
                            Nuestro LLM
                          </span>
                        )}
                        <span className="text-[11px] text-slate-500">({m.organization})</span>
                      </div>

                      <div className="flex items-center space-x-3 font-mono">
                        <span className="text-slate-400 text-[11px] hidden sm:inline">VRAM: {m.vramEfficiency}</span>
                        <span className={`font-extrabold text-sm ${m.isCustom ? 'text-emerald-400' : 'text-indigo-300'}`}>
                          {score} {isElo ? 'Elo' : '%'}
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800/80">
                      <div
                        className={`h-full transition-all duration-500 ${
                          m.isCustom
                            ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                            : idx === 0
                            ? 'bg-indigo-500'
                            : 'bg-slate-700'
                        }`}
                        style={{ width: `${percentageWidth}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>

      {/* Live Benchmark Exam Simulator Test Cases */}
      {testCases.length > 0 && (
        <div id="live-exam-simulator-panel" className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div className="space-y-1">
              <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                <Code className="w-5 h-5 text-emerald-400" />
                <span>Simulador de Examen de Benchmarks en Vivo (Muestras Reales)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Inspecciona cómo nuestro modelo responde a problemas reales extraídos de HumanEval, GSM8K y GPQA Diamond.
              </p>
            </div>

            <div className="flex space-x-2">
              {testCases.map((tc, idx) => (
                <button
                  key={tc.id}
                  id={`btn-tc-select-${idx}`}
                  onClick={() => setActiveTestCaseIndex(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold border transition-all ${
                    activeTestCaseIndex === idx
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  Test #{idx + 1}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between text-indigo-300 border-b border-slate-800/60 pb-2">
              <span className="font-bold">{testCases[activeTestCaseIndex].benchmark}</span>
              <span className="bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20 text-[10px]">
                {testCases[activeTestCaseIndex].aethelStatus}
              </span>
            </div>

            <div className="space-y-1">
              <span className="text-slate-500 text-[11px] uppercase tracking-wider block">Pregunta / Prompt del Examen:</span>
              <div className="bg-slate-900 p-3 rounded-lg text-slate-200 border border-slate-800">
                {testCases[activeTestCaseIndex].prompt}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-slate-500 text-[11px] uppercase tracking-wider block">Respuesta Generada y Verificada por Aethel-1:</span>
              <pre className="bg-slate-900 p-3 rounded-lg text-emerald-300 border border-slate-800 whitespace-pre-wrap overflow-x-auto">
                {testCases[activeTestCaseIndex].expectedOutput}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Computational VRAM Efficiency Calculator Sliders */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            <span>Calculadora de Consumo de Memoria VRAM y Escala</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">Comportamiento Asintótico O(N²) vs O(1)</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-slate-300">Longitud de Contexto ($N$ Tokens)</span>
              <span className="font-mono text-cyan-400 font-bold">{seqLength.toLocaleString()} tokens</span>
            </div>
            <input
              id="slider-seq-len"
              type="range"
              min="2048"
              max="131072"
              step="2048"
              value={seqLength}
              onChange={(e) => setSeqLength(Number(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-950 rounded-lg cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="font-semibold text-slate-300">Escala del Modelo (Miles de Millones de Parámetros)</span>
              <span className="font-mono text-indigo-400 font-bold">{paramsBillion}B params</span>
            </div>
            <input
              id="slider-params-billion"
              type="range"
              min="1"
              max="70"
              step="1"
              value={paramsBillion}
              onChange={(e) => setParamsBillion(Number(e.target.value))}
              className="w-full accent-indigo-500 bg-slate-950 rounded-lg cursor-pointer"
            />
          </div>
        </div>

        {/* Results Comparison Grid */}
        {data && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
            {/* Transformer Standard */}
            <div className="bg-slate-950 p-5 rounded-2xl border border-rose-500/30 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-rose-400 uppercase tracking-wider">Transformer Estándar</span>
                <span className="text-[10px] font-mono bg-rose-500/10 text-rose-400 px-2 py-0.5 rounded border border-rose-500/20">Attention O(N²)</span>
              </div>

              <div className="space-y-3 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Memoria KV Cache:</span>
                  <span className="text-rose-400 font-bold">{data.transformer.kvCacheMemoryMb.toLocaleString()} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">GFLOPS / Token:</span>
                  <span className="text-slate-200">{data.transformer.flopsPerTokenG} GFLOPS</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Velocidad Inferencia:</span>
                  <span className="text-slate-200">{data.transformer.inferenceSpeedTokensSec} tok/s</span>
                </div>
              </div>
            </div>

            {/* Mamba SSM */}
            <div className="bg-slate-950 p-5 rounded-2xl border border-indigo-500/30 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Mamba SSM</span>
                <span className="text-[10px] font-mono bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded border border-indigo-500/20">State Space O(1)</span>
              </div>

              <div className="space-y-3 font-mono text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Memoria KV Cache:</span>
                  <span className="text-emerald-400 font-bold">{data.ssmMamba.kvCacheMemoryMb} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">GFLOPS / Token:</span>
                  <span className="text-slate-200">{data.ssmMamba.flopsPerTokenG} GFLOPS</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Velocidad Inferencia:</span>
                  <span className="text-slate-200">{data.ssmMamba.inferenceSpeedTokensSec} tok/s</span>
                </div>
              </div>
            </div>

            {/* Aethel Hybrid SS-MoE */}
            {data.hybridAethel && (
              <div className="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 p-5 rounded-2xl border-2 border-emerald-500/60 shadow-xl space-y-4 col-span-1 md:col-span-2 lg:col-span-1 ring-1 ring-emerald-500/30">
                <div className="flex items-center justify-between pb-2 border-b border-emerald-500/30">
                  <span className="text-xs font-extrabold text-emerald-400 uppercase tracking-wider flex items-center space-x-1.5">
                    <Zap className="w-4 h-4 text-emerald-400" />
                    <span>Aethel SS-MoE (Nuestro Diseño)</span>
                  </span>
                  <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full font-bold border border-emerald-500/40">
                    Súper Eficiente
                  </span>
                </div>

                <div className="space-y-3 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Memoria KV Cache:</span>
                    <span className="text-emerald-300 font-bold">{data.hybridAethel.kvCacheMemoryMb} MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">GFLOPS / Token:</span>
                    <span className="text-emerald-300 font-bold">{data.hybridAethel.flopsPerTokenG} GFLOPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Velocidad Inferencia:</span>
                    <span className="text-emerald-300 font-bold">{data.hybridAethel.inferenceSpeedTokensSec} tok/s</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

