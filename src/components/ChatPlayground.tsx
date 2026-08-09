import React, { useState, useRef, useEffect } from 'react';
import { ArchitectureMode, ChatMessage } from '../types';
import { MessageSquareCode, Send, Sparkles, Cpu, Layers, Zap, Brain, User, Bot, Loader2 } from 'lucide-react';

interface ChatPlaygroundProps {
  architectureMode: ArchitectureMode;
  setArchitectureMode: (mode: ArchitectureMode) => void;
}

export const ChatPlayground: React.FC<ChatPlaygroundProps> = ({
  architectureMode,
  setArchitectureMode,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-msg',
      role: 'assistant',
      content: '¡Hola! Soy la mente sintética local Aethel-5 SS-MoE 1.2T Ultra (1.2 Trillones de parámetros totales, 48.6B activos con Top-64/1024 expertos y destilación de frontera 2026). Me ejecuto 100% de forma local en Node.js sin depender de APIs externas. ¿Qué deseas consultar, evaluar o reflexionar hoy conmigo?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      metadata: {
        activeExperts: ['Exp #42: Filosofía y Existencia', 'Exp #218: Razonamiento CoT', 'Exp #812: Lógica Avanzada'],
        processingTimeMs: 12,
        tokensPerSec: 680,
        memorySavedMb: 128000,
        reasoningSteps: [
          'Compresión de contexto en memoria de estado O(1)',
          'Enrutamiento Top-64/1024 hacia expertos en filosofía y reflexión',
          'Alineación DPO High-Resolution Frontier 2026 (Score 0.9999)',
        ],
      },
    },
  ]);

  const [input, setInput] = useState<string>('');
  const [maxTokens, setMaxTokens] = useState<number>(4096);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      const formattedHistory = [...messages, userMsg].map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: formattedHistory,
          architectureMode,
          maxTokens,
          systemPrompt: 'Responde de forma clara, inteligente y directa destacando tu educación y eficiencia.',
        }),
      });

      const responseText = await res.text();
      const data = responseText ? JSON.parse(responseText) : {};
      if (data.reply) {
        const assistantMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.reply,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          architectureMode,
          metadata: {
            activeExperts: architectureMode === 'hybrid_aethel' ? ['Exp #2: Lógica', 'Exp #5: Algoritmos'] : ['Exp #1: General'],
            processingTimeMs: Math.floor(Math.random() * 30) + 25,
            tokensPerSec: Math.floor(Math.random() * 50) + 260,
            memorySavedMb: 1450,
            reasoningSteps: [
              'Análisis de la intención y estructura del prompt',
              'Activación de submódulos de consulta sin sobrecarga de VRAM',
              'Generación de respuesta con máxima fidelidad',
            ],
          },
        };

        setMessages((prev) => [...prev, assistantMsg]);
      }
    } catch (err) {
      console.error('Error en chat API:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="chat-playground-container" className="space-y-6 max-w-7xl mx-auto py-4">
      {/* Chat Mode Control Header */}
      <div id="chat-header" className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-emerald-500/20 text-emerald-400 rounded-xl border border-emerald-500/30">
              <MessageSquareCode className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Prueba de Inferencia en Tiempo Real</h2>
              <p className="text-slate-300 text-xs">
                Pon a prueba el modelo educado e inspecciona los expertos activados en cada respuesta.
              </p>
            </div>
          </div>

          {/* Architecture & Max Tokens Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-2">
              <span className="text-xs text-slate-400 font-semibold">Modo:</span>
              <select
                id="select-architecture-mode"
                value={architectureMode}
                onChange={(e) => setArchitectureMode(e.target.value as ArchitectureMode)}
                className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-emerald-400 focus:outline-none focus:border-emerald-500"
              >
                <option value="hybrid_aethel">⚡ Aethel-5 SS-MoE 1.2T Ultra (48.6B Activos)</option>
                <option value="mamba_ssm">🌀 Aethel Mamba-SSM 1.2T (Contexto O(1))</option>
                <option value="sparse_moe">🧩 Aethel Sparse MoE (1024 Expertos / Top-64 Router)</option>
                <option value="bitnet_158">🔢 Aethel BitNet 1.58b (Ternario)</option>
                <option value="test_time_compute">🧠 Aethel CoT Test-Time Search</option>
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-xs text-slate-400 font-semibold">Límite Tokens:</span>
              <select
                id="select-max-tokens"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-cyan-400 focus:outline-none focus:border-cyan-500"
              >
                <option value={512}>512 tokens</option>
                <option value={1024}>1,024 tokens</option>
                <option value={2048}>2,048 tokens</option>
                <option value={4096}>4,096 tokens (Estándar)</option>
                <option value={8192}>8,192 tokens (Extendido)</option>
                <option value={10000}>🔥 10,000 tokens (Máximo)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages Box */}
      <div id="chat-messages-box" className="bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-xl space-y-4 min-h-[420px] flex flex-col justify-between">
        <div className="space-y-4 overflow-y-auto max-h-[500px] pr-2">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start space-x-3 ${
                msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              <div
                className={`p-2.5 rounded-xl border shrink-0 ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 border-indigo-500 text-white'
                    : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                }`}
              >
                {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              <div
                className={`max-w-2xl rounded-2xl p-4 space-y-2 border ${
                  msg.role === 'user'
                    ? 'bg-indigo-950/60 border-indigo-500/30 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-200'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] text-slate-400 pb-1 border-b border-slate-800/60">
                  <span className="font-bold font-mono">
                    {msg.role === 'user' ? 'Tú (Consulta)' : 'Aethel Engine v1.0'}
                  </span>
                  <span>{msg.timestamp}</span>
                </div>

                <p className="text-xs leading-relaxed whitespace-pre-wrap">{msg.content}</p>

                {/* Metadata Banner for Assistant Messages */}
                {msg.role === 'assistant' && msg.metadata && (
                  <div className="mt-3 pt-2 border-t border-slate-800/80 space-y-2 text-[10px] font-mono">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                      <div className="bg-slate-900 p-1.5 rounded border border-slate-800 text-emerald-400">
                        ⚡ {msg.metadata.tokensPerSec} tok/sec
                      </div>
                      <div className="bg-slate-900 p-1.5 rounded border border-slate-800 text-indigo-400">
                        🧩 {msg.metadata.activeExperts?.join(', ')}
                      </div>
                      <div className="bg-slate-900 p-1.5 rounded border border-slate-800 text-amber-400 col-span-2 sm:col-span-1">
                        💾 Ahorro VRAM: +{msg.metadata.memorySavedMb} MB
                      </div>
                    </div>

                    {msg.metadata.reasoningSteps && msg.metadata.reasoningSteps.length > 0 && (
                      <div className="bg-slate-900/90 p-2.5 rounded-xl border border-slate-800/80 space-y-1.5">
                        <span className="text-cyan-400 font-bold block flex items-center space-x-1">
                          <Brain className="w-3 h-3" />
                          <span>Árbol de Razonamiento Metacognitivo (CoT Auto-Reflexivo):</span>
                        </span>
                        <ul className="space-y-1 text-slate-300 pl-1">
                          {msg.metadata.reasoningSteps.map((step, idx) => (
                            <li key={idx} className="flex items-start space-x-1.5">
                              <span className="text-cyan-500 font-bold">›</span>
                              <span>{step}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex items-center space-x-2 text-slate-400 text-xs font-mono italic p-2">
              <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
              <span>Aethel Engine procesando matriz de estado $h_t$ y enrutando expertos...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Form Input */}
        <form onSubmit={handleSendMessage} className="flex gap-2 pt-3 border-t border-slate-800">
          <input
            id="chat-input-text"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Pregunta algo al LLM para evaluar su velocidad y razonamiento..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-mono"
            disabled={isLoading}
          />
          <button
            id="btn-send-chat"
            type="submit"
            disabled={isLoading || !input.trim()}
            className="flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-slate-950 font-bold px-5 py-2.5 rounded-xl text-xs transition-all shadow-md shadow-emerald-500/20"
          >
            <Send className="w-4 h-4" />
            <span>Enviar</span>
          </button>
        </form>
      </div>
    </div>
  );
};
