import React, { useState, useRef, useEffect } from 'react';
import { Send, Cpu, Terminal, Zap, Menu } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Iniciando Aethel-Génesis. Conexión neuronal establecida. ¿En qué puedo ayudarte hoy?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage })
      });
      
      const data = await response.json();
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Error de conexión con el núcleo lógico (Triton/Rust).' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#121212] text-gray-100 font-sans">
      {/* Sidebar - Minimalist */}
      <aside className="w-64 bg-[#171717] border-r border-gray-800 flex flex-col hidden md:flex">
        <div className="p-4 border-b border-gray-800 flex items-center gap-3">
          <div className="bg-indigo-600 p-1.5 rounded-lg">
            <Cpu size={20} className="text-white" />
          </div>
          <span className="font-semibold text-sm tracking-wide">Aethel V3</span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2">Módulos</div>
          <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-md transition-colors">
            <Terminal size={16} /> Interfaz Consola
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-md transition-colors">
            <Zap size={16} /> Estado Triton/GPU
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative">
        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between p-4 border-b border-gray-800 bg-[#171717]">
           <div className="flex items-center gap-2">
              <Cpu size={20} className="text-indigo-500" />
              <span className="font-semibold">Aethel</span>
           </div>
           <Menu size={24} className="text-gray-400" />
        </header>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
                    <Cpu size={16} className="text-white" />
                  </div>
                )}
                
                <div className={`px-4 py-3 rounded-2xl max-w-[85%] leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-gray-800 text-gray-100 rounded-br-sm' 
                    : 'bg-transparent text-gray-300'
                }`}>
                  {msg.content}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center shrink-0">
                    <span className="text-xs font-medium">Tú</span>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-4 justify-start">
                 <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
                    <Cpu size={16} className="text-white" />
                  </div>
                  <div className="px-4 py-3 flex items-center gap-2">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-75"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-150"></div>
                  </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gradient-to-t from-[#121212] via-[#121212] to-transparent">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={handleSubmit} className="relative flex items-end gap-2 bg-gray-800 rounded-2xl border border-gray-700 p-2 shadow-lg focus-within:ring-1 focus-within:ring-indigo-500 transition-shadow">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Comunícate con Aethel V3..."
                className="w-full max-h-48 min-h-[44px] bg-transparent text-gray-100 placeholder-gray-500 resize-none outline-none py-2 px-3 flex-1 overflow-y-auto"
                rows={1}
              />
              <button 
                type="submit" 
                disabled={!input.trim() || isLoading}
                className="p-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 transition-colors shrink-0 mb-0.5"
              >
                <Send size={18} />
              </button>
            </form>
            <div className="text-center mt-3 text-xs text-gray-500">
              Aethel-Génesis se está ejecutando sobre el simulador Node.js (Esperando conexión C++/Rust nativa).
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
