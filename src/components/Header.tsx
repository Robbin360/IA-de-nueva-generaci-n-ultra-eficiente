import React from 'react';
import { TabType } from '../types';
import { Cpu, Zap, Activity, MessageSquareCode, Sparkles, Sliders, GraduationCap, Code2 } from 'lucide-react';

interface HeaderProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header id="app-header" className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 backdrop-blur-md bg-opacity-90">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Cpu className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold text-slate-100 tracking-tight">
                  Aethel Engine AI Studio
                </h1>
                <span className="bg-indigo-500/10 text-indigo-400 text-xs px-2 py-0.5 rounded-full font-medium border border-indigo-500/20">
                  Fuera de la Caja v1.0
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Diseño, Entrenamiento e Inferencia de un LLM de Nueva Generación
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav id="navigation-tabs" className="hidden lg:flex overflow-x-auto space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 max-w-2xl scrollbar-none">
            <button
              id="tab-planner"
              onClick={() => setActiveTab('planner')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'planner'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              <span>1. Arquitectura</span>
            </button>

            <button
              id="tab-trainer"
              onClick={() => setActiveTab('trainer')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'trainer'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <GraduationCap className="w-3.5 h-3.5 text-amber-400" />
              <span>2. Entrenar</span>
            </button>

            <button
              id="tab-code"
              onClick={() => setActiveTab('code')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'code'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Code2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>3. Código Fuente</span>
            </button>

            <button
              id="tab-explainer"
              onClick={() => setActiveTab('explainer')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'explainer'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>4. Conceptos</span>
            </button>

            <button
              id="tab-simulators"
              onClick={() => setActiveTab('simulators')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'simulators'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Zap className="w-3.5 h-3.5 text-rose-400" />
              <span>5. Simuladores</span>
            </button>

            <button
              id="tab-chat"
              onClick={() => setActiveTab('chat')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                activeTab === 'chat'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <MessageSquareCode className="w-3.5 h-3.5 text-emerald-400" />
              <span>6. Chat Aethel-5</span>
            </button>

            <button
              id="tab-nano1m"
              onClick={() => setActiveTab('nano1m')}
              className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition-all cursor-pointer ${
                activeTab === 'nano1m'
                  ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30'
              }`}
            >
              <Cpu className="w-4 h-4 text-emerald-300 animate-pulse" />
              <span>7. Modelo 1.2T Local Nativo</span>
            </button>
          </nav>

          {/* Quick status indicator & Direct button */}
          <div className="flex items-center space-x-2">
            <button
              id="header-quick-nano1m-btn"
              onClick={() => setActiveTab('nano1m')}
              className="flex items-center space-x-1.5 text-xs font-bold text-emerald-300 bg-emerald-500/20 hover:bg-emerald-500/30 px-3 py-1.5 rounded-xl border border-emerald-500/40 transition-all cursor-pointer shadow-md shadow-emerald-500/10"
            >
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
              <span>🔥 Probar Modelo 1.2T Nativo</span>
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <div id="mobile-navigation" className="lg:hidden flex overflow-x-auto space-x-2 py-2 border-t border-slate-800/80">
          <button
            id="mobile-tab-nano1m"
            onClick={() => setActiveTab('nano1m')}
            className={`px-3 py-1.5 rounded-md text-xs font-bold whitespace-nowrap ${
              activeTab === 'nano1m' ? 'bg-emerald-500 text-slate-950' : 'text-emerald-300 bg-emerald-500/20 border border-emerald-500/30'
            }`}
          >
            🔥 Modelo 1.2T Local
          </button>
          <button
            id="mobile-tab-planner"
            onClick={() => setActiveTab('planner')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'planner' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Plan
          </button>
          <button
            id="mobile-tab-trainer"
            onClick={() => setActiveTab('trainer')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'trainer' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Entrenar
          </button>
          <button
            id="mobile-tab-code"
            onClick={() => setActiveTab('code')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'code' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Código Fuente
          </button>
          <button
            id="mobile-tab-explainer"
            onClick={() => setActiveTab('explainer')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'explainer' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Teoría
          </button>
          <button
            id="mobile-tab-simulators"
            onClick={() => setActiveTab('simulators')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'simulators' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Simuladores
          </button>
          <button
            id="mobile-tab-chat"
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap ${
              activeTab === 'chat' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-800/40'
            }`}
          >
            Chat
          </button>
        </div>
      </div>
    </header>
  );
};

