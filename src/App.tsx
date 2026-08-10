/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { TabType, ArchitectureMode, ModelHyperparameters } from './types';
import { Header } from './components/Header';
import { ArchitecturePlanner } from './components/ArchitecturePlanner';
import { TrainingEngine } from './components/TrainingEngine';
import { CodeInspector } from './components/CodeInspector';
import { ExplainerSection } from './components/ExplainerSection';
import { SimulatorsSection } from './components/SimulatorsSection';
import { ChatPlayground } from './components/ChatPlayground';
import { Nano1MEngineView } from './components/Nano1MEngineView';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('planner');
  const [architectureMode, setArchitectureMode] = useState<ArchitectureMode>('hybrid_aethel');

  // Global hyperparameters for the designed LLM (Defaults to 1.8T Total / 64B Active Ultra-Eficiente)
  const [hyperparameters, setHyperparameters] = useState<ModelHyperparameters>({
    modelName: 'Aethel-5 SS-MoE 1.8T Ultra (64B Activos)',
    hiddenDim: 32768,
    numLayers: 160,
    numExperts: 2048,
    activeExpertsPerToken: 128,
    stateDim: 512,
    quantizationBits: '1.58b',
    vocabSize: 128000,
    maxSequenceLength: 1000000,
    testTimeReasoningDepth: 128,
    enableSelfAdaptiveRouting: true,
    autoReasoningDepth: true,
    metacognitionRate: 0.999,
  });

  return (
    <div id="main-app-container" className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col justify-between selection:bg-indigo-500 selection:text-white">
      <div>
        {/* Navigation Header */}
        <Header activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Main Tab Content */}
        <main className="px-4 sm:px-6 lg:px-8 py-6">
          {activeTab === 'planner' && (
            <ArchitecturePlanner
              params={hyperparameters}
              setParams={setHyperparameters}
              onGoToTraining={() => setActiveTab('trainer')}
              onGoToCode={() => setActiveTab('code')}
            />
          )}

          {activeTab === 'trainer' && (
            <TrainingEngine
              params={hyperparameters}
              onGoToChat={() => setActiveTab('chat')}
            />
          )}

          {activeTab === 'code' && (
            <CodeInspector params={hyperparameters} />
          )}

          {activeTab === 'explainer' && (
            <ExplainerSection
              onSelectSimulator={(mode) => {
                setArchitectureMode(mode);
                setActiveTab('simulators');
              }}
            />
          )}

          {activeTab === 'simulators' && (
            <SimulatorsSection initialMode={architectureMode} />
          )}

          {activeTab === 'chat' && (
            <ChatPlayground
              architectureMode={architectureMode}
              setArchitectureMode={setArchitectureMode}
            />
          )}

          {activeTab === 'nano1m' && <Nano1MEngineView />}
        </main>
      </div>

      {/* Footer */}
      <footer id="app-footer" className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500 space-y-2">
        <p>Aethel AI Studio — Plataforma de Arquitectura e Inferencia de LLM Fuera de la Caja</p>
        <p className="text-[11px] text-slate-600 font-mono">
          State Space Memory O(1) + Sparse MoE Top-4 + BitNet 1.58b Ternario + Test-Time Search
        </p>
      </footer>
    </div>
  );
}
