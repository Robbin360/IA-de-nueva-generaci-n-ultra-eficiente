export type TabType = 'planner' | 'trainer' | 'code' | 'explainer' | 'simulators' | 'benchmarks' | 'chat' | 'nano1m' | 'lab' | 'nova';

export type ArchitectureMode = 'standard' | 'mamba_ssm' | 'sparse_moe' | 'bitnet_158' | 'test_time_compute' | 'spiking_nn' | 'hybrid_aethel';

export interface ModelHyperparameters {
  modelName: string;
  hiddenDim: number;
  numLayers: number;
  numExperts: number;
  activeExpertsPerToken: number;
  stateDim: number; // SSM state size d_state
  quantizationBits: '1.58b' | '4b' | '8b' | '16b';
  vocabSize: number;
  maxSequenceLength: number;
  testTimeReasoningDepth: number;
  enableSelfAdaptiveRouting?: boolean;
  autoReasoningDepth?: boolean;
  metacognitionRate?: number;
}

export interface TrainingLogPoint {
  step: number;
  epoch: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradientNorm: number;
  expertLoads: number[]; // Percentage load across experts 0..N
  tokensProcessed: number;
}

export interface TrainingConfig {
  corpusName: string;
  corpusContent: string;
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'AdamW' | 'Muon' | 'Lion';
  teacherSupervision: boolean; // Teacher model distillation
  targetPerplexity: number;
}

export interface ArchitectureConcept {
  id: ArchitectureMode;
  name: string;
  tagline: string;
  badge: string;
  color: string;
  complexity: string;
  memoryEfficiency: string;
  computeEfficiency: string;
  keyInnovation: string;
  howItWorks: string[];
  pros: string[];
  cons: string[];
  mathFormula: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  architectureMode?: ArchitectureMode;
  metadata?: {
    activeExperts?: string[];
    processingTimeMs?: number;
    tokensPerSec?: number;
    memorySavedMb?: number;
    reasoningSteps?: string[];
    trainedCheckpoints?: string;
    selfCorrectionScore?: number;
    adaptiveContextLength?: number;
    totalParamsActive?: string;
  };
}

export interface BenchmarkData {
  transformer: {
    name: string;
    kvCacheMemoryMb: number;
    flopsPerTokenG: number;
    energyJoulesPer1000Tokens: number;
    inferenceSpeedTokensSec: number;
  };
  ssmMamba: {
    name: string;
    kvCacheMemoryMb: number;
    flopsPerTokenG: number;
    energyJoulesPer1000Tokens: number;
    inferenceSpeedTokensSec: number;
  };
  sparseMoe: {
    name: string;
    kvCacheMemoryMb: number;
    flopsPerTokenG: number;
    energyJoulesPer1000Tokens: number;
    inferenceSpeedTokensSec: number;
  };
  bitNet: {
    name: string;
    kvCacheMemoryMb: number;
    flopsPerTokenG: number;
    energyJoulesPer1000Tokens: number;
    inferenceSpeedTokensSec: number;
  };
  hybridAethel?: {
    name: string;
    kvCacheMemoryMb: number;
    flopsPerTokenG: number;
    energyJoulesPer1000Tokens: number;
    inferenceSpeedTokensSec: number;
  };
}

export interface OpenDatasetInfo {
  id: string;
  name: string;
  organization: string;
  tokenCount: string;
  qualityRating: number;
  license: string;
  domain: string;
  description: string;
  sampleText: string;
  isPopular?: boolean;
}

export interface FrontierModelBenchmark {
  modelName: string;
  organization: string;
  isCustomModel?: boolean;
  mmluPro: number;
  humanEval: number;
  gsm8kMath: number;
  lmsysElo: number;
  vramMemoryEfficiency: string;
  inferenceSpeedTokSec: number;
}

export interface AethelLabVersion {
  versionId: string;
  versionName: string;
  date: string;
  status: 'baseline' | 'active' | 'archived';
  mmluPro: number;
  gpqa: number;
  math: number;
  liveCodeBench: number;
  sweBench: number;
  ifEval: number;
  overallScore: number;
  notes: string;
}

export interface DatasetDomainInfo {
  id: string;
  title: string;
  iconName: string;
  tokensCount: string;
  qualityScore: number;
  description: string;
  examples: string[];
}

export interface GRPORewardSample {
  prompt: string;
  generatedTrajectories: {
    id: string;
    text: string;
    isCorrect: boolean;
    reward: number; // +1.0 or 0.0
    feedback: string;
  }[];
  advantageScore: number;
}

export interface SandboxExecutionResult {
  code: string;
  language: 'python' | 'typescript';
  status: 'SUCCESS' | 'SYNTAX_ERROR' | 'TEST_FAILED';
  stdout: string;
  stderr?: string;
  executionTimeMs: number;
  autoRepairedCode?: string;
  passRate: number;
}

export interface AgentToolResult {
  step: number;
  observation: string;
  thought: string;
  action: string;
  toolUsed: 'terminal' | 'python_exec' | 'browser' | 'filesystem' | 'api_call';
  result: string;
  status: 'success' | 'failed';
}

export interface MoERouterStats {
  routingAccuracy: number; // e.g., 98.4%
  loadBalancingEntropy: number;
  expertUtilizationPercentage: number[];
  topExpertDomains: { expertId: number; domain: string; usage: number }[];
}


