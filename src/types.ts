/**
 * Core types for AlphaEvolve-Claude
 */

export interface Candidate {
  id: string;
  generation: number;
  code: string;
  parentIds: string[];
  mutationStrategy: MutationStrategy;
  fitness: FitnessScore | null;
  metadata: CandidateMetadata;
}

export interface FitnessScore {
  total: number;
  correctness: number; // 0 or 1 - must pass all tests
  speed: number; // normalized 0-1, higher is better
  memory: number; // normalized 0-1, higher is better (less memory)
  complexity: number; // normalized 0-1, higher is better (less complex)
  raw: {
    passedTests: number;
    totalTests: number;
    runtimeMs: number;
    memoryBytes: number;
    cyclomaticComplexity: number;
  };
}

export interface CandidateMetadata {
  createdAt: string;
  evaluatedAt: string | null;
  mutatorPrompt: string;
  linesOfCode: number;
}

export type MutationStrategy =
  | 'seed'
  | 'tweak'
  | 'restructure'
  | 'crossover'
  | 'specialize'
  | 'generalize'
  | 'alien'
  | 'hybrid'
  | 'unroll'
  | 'vectorize'
  | 'memoize'
  | 'parallelize';

export interface EvolutionConfig {
  // Parallelism
  mutatorCount: number;
  evaluatorCount: number;

  // Population
  populationSize: number;
  eliteCount: number; // Always keep top N
  diversityCount: number; // Keep N random for diversity

  // Stopping conditions
  maxGenerations: number;
  convergenceThreshold: number; // Stop after N generations without improvement
  targetFitness: number | null; // Stop if reached

  // Fitness weights
  fitnessWeights: {
    speed: number;
    memory: number;
    complexity: number;
  };

  // Paths
  outputDir: string;
  testCommand: string | null;
  benchmarkCommand: string | null;
}

export interface EvolutionState {
  config: EvolutionConfig;
  generation: number;
  candidates: Map<string, Candidate>;
  bestCandidateId: string | null;
  baselineFitness: number;
  generationsWithoutImprovement: number;
  startedAt: string;
  status: 'running' | 'converged' | 'completed' | 'failed';
}

export interface GenerationResult {
  generation: number;
  candidatesCreated: number;
  candidatesEvaluated: number;
  bestFitness: number;
  bestCandidateId: string;
  improvement: number; // percentage vs previous best
  strategies: Record<MutationStrategy, number>; // count per strategy
}

export interface EvolutionResult {
  success: boolean;
  generations: GenerationResult[];
  champion: Candidate | null;
  baselineFitness: number;
  finalFitness: number;
  improvement: number;
  totalCandidates: number;
  totalTimeMs: number;
  lineage: string[]; // ids from seed to champion
}

export interface MutatorInput {
  strategy: MutationStrategy;
  candidates: Candidate[]; // Parents to mutate/combine
  problemDescription: string;
  constraints: string[];
  targetMetrics: string[];
}

export interface EvaluatorInput {
  candidate: Candidate;
  testCommand: string | null;
  benchmarkInputs: unknown[];
  timeoutMs: number;
}

export interface BenchmarkResult {
  candidateId: string;
  runtimeMs: number;
  memoryBytes: number;
  output: unknown;
  error: string | null;
}

// Default configuration
export const DEFAULT_CONFIG: EvolutionConfig = {
  mutatorCount: 16,
  evaluatorCount: 8,
  populationSize: 50,
  eliteCount: 10,
  diversityCount: 5,
  maxGenerations: 20,
  convergenceThreshold: 5,
  targetFitness: null,
  fitnessWeights: {
    speed: 0.5,
    memory: 0.3,
    complexity: 0.2,
  },
  outputDir: '.evolve',
  testCommand: null,
  benchmarkCommand: null,
};
