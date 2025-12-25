/**
 * AlphaEvolve for Claude Code
 *
 * Evolutionary algorithm discovery system.
 * Evolves novel solutions to hard problems through massively parallel mutation and selection.
 */

export { Orchestrator, createFromFile } from './orchestrator.js';
export { Population } from './population.js';
export { Evaluator, createStringSearchEvaluator } from './evaluator.js';
export {
  getMutationPrompt,
  getStrategyTemperature,
  getStrategyMix,
  MUTATION_STRATEGIES,
  STRATEGY_DEFINITIONS,
} from './mutations/strategies.js';
export {
  generateMutationTasks,
  parseMutationResult,
  formatAsToolCalls,
  createEvolutionInstructions,
} from './claude-integration.js';
export {
  runBenchmark,
  createStringSearchSuite,
  generateTestCorpus,
  compareCandidates,
} from './benchmarks/runner.js';
export * from './types.js';

// Default export for convenience
import { Orchestrator } from './orchestrator.js';
export default Orchestrator;
