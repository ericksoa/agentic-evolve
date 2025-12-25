/**
 * Claude Agent Integration
 *
 * This module provides the bridge between AlphaEvolve and Claude Code.
 * It generates the prompts and instructions that Claude agents use
 * to perform mutations.
 *
 * When running inside Claude Code:
 * - The orchestrator calls this module to generate mutation tasks
 * - Each task becomes a call to Claude Code's Task tool
 * - Multiple tasks run in parallel as background agents
 * - Results are collected and added to the population
 */

import { Candidate, MutationStrategy, EvolutionConfig } from './types.js';
import { getMutationPrompt, getStrategyTemperature, STRATEGY_DEFINITIONS } from './mutations/strategies.js';

export interface ClaudeTaskSpec {
  description: string;
  prompt: string;
  subagent_type: string;
  model: 'sonnet' | 'opus' | 'haiku';
  run_in_background: boolean;
}

/**
 * Generate Claude Code Task specifications for a batch of mutations
 */
export function generateMutationTasks(
  parents: Candidate[],
  strategies: MutationStrategy[],
  problemDescription: string,
  _config: EvolutionConfig
): ClaudeTaskSpec[] {
  const tasks: ClaudeTaskSpec[] = [];

  for (let i = 0; i < strategies.length; i++) {
    const strategy = strategies[i];
    const parentIndex = i % parents.length;
    const parent = parents[parentIndex];

    // For crossover, get a second parent
    const secondParent = strategy === 'crossover'
      ? parents[(parentIndex + 1) % parents.length]
      : null;

    const parentCodes = secondParent
      ? [parent.code, secondParent.code]
      : [parent.code];

    const mutationPrompt = getMutationPrompt(strategy, parentCodes, problemDescription);
    const temperature = getStrategyTemperature(strategy);

    // Choose model based on strategy complexity
    // More creative strategies use more capable models
    const model: 'sonnet' | 'opus' | 'haiku' = temperature > 0.7 ? 'opus' : temperature > 0.4 ? 'sonnet' : 'haiku';

    tasks.push({
      description: `Mutate with ${strategy}`,
      prompt: buildAgentPrompt(strategy, mutationPrompt, parent.id),
      subagent_type: 'general-purpose',
      model,
      run_in_background: true,
    });
  }

  return tasks;
}

/**
 * Build the full prompt for a mutation agent
 */
function buildAgentPrompt(
  strategy: MutationStrategy,
  mutationPrompt: string,
  parentId: string
): string {
  return `You are an algorithm evolution agent. Your task is to generate a SINGLE improved variant of the given code.

MUTATION STRATEGY: ${strategy}
${STRATEGY_DEFINITIONS[strategy].description}

${mutationPrompt}

IMPORTANT RULES:
1. Return ONLY valid, runnable code
2. The code must be a complete, self-contained function
3. Do NOT include explanations, comments about changes, or markdown
4. Do NOT change the function signature
5. Preserve correctness - the function must return the same results for the same inputs
6. Focus on PERFORMANCE improvements

Your output will be directly used as code, so output NOTHING except the improved code.

Parent ID for reference: ${parentId}`;
}

/**
 * Parse the result from a Claude mutation agent
 */
export function parseMutationResult(
  agentOutput: string,
  _strategy: MutationStrategy,
  _parentIds: string[],
  _generation: number
): { code: string; valid: boolean; error?: string } {
  // Clean up the output
  let code = agentOutput.trim();

  // Remove markdown code blocks if present
  code = code.replace(/^```(?:typescript|javascript|ts|js)?\n?/gm, '');
  code = code.replace(/```$/gm, '');
  code = code.trim();

  // Basic validation
  if (!code) {
    return { code: '', valid: false, error: 'Empty output' };
  }

  // Check for function definition
  const hasFunctionDef =
    code.includes('function ') ||
    code.includes('=>') ||
    code.includes('export ');

  if (!hasFunctionDef) {
    return { code, valid: false, error: 'No function definition found' };
  }

  // Check for obvious errors
  const suspiciousPatterns = [
    /undefined(?!\s*[;,\)])/,  // undefined not as a value
    /TODO/,
    /FIXME/,
    /\.\.\.(?![\w])/,  // spread operator is ok, ellipsis not
  ];

  for (const pattern of suspiciousPatterns) {
    if (pattern.test(code)) {
      return { code, valid: false, error: `Suspicious pattern found` };
    }
  }

  return { code, valid: true };
}

/**
 * Format mutation tasks as Claude Code tool calls
 * Returns a string representation for documentation/debugging
 */
export function formatAsToolCalls(tasks: ClaudeTaskSpec[]): string {
  const calls = tasks.map(task => {
    return `Task(
  description: "${task.description}",
  prompt: "...",
  subagent_type: "${task.subagent_type}",
  model: "${task.model}",
  run_in_background: ${task.run_in_background}
)`;
  });

  return calls.join('\n\n');
}

/**
 * Create the evolution loop that runs inside Claude Code
 * This returns the instructions that go in the skill file
 */
export function createEvolutionInstructions(): string {
  return `
EVOLUTION LOOP INSTRUCTIONS:

1. INITIALIZATION:
   - Read the target code file specified by the user
   - Create the .evolve/ directory structure
   - Evaluate baseline fitness by running tests

2. FOR EACH GENERATION:

   a. SELECT PARENTS:
      - Get top 10 candidates by fitness
      - Add 5 random candidates for diversity

   b. SPAWN MUTATORS (use Task tool with run_in_background=true):
      - Create 16 parallel mutation tasks
      - Distribute strategies: tweak, restructure, crossover, specialize,
        generalize, alien, hybrid, unroll, vectorize, memoize, parallelize
      - Each mutator receives one parent and its strategy prompt
      - Use haiku for simple mutations, sonnet for moderate, opus for creative

   c. COLLECT RESULTS:
      - Use TaskOutput to wait for all mutators
      - Parse each result, extract the mutated code
      - Validate syntax and function signature

   d. EVALUATE:
      - Run tests on each valid candidate
      - Benchmark performance (runtime, memory)
      - Calculate fitness scores

   e. UPDATE POPULATION:
      - Add successful candidates to population
      - Prune to population size limit
      - Keep elites and some random for diversity

   f. CHECK CONVERGENCE:
      - If no improvement for N generations, stop
      - If target fitness reached, stop
      - If max generations reached, stop

3. FINALIZATION:
   - Write champion code to target file
   - Save evolution report
   - Display results to user
`;
}
