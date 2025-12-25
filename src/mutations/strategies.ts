/**
 * Mutation Strategies
 *
 * Defines the different approaches for mutating candidate solutions.
 * Each strategy has a specific prompt template designed to elicit
 * particular types of improvements from the LLM.
 */

import { MutationStrategy } from '../types.js';

export const MUTATION_STRATEGIES: MutationStrategy[] = [
  'tweak',
  'restructure',
  'crossover',
  'specialize',
  'generalize',
  'alien',
  'hybrid',
  'unroll',
  'vectorize',
  'memoize',
  'parallelize',
];

interface StrategyDefinition {
  name: MutationStrategy;
  description: string;
  prompt: string;
  temperature: number; // Higher = more creative
  requiresMultipleParents: boolean;
}

export const STRATEGY_DEFINITIONS: Record<MutationStrategy, StrategyDefinition> = {
  seed: {
    name: 'seed',
    description: 'Original code (no mutation)',
    prompt: '',
    temperature: 0,
    requiresMultipleParents: false,
  },

  tweak: {
    name: 'tweak',
    description: 'Small, targeted optimizations',
    prompt: `You are an expert algorithm optimizer. Your task is to make SMALL, TARGETED improvements to the given code.

CONSTRAINTS:
- Make only 1-3 specific changes
- Preserve the overall algorithm structure
- Focus on micro-optimizations: loop bounds, conditionals, variable reuse
- Do NOT change the algorithm fundamentally

Examples of good tweaks:
- Change "i < arr.length" to cache the length
- Replace division with bit shifts where possible
- Reorder conditionals for early exit
- Reduce unnecessary allocations

CODE TO OPTIMIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the improved code, no explanations.`,
    temperature: 0.3,
    requiresMultipleParents: false,
  },

  restructure: {
    name: 'restructure',
    description: 'Algorithm-level rewrites',
    prompt: `You are an algorithm architect. Your task is to FUNDAMENTALLY RESTRUCTURE this code using a different algorithmic approach.

REQUIREMENTS:
- Use a DIFFERENT algorithm or data structure
- Maintain correctness (same inputs -> same outputs)
- Aim for better time or space complexity
- Think about: divide-and-conquer, dynamic programming, greedy, hash-based, tree-based approaches

Consider:
- Can this use a different data structure (hash map, tree, heap)?
- Can this use a different paradigm (iterative vs recursive, top-down vs bottom-up)?
- Are there preprocessing steps that would help?

CODE TO RESTRUCTURE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the restructured code, no explanations.`,
    temperature: 0.7,
    requiresMultipleParents: false,
  },

  crossover: {
    name: 'crossover',
    description: 'Combine best aspects of two solutions',
    prompt: `You are a genetic algorithm crossover operator. Your task is to COMBINE the best aspects of two different solutions into a superior hybrid.

PARENT SOLUTION 1:
\`\`\`
{{CODE_1}}
\`\`\`

PARENT SOLUTION 2:
\`\`\`
{{CODE_2}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

REQUIREMENTS:
- Identify the strengths of each parent
- Create a NEW solution that combines their best features
- The result should be better than either parent alone
- Maintain correctness

Think about:
- Which data structures from each are better?
- Which algorithmic tricks from each should be kept?
- Can techniques from both be layered together?

Return ONLY the hybrid code, no explanations.`,
    temperature: 0.5,
    requiresMultipleParents: true,
  },

  specialize: {
    name: 'specialize',
    description: 'Optimize for common cases',
    prompt: `You are a performance specialist. Your task is to SPECIALIZE this code for common cases while maintaining correctness for all cases.

REQUIREMENTS:
- Identify the most common/likely input patterns
- Add fast paths for these common cases
- Fall back to general algorithm for edge cases
- Use branch prediction hints where possible

Consider:
- What are the most frequent input sizes/patterns?
- Can we detect and fast-path small inputs?
- Are there common patterns we can check first?
- Can we use sentinel values or early termination?

CODE TO SPECIALIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the specialized code, no explanations.`,
    temperature: 0.4,
    requiresMultipleParents: false,
  },

  generalize: {
    name: 'generalize',
    description: 'Handle edge cases efficiently',
    prompt: `You are a robustness engineer. Your task is to GENERALIZE this code to handle edge cases MORE EFFICIENTLY.

REQUIREMENTS:
- Identify potential edge cases (empty inputs, single elements, duplicates, etc.)
- Handle these cases with MINIMAL overhead
- Don't add unnecessary checks that slow down common cases
- Use clever techniques to unify handling where possible

Consider:
- Can edge cases be handled by the main algorithm with small tweaks?
- Are there mathematical properties that simplify edge cases?
- Can we use sentinel values or dummy elements?

CODE TO GENERALIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the generalized code, no explanations.`,
    temperature: 0.4,
    requiresMultipleParents: false,
  },

  alien: {
    name: 'alien',
    description: 'Radically different approach',
    prompt: `You are an unconventional algorithm designer. Your task is to solve this problem in a COMPLETELY DIFFERENT and UNEXPECTED way.

REQUIREMENTS:
- Do NOT use any technique from the original code
- Think outside the box: bit manipulation, mathematical properties, exotic data structures
- Prioritize novelty while maintaining correctness
- It's okay if this is slower - we want diversity

Consider UNUSUAL approaches:
- Bit manipulation tricks
- Mathematical transformations
- Probabilistic methods (with verification)
- Finite automata or state machines
- SIMD-style operations (even in scalar code)
- Backward processing
- Compression-based techniques

ORIGINAL CODE (do NOT copy this approach):
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the alien approach code, no explanations.`,
    temperature: 0.9,
    requiresMultipleParents: false,
  },

  hybrid: {
    name: 'hybrid',
    description: 'Mix algorithmic paradigms',
    prompt: `You are an algorithm synthesist. Your task is to create a HYBRID solution that combines multiple algorithmic paradigms.

REQUIREMENTS:
- Combine at least 2 different algorithmic techniques
- Use each technique where it's most effective
- Create smooth transitions between approaches
- The whole should be greater than the sum of parts

Paradigms to consider mixing:
- Divide and conquer + dynamic programming
- Greedy + backtracking
- Hash-based + tree-based
- Iterative + recursive (each where appropriate)
- Preprocessing + online processing

CODE TO HYBRIDIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the hybrid code, no explanations.`,
    temperature: 0.6,
    requiresMultipleParents: false,
  },

  unroll: {
    name: 'unroll',
    description: 'Loop and recursion optimization',
    prompt: `You are a low-level optimizer. Your task is to UNROLL loops and optimize recursion for better performance.

TECHNIQUES:
- Loop unrolling (process multiple elements per iteration)
- Duff's device style optimizations
- Tail recursion optimization
- Convert recursion to iteration where beneficial
- Reduce loop overhead

REQUIREMENTS:
- Maintain correctness for all input sizes
- Handle remainder/edge cases from unrolling
- Don't unroll so much that code becomes unreadable
- Focus on hot paths

CODE TO UNROLL:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the unrolled code, no explanations.`,
    temperature: 0.3,
    requiresMultipleParents: false,
  },

  vectorize: {
    name: 'vectorize',
    description: 'SIMD-style parallel processing',
    prompt: `You are a vectorization expert. Your task is to restructure this code for SIMD-STYLE parallel processing.

TECHNIQUES:
- Process multiple elements simultaneously
- Use array operations instead of element-by-element
- Align data access patterns
- Reduce branches in hot loops
- Use TypedArrays where applicable (JavaScript)

REQUIREMENTS:
- Even without actual SIMD, structure code for data parallelism
- Minimize dependencies between iterations
- Use predictable memory access patterns
- Consider cache efficiency

CODE TO VECTORIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the vectorized code, no explanations.`,
    temperature: 0.4,
    requiresMultipleParents: false,
  },

  memoize: {
    name: 'memoize',
    description: 'Add intelligent caching',
    prompt: `You are a caching specialist. Your task is to add INTELLIGENT MEMOIZATION to this code.

TECHNIQUES:
- Identify repeated computations
- Design appropriate cache keys
- Choose cache data structures (Map, object, array)
- Consider cache size limits
- Think about cache invalidation

REQUIREMENTS:
- Only cache where it provides benefit
- Use memory efficiently
- Consider the space-time tradeoff
- Handle cache misses gracefully

Consider:
- What computations are repeated?
- What's the best granularity for caching?
- Can preprocessing build useful lookup tables?

CODE TO MEMOIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the memoized code, no explanations.`,
    temperature: 0.4,
    requiresMultipleParents: false,
  },

  parallelize: {
    name: 'parallelize',
    description: 'Structure for parallel execution',
    prompt: `You are a concurrency architect. Your task is to restructure this code for PARALLEL EXECUTION.

TECHNIQUES:
- Identify independent work units
- Design for map-reduce patterns
- Minimize shared state
- Use divide-and-conquer for parallel decomposition
- Consider work stealing patterns

REQUIREMENTS:
- Even without actual parallelism, structure code to be parallelizable
- Reduce dependencies between operations
- Make work units roughly equal size
- Handle combining results efficiently

Consider:
- What work can be done independently?
- How should results be merged?
- What's the optimal granularity?

CODE TO PARALLELIZE:
\`\`\`
{{CODE}}
\`\`\`

PROBLEM CONTEXT:
{{PROBLEM}}

Return ONLY the parallelized code, no explanations.`,
    temperature: 0.5,
    requiresMultipleParents: false,
  },
};

/**
 * Get the mutation prompt for a strategy
 */
export function getMutationPrompt(
  strategy: MutationStrategy,
  parentCodes: string[],
  problemDescription: string
): string {
  const def = STRATEGY_DEFINITIONS[strategy];

  if (!def.prompt) {
    return '';
  }

  let prompt = def.prompt
    .replace('{{PROBLEM}}', problemDescription)
    .replace('{{CODE}}', parentCodes[0] || '');

  // Handle crossover with multiple parents
  if (strategy === 'crossover' && parentCodes.length >= 2) {
    prompt = prompt
      .replace('{{CODE_1}}', parentCodes[0])
      .replace('{{CODE_2}}', parentCodes[1]);
  }

  return prompt;
}

/**
 * Get strategy temperature for LLM sampling
 */
export function getStrategyTemperature(strategy: MutationStrategy): number {
  return STRATEGY_DEFINITIONS[strategy].temperature;
}

/**
 * Check if strategy requires multiple parents
 */
export function requiresMultipleParents(strategy: MutationStrategy): boolean {
  return STRATEGY_DEFINITIONS[strategy].requiresMultipleParents;
}

/**
 * Get a balanced mix of strategies for a generation
 */
export function getStrategyMix(count: number): MutationStrategy[] {
  const mix: MutationStrategy[] = [];
  const strategies = MUTATION_STRATEGIES.filter((s) => s !== 'seed');

  for (let i = 0; i < count; i++) {
    mix.push(strategies[i % strategies.length]);
  }

  // Shuffle to avoid predictable ordering
  for (let i = mix.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [mix[i], mix[j]] = [mix[j], mix[i]];
  }

  return mix;
}
