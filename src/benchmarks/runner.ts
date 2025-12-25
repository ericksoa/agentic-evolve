/**
 * Benchmark Runner
 *
 * Executes performance benchmarks for candidate evaluation.
 * Provides consistent, reliable measurements across runs.
 */

import { BenchmarkResult, Candidate } from '../types.js';

export interface BenchmarkConfig {
  warmupRuns: number;
  measuredRuns: number;
  timeoutMs: number;
  inputGenerator: () => unknown[];
}

export interface BenchmarkSuite {
  name: string;
  inputs: unknown[];
  expectedOutputs?: unknown[];
  validate?: (input: unknown, output: unknown) => boolean;
}

/**
 * Run a benchmark suite on a candidate
 */
export async function runBenchmark(
  candidate: Candidate,
  suite: BenchmarkSuite,
  config: Partial<BenchmarkConfig> = {}
): Promise<BenchmarkResult> {
  const fullConfig: BenchmarkConfig = {
    warmupRuns: 3,
    measuredRuns: 10,
    timeoutMs: 30000,
    inputGenerator: () => suite.inputs,
    ...config,
  };

  try {
    // Create a function from the candidate code
    const fn = createFunctionFromCode(candidate.code);

    // Warmup runs
    for (let i = 0; i < fullConfig.warmupRuns; i++) {
      for (const input of suite.inputs) {
        fn(input);
      }
    }

    // Measured runs
    const times: number[] = [];
    const memoryBefore = process.memoryUsage().heapUsed;

    for (let i = 0; i < fullConfig.measuredRuns; i++) {
      const startTime = performance.now();
      for (const input of suite.inputs) {
        fn(input);
      }
      times.push(performance.now() - startTime);
    }

    const memoryAfter = process.memoryUsage().heapUsed;

    // Calculate median runtime (more stable than mean)
    times.sort((a, b) => a - b);
    const medianTime = times[Math.floor(times.length / 2)];

    return {
      candidateId: candidate.id,
      runtimeMs: medianTime,
      memoryBytes: Math.max(0, memoryAfter - memoryBefore),
      output: null,
      error: null,
    };
  } catch (error) {
    return {
      candidateId: candidate.id,
      runtimeMs: Infinity,
      memoryBytes: 0,
      output: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Create an executable function from code string
 */
function createFunctionFromCode(code: string): (input: unknown) => unknown {
  // Extract the main function from the code
  // This is a simplified version - real implementation would be more robust

  // Try to find an exported function
  const exportMatch = code.match(/export\s+(?:default\s+)?function\s+(\w+)/);
  const functionMatch = code.match(/function\s+(\w+)\s*\(/);
  const arrowMatch = code.match(/(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])\s*=>/);

  const funcName = exportMatch?.[1] || functionMatch?.[1] || arrowMatch?.[1] || 'search';

  // Wrap code to extract the function
  const wrappedCode = `
    ${code}
    return ${funcName};
  `;

  try {
    const factory = new Function(wrappedCode);
    return factory();
  } catch {
    // If that fails, try evaluating as expression
    return new Function('input', code) as (input: unknown) => unknown;
  }
}

/**
 * String Search Benchmark Suite
 */
export function createStringSearchSuite(
  texts: string[],
  patterns: string[]
): BenchmarkSuite {
  const inputs: Array<{ text: string; pattern: string }> = [];

  for (const text of texts) {
    for (const pattern of patterns) {
      inputs.push({ text, pattern });
    }
  }

  return {
    name: 'String Search',
    inputs,
    validate: (input, output) => {
      const { text, pattern } = input as { text: string; pattern: string };
      const indices = output as number[];

      // Verify each reported match is valid
      for (const idx of indices) {
        if (text.slice(idx, idx + pattern.length) !== pattern) {
          return false;
        }
      }

      return true;
    },
  };
}

/**
 * Generate test corpus for string search
 */
export function generateTestCorpus(): { texts: string[]; patterns: string[] } {
  // Generate varied test texts
  const texts: string[] = [];

  // Random text
  const chars = 'abcdefghijklmnopqrstuvwxyz';
  for (let size of [1000, 10000, 100000]) {
    let text = '';
    for (let i = 0; i < size; i++) {
      text += chars[Math.floor(Math.random() * chars.length)];
    }
    texts.push(text);
  }

  // Repetitive text (stress test for naive algorithms)
  texts.push('a'.repeat(10000) + 'b');
  texts.push('ab'.repeat(5000));

  // Natural language-like text
  const words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'];
  let naturalText = '';
  for (let i = 0; i < 2000; i++) {
    naturalText += words[Math.floor(Math.random() * words.length)] + ' ';
  }
  texts.push(naturalText);

  // DNA-like text
  const dnaChars = 'ACGT';
  let dnaText = '';
  for (let i = 0; i < 50000; i++) {
    dnaText += dnaChars[Math.floor(Math.random() * dnaChars.length)];
  }
  texts.push(dnaText);

  // Patterns of varying lengths
  const patterns = [
    'abc', // short
    'abcdef', // medium
    'the quick brown', // with spaces
    'aaaaaaaab', // repetitive
    'ACGTACGT', // DNA motif
    'fox jumps over', // natural language
  ];

  return { texts, patterns };
}

/**
 * Compare candidates head-to-head
 */
export async function compareCandidates(
  candidates: Candidate[],
  suite: BenchmarkSuite
): Promise<
  Array<{
    candidateId: string;
    rank: number;
    runtimeMs: number;
    relativeSpeed: number;
  }>
> {
  const results: Array<{ candidateId: string; runtimeMs: number }> = [];

  for (const candidate of candidates) {
    const result = await runBenchmark(candidate, suite);
    results.push({
      candidateId: candidate.id,
      runtimeMs: result.runtimeMs,
    });
  }

  // Sort by runtime (faster = better)
  results.sort((a, b) => a.runtimeMs - b.runtimeMs);

  const fastestTime = results[0]?.runtimeMs || 1;

  return results.map((r, i) => ({
    candidateId: r.candidateId,
    rank: i + 1,
    runtimeMs: r.runtimeMs,
    relativeSpeed: fastestTime / r.runtimeMs, // 1.0 = fastest
  }));
}
