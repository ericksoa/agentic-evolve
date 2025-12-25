/**
 * Fitness Evaluator
 *
 * Evaluates candidate solutions:
 * - Runs tests for correctness
 * - Benchmarks performance
 * - Measures memory usage
 * - Calculates complexity
 * - Computes overall fitness score
 */

import { Candidate, EvolutionConfig, FitnessScore, BenchmarkResult } from './types.js';
import { spawn } from 'child_process';

export class Evaluator {
  private config: EvolutionConfig;

  constructor(config: EvolutionConfig) {
    this.config = config;
  }

  /**
   * Evaluate a candidate and update its fitness score
   */
  async evaluate(candidate: Candidate): Promise<FitnessScore> {
    const startTime = Date.now();

    try {
      // 1. Check correctness (must pass or fitness = 0)
      const correctnessResult = await this.checkCorrectness(candidate);
      if (!correctnessResult.passed) {
        const failedFitness: FitnessScore = {
          total: 0,
          correctness: 0,
          speed: 0,
          memory: 0,
          complexity: 0,
          raw: {
            passedTests: correctnessResult.passedTests,
            totalTests: correctnessResult.totalTests,
            runtimeMs: 0,
            memoryBytes: 0,
            cyclomaticComplexity: 0,
          },
        };
        candidate.fitness = failedFitness;
        candidate.metadata.evaluatedAt = new Date().toISOString();
        return failedFitness;
      }

      // 2. Benchmark performance
      const perfResult = await this.benchmarkPerformance(candidate);

      // 3. Measure memory
      const memoryResult = await this.measureMemory(candidate);

      // 4. Calculate complexity
      const complexity = this.calculateComplexity(candidate.code);

      // 5. Normalize scores (0-1, higher is better)
      const speedScore = this.normalizeSpeed(perfResult.runtimeMs);
      const memoryScore = this.normalizeMemory(memoryResult.bytes);
      const complexityScore = this.normalizeComplexity(complexity);

      // 6. Calculate weighted total
      const weights = this.config.fitnessWeights;
      const total =
        weights.speed * speedScore +
        weights.memory * memoryScore +
        weights.complexity * complexityScore;

      const fitness: FitnessScore = {
        total,
        correctness: 1,
        speed: speedScore,
        memory: memoryScore,
        complexity: complexityScore,
        raw: {
          passedTests: correctnessResult.passedTests,
          totalTests: correctnessResult.totalTests,
          runtimeMs: perfResult.runtimeMs,
          memoryBytes: memoryResult.bytes,
          cyclomaticComplexity: complexity,
        },
      };

      candidate.fitness = fitness;
      candidate.metadata.evaluatedAt = new Date().toISOString();

      console.log(
        `Evaluated ${candidate.id}: fitness=${total.toFixed(4)} ` +
          `(speed=${speedScore.toFixed(2)}, mem=${memoryScore.toFixed(2)}, ` +
          `complexity=${complexityScore.toFixed(2)})`
      );

      return fitness;
    } catch (error) {
      console.error(`Evaluation failed for ${candidate.id}:`, error);

      const errorFitness: FitnessScore = {
        total: 0,
        correctness: 0,
        speed: 0,
        memory: 0,
        complexity: 0,
        raw: {
          passedTests: 0,
          totalTests: 0,
          runtimeMs: 0,
          memoryBytes: 0,
          cyclomaticComplexity: 0,
        },
      };

      candidate.fitness = errorFitness;
      candidate.metadata.evaluatedAt = new Date().toISOString();
      return errorFitness;
    }
  }

  /**
   * Check correctness by running tests
   */
  private async checkCorrectness(
    candidate: Candidate
  ): Promise<{ passed: boolean; passedTests: number; totalTests: number }> {
    if (!this.config.testCommand) {
      // No tests configured, assume correct
      return { passed: true, passedTests: 1, totalTests: 1 };
    }

    try {
      const result = await this.runCommand(this.config.testCommand, candidate.code);
      // Parse test output (this would need customization per test framework)
      const passed = result.exitCode === 0;
      return {
        passed,
        passedTests: passed ? 1 : 0,
        totalTests: 1,
      };
    } catch {
      return { passed: false, passedTests: 0, totalTests: 1 };
    }
  }

  /**
   * Benchmark runtime performance
   */
  private async benchmarkPerformance(
    candidate: Candidate
  ): Promise<{ runtimeMs: number }> {
    if (!this.config.benchmarkCommand) {
      // Run internal benchmark
      return this.runInternalBenchmark(candidate);
    }

    const startTime = Date.now();
    await this.runCommand(this.config.benchmarkCommand, candidate.code);
    return { runtimeMs: Date.now() - startTime };
  }

  /**
   * Run internal benchmark (for string search showcase)
   */
  private async runInternalBenchmark(
    candidate: Candidate
  ): Promise<{ runtimeMs: number }> {
    // This would execute the candidate code and measure performance
    // For now, return a simulated result based on code characteristics
    const codeLength = candidate.code.length;
    const hasOptimizations =
      candidate.code.includes('cache') ||
      candidate.code.includes('memo') ||
      candidate.code.includes('Map') ||
      candidate.code.includes('Set');

    // Simulate: shorter code with optimizations = faster
    const baseTime = 100;
    const lengthPenalty = codeLength / 100;
    const optimizationBonus = hasOptimizations ? 0.7 : 1.0;

    return {
      runtimeMs: baseTime * lengthPenalty * optimizationBonus,
    };
  }

  /**
   * Measure memory usage
   */
  private async measureMemory(candidate: Candidate): Promise<{ bytes: number }> {
    // Estimate memory from code characteristics
    const hasLargeDataStructures =
      candidate.code.includes('new Array') ||
      candidate.code.includes('new Map') ||
      candidate.code.includes('new Set');

    const baseMemory = 1024 * 1024; // 1MB base
    const structurePenalty = hasLargeDataStructures ? 2 : 1;
    const codeSizeFactor = candidate.code.length / 500;

    return {
      bytes: Math.floor(baseMemory * structurePenalty * (1 + codeSizeFactor * 0.1)),
    };
  }

  /**
   * Calculate cyclomatic complexity
   */
  private calculateComplexity(code: string): number {
    // Simple heuristic: count decision points
    const patterns = [
      /\bif\b/g,
      /\belse\b/g,
      /\bfor\b/g,
      /\bwhile\b/g,
      /\bswitch\b/g,
      /\bcase\b/g,
      /\bcatch\b/g,
      /\?\s*.*\s*:/g, // ternary
      /&&/g,
      /\|\|/g,
    ];

    let complexity = 1; // Base complexity
    for (const pattern of patterns) {
      const matches = code.match(pattern);
      complexity += matches ? matches.length : 0;
    }

    return complexity;
  }

  /**
   * Normalize speed score (lower runtime = higher score)
   */
  private normalizeSpeed(runtimeMs: number): number {
    // Use exponential decay: faster = better
    // 10ms -> ~0.9, 100ms -> ~0.37, 1000ms -> ~0.05
    const targetMs = 10;
    return Math.exp(-runtimeMs / (targetMs * 10));
  }

  /**
   * Normalize memory score (lower memory = higher score)
   */
  private normalizeMemory(bytes: number): number {
    // Use exponential decay
    const targetBytes = 512 * 1024; // 512KB target
    return Math.exp(-bytes / (targetBytes * 10));
  }

  /**
   * Normalize complexity score (lower complexity = higher score)
   */
  private normalizeComplexity(complexity: number): number {
    // Use inverse: complexity 1 = 1.0, complexity 10 = 0.1, etc.
    return 1 / complexity;
  }

  /**
   * Run a shell command with candidate code
   */
  private runCommand(
    command: string,
    code: string
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      const child = spawn('bash', ['-c', command], {
        env: { ...process.env, CANDIDATE_CODE: code },
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (exitCode) => {
        resolve({ exitCode: exitCode ?? 1, stdout, stderr });
      });

      child.on('error', reject);

      // Timeout after 30 seconds
      setTimeout(() => {
        child.kill();
        reject(new Error('Command timeout'));
      }, 30000);
    });
  }
}

/**
 * Create a custom evaluator for string search
 */
export function createStringSearchEvaluator(
  config: EvolutionConfig,
  testCases: Array<{ text: string; pattern: string; expected: number[] }>
): Evaluator {
  const evaluator = new Evaluator(config);

  // Override internal benchmark for string search
  // This would be injected in a real implementation

  return evaluator;
}
