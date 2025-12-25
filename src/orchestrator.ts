/**
 * Evolution Orchestrator
 *
 * Controls the evolutionary loop:
 * 1. Initialize population with seed
 * 2. For each generation:
 *    a. Select parents for mutation
 *    b. Spawn parallel mutator agents
 *    c. Evaluate new candidates
 *    d. Update population (selection + pruning)
 *    e. Check convergence
 * 3. Return champion
 */

import { nanoid } from 'nanoid';
import {
  Candidate,
  EvolutionConfig,
  EvolutionState,
  EvolutionResult,
  GenerationResult,
  MutationStrategy,
  DEFAULT_CONFIG,
  FitnessScore,
} from './types.js';
import { Population } from './population.js';
import { Evaluator } from './evaluator.js';
import { getMutationPrompt, MUTATION_STRATEGIES } from './mutations/strategies.js';

export class Orchestrator {
  private config: EvolutionConfig;
  private population: Population;
  private evaluator: Evaluator;
  private state: EvolutionState;
  private problemDescription: string;
  private onProgress: ((result: GenerationResult) => void) | null = null;

  constructor(
    problemDescription: string,
    seedCode: string,
    config: Partial<EvolutionConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.problemDescription = problemDescription;
    this.population = new Population(this.config);
    this.evaluator = new Evaluator(this.config);

    // Initialize state
    this.state = {
      config: this.config,
      generation: 0,
      candidates: new Map(),
      bestCandidateId: null,
      baselineFitness: 0,
      generationsWithoutImprovement: 0,
      startedAt: new Date().toISOString(),
      status: 'running',
    };

    // Create seed candidate
    const seed = this.createSeedCandidate(seedCode);
    this.population.add(seed);
  }

  /**
   * Set progress callback for real-time updates
   */
  setProgressCallback(callback: (result: GenerationResult) => void): void {
    this.onProgress = callback;
  }

  /**
   * Run the full evolution loop
   */
  async evolve(): Promise<EvolutionResult> {
    const startTime = Date.now();
    const generationResults: GenerationResult[] = [];

    try {
      // Evaluate seed
      console.log('Evaluating seed candidate...');
      const seed = this.population.getSeed();
      if (seed) {
        await this.evaluator.evaluate(seed);
        this.state.baselineFitness = seed.fitness?.total ?? 0;
        this.state.bestCandidateId = seed.id;
        console.log(`Baseline fitness: ${this.state.baselineFitness.toFixed(4)}`);
      }

      // Evolution loop
      while (this.shouldContinue()) {
        this.state.generation++;
        console.log(`\n--- Generation ${this.state.generation} ---`);

        const result = await this.runGeneration();
        generationResults.push(result);

        if (this.onProgress) {
          this.onProgress(result);
        }

        // Check for improvement
        if (result.improvement > 0.05) {
          // 5% improvement threshold
          this.state.generationsWithoutImprovement = 0;
        } else {
          this.state.generationsWithoutImprovement++;
        }

        // Prune population
        this.population.prune();
      }

      // Determine final status
      if (this.state.generationsWithoutImprovement >= this.config.convergenceThreshold) {
        this.state.status = 'converged';
      } else {
        this.state.status = 'completed';
      }

      // Get champion
      const champion = this.population.getBest();

      return {
        success: true,
        generations: generationResults,
        champion,
        baselineFitness: this.state.baselineFitness,
        finalFitness: champion?.fitness?.total ?? 0,
        improvement: champion
          ? ((champion.fitness?.total ?? 0) - this.state.baselineFitness) /
            this.state.baselineFitness
          : 0,
        totalCandidates: this.population.size(),
        totalTimeMs: Date.now() - startTime,
        lineage: champion ? this.population.getLineage(champion.id) : [],
      };
    } catch (error) {
      this.state.status = 'failed';
      throw error;
    }
  }

  /**
   * Run a single generation
   */
  private async runGeneration(): Promise<GenerationResult> {
    const previousBest = this.population.getBest()?.fitness?.total ?? 0;

    // 1. Select parents for mutation
    const parents = this.selectParents();
    console.log(`Selected ${parents.length} parents for mutation`);

    // 2. Generate mutation tasks
    const mutationTasks = this.createMutationTasks(parents);
    console.log(`Created ${mutationTasks.length} mutation tasks`);

    // 3. Execute mutations in parallel (this would spawn Claude agents)
    const newCandidates = await this.executeMutations(mutationTasks);
    console.log(`Generated ${newCandidates.length} new candidates`);

    // 4. Add to population
    for (const candidate of newCandidates) {
      this.population.add(candidate);
    }

    // 5. Evaluate new candidates in parallel
    await this.evaluateInParallel(newCandidates);

    // 6. Get stats
    const best = this.population.getBest();
    const bestFitness = best?.fitness?.total ?? 0;
    const improvement = previousBest > 0 ? (bestFitness - previousBest) / previousBest : 0;

    if (bestFitness > previousBest) {
      this.state.bestCandidateId = best?.id ?? null;
      console.log(`New best: ${bestFitness.toFixed(4)} (+${(improvement * 100).toFixed(1)}%)`);
    }

    // Count strategies used
    const strategies: Record<MutationStrategy, number> = {} as Record<MutationStrategy, number>;
    for (const candidate of newCandidates) {
      strategies[candidate.mutationStrategy] =
        (strategies[candidate.mutationStrategy] || 0) + 1;
    }

    return {
      generation: this.state.generation,
      candidatesCreated: newCandidates.length,
      candidatesEvaluated: newCandidates.filter((c) => c.fitness !== null).length,
      bestFitness,
      bestCandidateId: best?.id ?? '',
      improvement,
      strategies,
    };
  }

  /**
   * Check if evolution should continue
   */
  private shouldContinue(): boolean {
    if (this.state.generation >= this.config.maxGenerations) {
      console.log('Reached maximum generations');
      return false;
    }

    if (this.state.generationsWithoutImprovement >= this.config.convergenceThreshold) {
      console.log('Converged (no improvement)');
      return false;
    }

    const best = this.population.getBest();
    if (this.config.targetFitness && best?.fitness?.total) {
      if (best.fitness.total >= this.config.targetFitness) {
        console.log('Reached target fitness');
        return false;
      }
    }

    return true;
  }

  /**
   * Select parents for mutation using tournament selection + elitism
   */
  private selectParents(): Candidate[] {
    const parents: Candidate[] = [];

    // Always include elites
    const elites = this.population.getTop(this.config.eliteCount);
    parents.push(...elites);

    // Tournament selection for the rest
    const tournamentSize = 3;
    const additionalNeeded = this.config.mutatorCount - elites.length;

    for (let i = 0; i < additionalNeeded; i++) {
      const tournament = this.population.getRandomSample(tournamentSize);
      const winner = tournament.reduce((best, current) =>
        (current.fitness?.total ?? 0) > (best.fitness?.total ?? 0) ? current : best
      );
      parents.push(winner);
    }

    // Add some random for diversity
    const randomCandidates = this.population.getRandomSample(this.config.diversityCount);
    parents.push(...randomCandidates);

    return parents;
  }

  /**
   * Create mutation tasks distributed across strategies
   */
  private createMutationTasks(
    parents: Candidate[]
  ): Array<{ strategy: MutationStrategy; parents: Candidate[] }> {
    const tasks: Array<{ strategy: MutationStrategy; parents: Candidate[] }> = [];
    const strategies = [...MUTATION_STRATEGIES];

    for (let i = 0; i < this.config.mutatorCount; i++) {
      const strategy = strategies[i % strategies.length];
      const taskParents: Candidate[] = [];

      if (strategy === 'crossover') {
        // Crossover needs two parents
        taskParents.push(parents[i % parents.length]);
        taskParents.push(parents[(i + 1) % parents.length]);
      } else {
        // Single parent mutation
        taskParents.push(parents[i % parents.length]);
      }

      tasks.push({ strategy, parents: taskParents });
    }

    return tasks;
  }

  /**
   * Execute mutations in parallel
   * In real implementation, this spawns Claude agents via Task tool
   */
  private async executeMutations(
    tasks: Array<{ strategy: MutationStrategy; parents: Candidate[] }>
  ): Promise<Candidate[]> {
    // This is where we'd spawn parallel Claude agents
    // For now, simulate with sequential execution

    const results: Candidate[] = [];

    // In production, this would be:
    // const promises = tasks.map(task => this.spawnMutatorAgent(task));
    // const results = await Promise.all(promises);

    for (const task of tasks) {
      const prompt = getMutationPrompt(
        task.strategy,
        task.parents.map((p) => p.code),
        this.problemDescription
      );

      // Simulate mutation (in real implementation, Claude generates this)
      const mutatedCode = this.simulateMutation(task.parents[0].code, task.strategy);

      const candidate: Candidate = {
        id: nanoid(10),
        generation: this.state.generation,
        code: mutatedCode,
        parentIds: task.parents.map((p) => p.id),
        mutationStrategy: task.strategy,
        fitness: null,
        metadata: {
          createdAt: new Date().toISOString(),
          evaluatedAt: null,
          mutatorPrompt: prompt,
          linesOfCode: mutatedCode.split('\n').length,
        },
      };

      results.push(candidate);
    }

    return results;
  }

  /**
   * Evaluate candidates in parallel
   */
  private async evaluateInParallel(candidates: Candidate[]): Promise<void> {
    // Batch candidates for parallel evaluation
    const batchSize = this.config.evaluatorCount;

    for (let i = 0; i < candidates.length; i += batchSize) {
      const batch = candidates.slice(i, i + batchSize);
      await Promise.all(batch.map((c) => this.evaluator.evaluate(c)));
    }
  }

  /**
   * Create seed candidate from initial code
   */
  private createSeedCandidate(code: string): Candidate {
    return {
      id: nanoid(10),
      generation: 0,
      code,
      parentIds: [],
      mutationStrategy: 'seed',
      fitness: null,
      metadata: {
        createdAt: new Date().toISOString(),
        evaluatedAt: null,
        mutatorPrompt: '',
        linesOfCode: code.split('\n').length,
      },
    };
  }

  /**
   * Simulate mutation (placeholder for Claude agent)
   * In real implementation, Claude generates the mutated code
   */
  private simulateMutation(code: string, strategy: MutationStrategy): string {
    // This is a placeholder - real mutations come from Claude
    return code + `\n// Mutated with strategy: ${strategy}`;
  }

  /**
   * Get current state for persistence
   */
  getState(): EvolutionState {
    return { ...this.state };
  }

  /**
   * Save state to disk
   */
  async saveState(): Promise<void> {
    await this.population.save();
  }
}

/**
 * Factory function to create orchestrator from file path
 */
export async function createFromFile(
  problemDescription: string,
  filePath: string,
  config?: Partial<EvolutionConfig>
): Promise<Orchestrator> {
  const fs = await import('fs/promises');
  const seedCode = await fs.readFile(filePath, 'utf-8');
  return new Orchestrator(problemDescription, seedCode, config);
}
