/**
 * Population Manager
 *
 * Manages the population of candidate solutions:
 * - Storage and retrieval
 * - Selection operations
 * - Pruning and diversity maintenance
 * - Persistence to disk
 */

import { nanoid } from 'nanoid';
import {
  Candidate,
  EvolutionConfig,
  FitnessScore,
} from './types.js';

export class Population {
  private candidates: Map<string, Candidate> = new Map();
  private config: EvolutionConfig;
  private seedId: string | null = null;

  constructor(config: EvolutionConfig) {
    this.config = config;
  }

  /**
   * Add a candidate to the population
   */
  add(candidate: Candidate): void {
    this.candidates.set(candidate.id, candidate);
    if (candidate.generation === 0 && candidate.mutationStrategy === 'seed') {
      this.seedId = candidate.id;
    }
  }

  /**
   * Get a candidate by ID
   */
  get(id: string): Candidate | undefined {
    return this.candidates.get(id);
  }

  /**
   * Get the seed candidate
   */
  getSeed(): Candidate | null {
    return this.seedId ? this.candidates.get(this.seedId) ?? null : null;
  }

  /**
   * Get all candidates
   */
  getAll(): Candidate[] {
    return Array.from(this.candidates.values());
  }

  /**
   * Get population size
   */
  size(): number {
    return this.candidates.size;
  }

  /**
   * Get top N candidates by fitness
   */
  getTop(n: number): Candidate[] {
    return this.getAll()
      .filter((c) => c.fitness !== null)
      .sort((a, b) => (b.fitness?.total ?? 0) - (a.fitness?.total ?? 0))
      .slice(0, n);
  }

  /**
   * Get the best candidate
   */
  getBest(): Candidate | null {
    const top = this.getTop(1);
    return top[0] ?? null;
  }

  /**
   * Get random sample of candidates
   */
  getRandomSample(n: number): Candidate[] {
    const all = this.getAll();
    const shuffled = all.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(n, shuffled.length));
  }

  /**
   * Get candidates from a specific generation
   */
  getGeneration(gen: number): Candidate[] {
    return this.getAll().filter((c) => c.generation === gen);
  }

  /**
   * Prune population to maintain size limit
   * Keeps elites + diverse samples
   */
  prune(): void {
    const maxSize = this.config.populationSize;
    if (this.candidates.size <= maxSize) {
      return;
    }

    const evaluated = this.getAll().filter((c) => c.fitness !== null);
    const unevaluated = this.getAll().filter((c) => c.fitness === null);

    // Always keep elites
    const elites = new Set(this.getTop(this.config.eliteCount).map((c) => c.id));

    // Always keep seed
    if (this.seedId) {
      elites.add(this.seedId);
    }

    // Sort rest by fitness
    const remaining = evaluated
      .filter((c) => !elites.has(c.id))
      .sort((a, b) => (b.fitness?.total ?? 0) - (a.fitness?.total ?? 0));

    // Keep top performers
    const keepCount = maxSize - elites.size - this.config.diversityCount;
    const toKeep = new Set([
      ...elites,
      ...remaining.slice(0, Math.max(0, keepCount)).map((c) => c.id),
    ]);

    // Add diversity samples (random from lower performers)
    const lowerPerformers = remaining.slice(keepCount);
    const diversitySamples = lowerPerformers
      .sort(() => Math.random() - 0.5)
      .slice(0, this.config.diversityCount);
    for (const c of diversitySamples) {
      toKeep.add(c.id);
    }

    // Remove candidates not in keep set
    for (const id of this.candidates.keys()) {
      if (!toKeep.has(id)) {
        this.candidates.delete(id);
      }
    }

    console.log(
      `Pruned population: ${evaluated.length + unevaluated.length} -> ${this.candidates.size}`
    );
  }

  /**
   * Get lineage (ancestry) of a candidate
   */
  getLineage(id: string): string[] {
    const lineage: string[] = [];
    let current = this.candidates.get(id);

    while (current) {
      lineage.unshift(current.id);
      if (current.parentIds.length === 0) {
        break;
      }
      current = this.candidates.get(current.parentIds[0]);
    }

    return lineage;
  }

  /**
   * Get diversity metrics
   */
  getDiversityMetrics(): {
    uniqueStrategies: number;
    generationSpread: number;
    fitnessVariance: number;
  } {
    const candidates = this.getAll();
    const strategies = new Set(candidates.map((c) => c.mutationStrategy));
    const generations = new Set(candidates.map((c) => c.generation));

    const fitnesses = candidates
      .filter((c) => c.fitness !== null)
      .map((c) => c.fitness!.total);

    const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const variance =
      fitnesses.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / fitnesses.length;

    return {
      uniqueStrategies: strategies.size,
      generationSpread: generations.size,
      fitnessVariance: variance,
    };
  }

  /**
   * Get statistics about the population
   */
  getStats(): {
    size: number;
    evaluated: number;
    generations: number;
    bestFitness: number;
    averageFitness: number;
    strategies: Record<string, number>;
  } {
    const candidates = this.getAll();
    const evaluated = candidates.filter((c) => c.fitness !== null);
    const fitnesses = evaluated.map((c) => c.fitness!.total);

    const strategies: Record<string, number> = {};
    for (const c of candidates) {
      strategies[c.mutationStrategy] = (strategies[c.mutationStrategy] || 0) + 1;
    }

    return {
      size: candidates.length,
      evaluated: evaluated.length,
      generations: Math.max(...candidates.map((c) => c.generation), 0),
      bestFitness: Math.max(...fitnesses, 0),
      averageFitness:
        fitnesses.length > 0 ? fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length : 0,
      strategies,
    };
  }

  /**
   * Save population to disk
   */
  async save(): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');

    const outputDir = this.config.outputDir;
    await fs.mkdir(outputDir, { recursive: true });
    await fs.mkdir(path.join(outputDir, 'candidates'), { recursive: true });

    // Save population metadata
    const metadata = {
      stats: this.getStats(),
      diversity: this.getDiversityMetrics(),
      candidates: Array.from(this.candidates.keys()),
    };
    await fs.writeFile(
      path.join(outputDir, 'population.json'),
      JSON.stringify(metadata, null, 2)
    );

    // Save each candidate
    for (const [id, candidate] of this.candidates) {
      await fs.writeFile(
        path.join(outputDir, 'candidates', `${id}.json`),
        JSON.stringify(candidate, null, 2)
      );

      // Also save just the code for easy inspection
      const ext = this.detectExtension(candidate.code);
      await fs.writeFile(
        path.join(outputDir, 'candidates', `${id}${ext}`),
        candidate.code
      );
    }

    // Save best candidate prominently
    const best = this.getBest();
    if (best) {
      const ext = this.detectExtension(best.code);
      await fs.writeFile(path.join(outputDir, `best${ext}`), best.code);
    }

    // Save lineage
    if (best) {
      const lineage = this.getLineage(best.id);
      await fs.writeFile(
        path.join(outputDir, 'lineage.json'),
        JSON.stringify(
          {
            championId: best.id,
            lineage,
            lineageDetails: lineage.map((id) => {
              const c = this.candidates.get(id);
              return {
                id,
                generation: c?.generation,
                strategy: c?.mutationStrategy,
                fitness: c?.fitness?.total,
              };
            }),
          },
          null,
          2
        )
      );
    }

    console.log(`Population saved to ${outputDir}`);
  }

  /**
   * Load population from disk
   */
  async load(): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');

    const outputDir = this.config.outputDir;
    const candidatesDir = path.join(outputDir, 'candidates');

    try {
      const files = await fs.readdir(candidatesDir);
      const jsonFiles = files.filter((f) => f.endsWith('.json'));

      for (const file of jsonFiles) {
        const content = await fs.readFile(path.join(candidatesDir, file), 'utf-8');
        const candidate: Candidate = JSON.parse(content);
        this.add(candidate);
      }

      console.log(`Loaded ${this.candidates.size} candidates from ${outputDir}`);
    } catch {
      console.log('No existing population found, starting fresh');
    }
  }

  /**
   * Detect file extension from code content
   */
  private detectExtension(code: string): string {
    if (code.includes('import ') || code.includes('export ')) {
      if (code.includes(': ') || code.includes('interface ') || code.includes('type ')) {
        return '.ts';
      }
      return '.js';
    }
    if (code.includes('def ') || code.includes('import ')) {
      return '.py';
    }
    if (code.includes('func ') || code.includes('package ')) {
      return '.go';
    }
    if (code.includes('fn ') || code.includes('let mut')) {
      return '.rs';
    }
    return '.txt';
  }
}
