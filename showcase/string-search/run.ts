#!/usr/bin/env npx tsx
/**
 * String Search Evolution Runner
 *
 * Demonstrates the AlphaEvolve system by evolving a string search algorithm.
 * Run with: npx tsx showcase/string-search/run.ts
 */

import { Orchestrator } from '../../src/orchestrator.js';
import { generateTestCorpus, createStringSearchSuite, compareCandidates } from '../../src/benchmarks/runner.js';
import { algorithms } from './baseline.js';
import seedSearch from './seed.js';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║         AlphaEvolve String Search Evolution                ║');
  console.log('║     Evolving algorithms to beat Boyer-Moore                ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  // Load seed algorithm
  const seedCode = readFileSync(join(__dirname, 'seed.ts'), 'utf-8');

  // Create benchmark suite
  console.log('Generating test corpus...');
  const { texts, patterns } = generateTestCorpus();
  const suite = createStringSearchSuite(texts, patterns);
  console.log(`  ${texts.length} text samples`);
  console.log(`  ${patterns.length} patterns`);
  console.log(`  ${texts.length * patterns.length} total test cases\n`);

  // Benchmark baselines
  console.log('Benchmarking baseline algorithms...\n');
  console.log('Algorithm       | Ops/sec     | Relative');
  console.log('----------------|-------------|----------');

  const baselineResults: Array<{ name: string; opsPerSec: number }> = [];

  for (const [name, fn] of Object.entries(algorithms)) {
    const startTime = performance.now();
    let ops = 0;

    // Run for at least 1 second
    while (performance.now() - startTime < 1000) {
      for (const input of suite.inputs.slice(0, 10)) {
        const { text, pattern } = input as { text: string; pattern: string };
        fn(text, pattern);
        ops++;
      }
    }

    const elapsed = (performance.now() - startTime) / 1000;
    const opsPerSec = ops / elapsed;
    baselineResults.push({ name, opsPerSec });
  }

  // Sort by performance
  baselineResults.sort((a, b) => b.opsPerSec - a.opsPerSec);
  const bestBaseline = baselineResults[0].opsPerSec;

  for (const { name, opsPerSec } of baselineResults) {
    const relative = opsPerSec / bestBaseline;
    console.log(
      `${name.padEnd(15)} | ${opsPerSec.toFixed(0).padStart(11)} | ${(relative * 100).toFixed(1)}%`
    );
  }

  console.log('\n');

  // Initialize evolution
  console.log('Initializing evolution...');
  const orchestrator = new Orchestrator(
    'Find all occurrences of a pattern string within a larger text string. Return an array of starting indices where the pattern is found. The algorithm should be as fast as possible while maintaining correctness. Consider: preprocessing the pattern, skipping text efficiently, cache-friendly memory access, handling edge cases (empty strings, single characters, overlapping matches).',
    seedCode,
    {
      mutatorCount: 16,
      evaluatorCount: 8,
      populationSize: 50,
      maxGenerations: 15,
      convergenceThreshold: 4,
      outputDir: '.evolve/string-search',
    }
  );

  // Set up progress reporting
  orchestrator.setProgressCallback((result) => {
    const improvement = result.improvement * 100;
    const sign = improvement >= 0 ? '+' : '';
    console.log(
      `Generation ${result.generation}: ` +
        `${result.candidatesCreated} variants | ` +
        `Best: ${result.bestFitness.toFixed(4)} (${sign}${improvement.toFixed(1)}%)`
    );
  });

  // Run evolution
  console.log('\nStarting evolution...\n');
  console.log('Generation | Variants | Best Fitness | Improvement');
  console.log('-----------|----------|--------------|------------');

  const result = await orchestrator.evolve();

  // Report results
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║                    EVOLUTION COMPLETE                      ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  console.log(`Status: ${result.success ? 'SUCCESS' : 'FAILED'}`);
  console.log(`Generations: ${result.generations.length}`);
  console.log(`Total candidates: ${result.totalCandidates}`);
  console.log(`Time elapsed: ${(result.totalTimeMs / 1000).toFixed(1)}s`);
  console.log(`\nFitness improvement: ${(result.improvement * 100).toFixed(1)}%`);
  console.log(`  Baseline: ${result.baselineFitness.toFixed(4)}`);
  console.log(`  Final:    ${result.finalFitness.toFixed(4)}`);

  if (result.champion) {
    console.log('\nChampion lineage:');
    for (const id of result.lineage) {
      console.log(`  → ${id}`);
    }
    console.log('\nChampion code saved to: .evolve/string-search/best.ts');
  }

  // Save state
  await orchestrator.saveState();

  console.log('\nEvolution artifacts saved to .evolve/string-search/');
}

// Run if executed directly
main().catch(console.error);
