"""
Main entry point for cuOpt LP AutoTuner.

Usage:
    python -m showcase.cuopt_lp_autotuner benchmark --policy baseline
    python -m showcase.cuopt_lp_autotuner evolve --generations 20
    python -m showcase.cuopt_lp_autotuner evolve --resume
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='cuOpt LP AutoTuner - Evolutionary policy learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baseline benchmark
    python -m showcase.cuopt_lp_autotuner benchmark --policy baseline

    # Benchmark a specific policy
    python -m showcase.cuopt_lp_autotuner benchmark --policy best_policy.json

    # Run evolution
    python -m showcase.cuopt_lp_autotuner evolve --generations 20

    # Resume evolution
    python -m showcase.cuopt_lp_autotuner evolve --resume
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Evaluate a policy')
    bench_parser.add_argument('--policy', type=str, default='baseline',
                              help='Policy path or "baseline" (default: baseline)')
    bench_parser.add_argument('--n-train', type=int, default=50,
                              help='Number of training instances (default: 50)')
    bench_parser.add_argument('--n-test', type=int, default=20,
                              help='Number of test instances (default: 20)')
    bench_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed (default: 42)')
    bench_parser.add_argument('--output', type=str, default=None,
                              help='Output JSON file for results')
    bench_parser.add_argument('--quiet', action='store_true',
                              help='Suppress verbose output')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve', help='Evolve policies')
    evolve_parser.add_argument('--resume', action='store_true',
                               help='Resume from checkpoint')
    evolve_parser.add_argument('--generations', type=int, default=20,
                               help='Max generations (default: 20)')
    evolve_parser.add_argument('--population', type=int, default=15,
                               help='Population size (default: 15)')
    evolve_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed (default: 42)')
    evolve_parser.add_argument('--output', type=str,
                               default='.evolve-sdk/cuopt_lp_autotuner',
                               help='Output directory')
    evolve_parser.add_argument('--quiet', action='store_true',
                               help='Suppress verbose output')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate LP corpus')
    gen_parser.add_argument('--n', type=int, default=100,
                            help='Number of instances (default: 100)')
    gen_parser.add_argument('--type', type=str, default=None,
                            help='LP type (transportation, capacity_planning, etc.)')
    gen_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    gen_parser.add_argument('--output', type=str, default='corpus',
                            help='Output directory')

    args = parser.parse_args()

    if args.command == 'benchmark':
        from .benchmark import run_benchmark
        import json

        results = run_benchmark(
            policy_path=args.policy,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
            verbose=not args.quiet,
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults written to {args.output}")

    elif args.command == 'evolve':
        from .evolve import run_evolution

        config = {
            'max_generations': args.generations,
            'population_size': args.population,
            'seed': args.seed,
            'verbose': not args.quiet,
            'output_dir': args.output,
        }

        results = run_evolution(
            config=config,
            resume=args.resume,
            output_dir=args.output,
        )

        print(f"\nEvolution complete!")
        print(f"  Champion: {results['champion_id']}")
        if results['champion_fitness'] is not None:
            print(f"  Fitness: {results['champion_fitness']:.2f}")
        print(f"  Generations: {results['generations']}")

    elif args.command == 'generate':
        from .lp_generator import LPGenerator, LPType
        import json
        from pathlib import Path

        gen = LPGenerator(seed=args.seed)

        lp_type = None
        if args.type:
            lp_type = LPType(args.type)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {args.n} LP instances...")

        for i in range(args.n):
            lp = gen.generate(lp_type)
            print(f"  {lp.name}: {lp.n_vars} vars, {lp.n_cons} cons, {lp.nnz} nnz")

        print(f"\nGenerated {args.n} instances")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
