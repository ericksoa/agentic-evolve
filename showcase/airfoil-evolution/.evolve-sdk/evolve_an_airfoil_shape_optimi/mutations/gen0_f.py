#!/usr/bin/env python3
"""
Gen 0 Variant F: Hierarchical Clustering Initialization
Uses hierarchical clustering in CST parameter space to ensure maximum diversity.
Optimized with distance matrix caching and vectorized cluster operations.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import parent modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from airfoil import Airfoil
from evaluate import evaluate_airfoil, DesignRequirements

class HierarchicalClusteringInitializer:
    """Initialize population using hierarchical clustering for maximum diversity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.distance_cache = {}

    def compute_airfoil_distance(self, airfoil1: Airfoil, airfoil2: Airfoil) -> float:
        """
        Compute distance between airfoils in CST parameter space.
        Uses weighted Euclidean distance with geometric feature weighting.
        """
        # Coefficient distance (primary)
        upper_dist = np.linalg.norm(np.array(airfoil1.upper_coeffs) - np.array(airfoil2.upper_coeffs))
        lower_dist = np.linalg.norm(np.array(airfoil1.lower_coeffs) - np.array(airfoil2.lower_coeffs))

        # Geometric feature distance (secondary)
        thick1, _ = airfoil1.max_thickness()
        thick2, _ = airfoil2.max_thickness()
        camber1, _ = airfoil1.max_camber()
        camber2, _ = airfoil2.max_camber()

        geometric_dist = np.sqrt((thick1 - thick2)**2 + (camber1 - camber2)**2)

        # Weighted combination
        return 0.7 * (upper_dist + lower_dist) + 0.3 * geometric_dist

    def generate_overcomplete_population(self, size: int = 200) -> List[Airfoil]:
        """Generate large overcomplete population for clustering."""
        population = []

        # Strategy 1: Dense random sampling (40%)
        n_random = int(0.4 * size)
        for _ in range(n_random):
            upper = self.rng.uniform(0.01, 0.25, 5).tolist()
            lower = self.rng.uniform(-0.25, -0.01, 5).tolist()
            population.append(Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0))

        # Strategy 2: Systematic grid sampling (30%)
        n_grid = int(0.3 * size)
        grid_points = self._generate_grid_samples(n_grid)
        for upper, lower in grid_points:
            population.append(Airfoil(upper_coeffs=upper, lower_coeffs=lower, zte=0.0))

        # Strategy 3: Perturbations of known designs (20%)
        n_perturb = int(0.2 * size)
        base_designs = self._get_base_designs()
        for _ in range(n_perturb):
            base = self.rng.choice(base_designs)
            noise_scale = 0.03
            upper = np.array(base[0]) + self.rng.normal(0, noise_scale, 5)
            lower = np.array(base[1]) + self.rng.normal(0, noise_scale, 5)
            upper = np.clip(upper, 0.01, 0.3)
            lower = np.clip(lower, -0.3, -0.01)
            population.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist(), zte=0.0))

        # Strategy 4: Edge cases (10%)
        n_edge = size - len(population)
        edge_cases = self._generate_edge_cases(n_edge)
        population.extend(edge_cases)

        return population

    def _generate_grid_samples(self, n_samples: int) -> List[Tuple[List[float], List[float]]]:
        """Generate grid-based samples in CST space."""
        # 5D grid for upper coefficients, 5D for lower
        samples = []
        grid_size = int(np.ceil(n_samples ** (1/10)))  # 10D hypercube

        # Simple systematic sampling (production would use Halton sequence)
        for i in range(n_samples):
            # Convert index to multi-dimensional grid coordinates
            coords = []
            temp_i = i
            for _ in range(10):
                coords.append((temp_i % grid_size) / grid_size)
                temp_i //= grid_size

            upper = [0.01 + coords[j] * 0.24 for j in range(5)]
            lower = [-0.25 + coords[j+5] * 0.24 for j in range(5)]
            samples.append((upper, lower))

        return samples

    def _get_base_designs(self) -> List[Tuple[List[float], List[float]]]:
        """Known good base designs for perturbation."""
        return [
            ([0.17, 0.16, 0.14, 0.12, 0.10], [-0.17, -0.16, -0.14, -0.12, -0.10]),  # NACA 0012-like
            ([0.20, 0.18, 0.16, 0.13, 0.10], [-0.14, -0.12, -0.10, -0.09, -0.08]),  # NACA 2412-like
            ([0.13, 0.12, 0.11, 0.09, 0.07], [-0.13, -0.12, -0.11, -0.09, -0.07]),  # Thin symmetric
            ([0.22, 0.20, 0.17, 0.14, 0.11], [-0.18, -0.15, -0.12, -0.10, -0.08]),  # Thick cambered
            ([0.15, 0.14, 0.12, 0.10, 0.08], [-0.12, -0.10, -0.08, -0.06, -0.05]),  # Balanced
        ]

    def _generate_edge_cases(self, n_samples: int) -> List[Airfoil]:
        """Generate edge case airfoils (extremes of design space)."""
        edge_cases = []

        for _ in range(n_samples):
            # Random selection of extreme cases
            case_type = self.rng.choice(['max_thickness', 'min_thickness', 'max_camber', 'flat'])

            if case_type == 'max_thickness':
                upper = [0.25, 0.22, 0.19, 0.16, 0.13]
                lower = [-0.25, -0.22, -0.19, -0.16, -0.13]
            elif case_type == 'min_thickness':
                upper = [0.06, 0.05, 0.04, 0.03, 0.02]
                lower = [-0.06, -0.05, -0.04, -0.03, -0.02]
            elif case_type == 'max_camber':
                upper = [0.20, 0.18, 0.16, 0.14, 0.12]
                lower = [-0.10, -0.08, -0.06, -0.05, -0.04]
            else:  # flat
                upper = [0.01, 0.01, 0.01, 0.01, 0.01]
                lower = [-0.01, -0.01, -0.01, -0.01, -0.01]

            # Add small perturbation
            upper = np.array(upper) + self.rng.normal(0, 0.005, 5)
            lower = np.array(lower) + self.rng.normal(0, 0.005, 5)
            upper = np.clip(upper, 0.01, 0.3)
            lower = np.clip(lower, -0.3, -0.01)

            edge_cases.append(Airfoil(upper_coeffs=upper.tolist(), lower_coeffs=lower.tolist(), zte=0.0))

        return edge_cases

    def cluster_and_select(self, population: List[Airfoil], target_size: int = 50) -> List[Airfoil]:
        """Use hierarchical clustering to select diverse representatives."""
        n_pop = len(population)

        # Compute pairwise distance matrix (vectorized where possible)
        print(f"Computing distance matrix for {n_pop} airfoils...")
        distances = np.zeros((n_pop, n_pop))

        # Use symmetric property for efficiency
        for i in range(n_pop):
            for j in range(i+1, n_pop):
                dist = self.compute_airfoil_distance(population[i], population[j])
                distances[i, j] = distances[j, i] = dist

        # Convert to condensed form for scipy
        condensed_distances = squareform(distances)

        # Hierarchical clustering
        print("Performing hierarchical clustering...")
        linkage_matrix = linkage(condensed_distances, method='ward')

        # Form clusters
        cluster_labels = fcluster(linkage_matrix, target_size, criterion='maxclust')

        # Select representative from each cluster
        selected_airfoils = []
        for cluster_id in range(1, target_size + 1):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # Select cluster centroid (airfoil closest to cluster center)
            if len(cluster_indices) == 1:
                selected_airfoils.append(population[cluster_indices[0]])
            else:
                # Find centroid of cluster in distance space
                cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                centroid_idx = cluster_indices[np.argmin(np.sum(cluster_distances, axis=1))]
                selected_airfoils.append(population[centroid_idx])

        print(f"Selected {len(selected_airfoils)} diverse representatives")
        return selected_airfoils

def generate_population() -> List[Airfoil]:
    """Generate hierarchically clustered diverse population."""
    initializer = HierarchicalClusteringInitializer(seed=161718)

    # Generate overcomplete population
    overcomplete_pop = initializer.generate_overcomplete_population(150)

    # Filter out invalid designs
    valid_pop = [a for a in overcomplete_pop if a.is_valid()[0]]
    print(f"Valid airfoils: {len(valid_pop)}/{len(overcomplete_pop)}")

    # Cluster and select diverse subset
    if len(valid_pop) > 50:
        selected_pop = initializer.cluster_and_select(valid_pop, 50)
    else:
        selected_pop = valid_pop

    return selected_pop

def evaluate_population(population: List[Airfoil]) -> List[Tuple[Airfoil, float]]:
    """Evaluate clustered population."""
    req = DesignRequirements(reynolds=200000, objective="max_ld")
    results = []

    for airfoil in population:
        result = evaluate_airfoil(airfoil, req, use_xfoil=False)
        fitness = result.get('fitness', 0.0) if result.get('valid', False) else 0.0
        results.append((airfoil, fitness))

    return results

def analyze_clustering_quality(population: List[Airfoil]) -> Dict[str, float]:
    """Analyze quality of clustered population."""
    n_pop = len(population)

    # Minimum pairwise distance (diversity measure)
    min_distances = []
    for i in range(n_pop):
        distances_i = []
        for j in range(n_pop):
            if i != j:
                dist = np.linalg.norm(
                    np.array(population[i].upper_coeffs + population[i].lower_coeffs) -
                    np.array(population[j].upper_coeffs + population[j].lower_coeffs)
                )
                distances_i.append(dist)
        min_distances.append(min(distances_i))

    # Coverage metrics
    all_coeffs = np.array([a.upper_coeffs + a.lower_coeffs for a in population])
    ranges = np.max(all_coeffs, axis=0) - np.min(all_coeffs, axis=0)

    return {
        'min_pairwise_distance': min(min_distances),
        'mean_min_distance': np.mean(min_distances),
        'coefficient_coverage': np.mean(ranges),
        'diversity_score': np.mean(min_distances) * np.mean(ranges)
    }

def main():
    """Main hierarchical clustering initialization."""
    print("Generating hierarchically clustered airfoil population...")

    # Generate population
    population = generate_population()

    # Evaluate fitness
    evaluated = evaluate_population(population)

    # Sort by fitness
    evaluated.sort(key=lambda x: x[1], reverse=True)

    # Analyze clustering quality
    clustering_quality = analyze_clustering_quality(population)

    # Save results
    os.makedirs("results", exist_ok=True)

    best_results = []
    for i, (airfoil, fitness) in enumerate(evaluated[:20]):
        filename = f"results/clustered_airfoil_{i:02d}_fitness_{fitness:.3f}.json"
        with open(filename, 'w') as f:
            data = airfoil.to_dict()
            data['cluster_rank'] = i
            data['fitness'] = fitness
            json.dump(data, f, indent=2)

        best_results.append({
            "filename": filename,
            "fitness": fitness,
            "cluster_rank": i,
            "clustering_quality": clustering_quality
        })

    print(f"Generated {len(population)} clustered airfoils")
    print(f"Best fitness: {evaluated[0][1]:.4f}")
    print(f"Clustering quality metrics:")
    for key, value in clustering_quality.items():
        print(f"  {key}: {value:.4f}")

    # Save clustering analysis
    with open("results/clustering_analysis.json", 'w') as f:
        json.dump({
            "clustering_quality": clustering_quality,
            "population_size": len(population),
            "best_fitness": evaluated[0][1]
        }, f, indent=2)

    return best_results

if __name__ == "__main__":
    main()