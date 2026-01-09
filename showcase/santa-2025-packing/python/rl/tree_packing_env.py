#!/usr/bin/env python3
"""
Gymnasium environment for tree packing.

The agent learns to place trees one at a time to minimize the bounding square.

MDP Formulation:
- State: Positions/angles of placed trees + number remaining
- Action: (x, y, angle) for next tree placement
- Reward: -delta_bbox (negative increase in bounding box) + bonus for valid placement
- Done: All trees placed or invalid placement
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List

# Tree polygon vertices (15 vertices)
TREE_VERTICES = np.array([
    [0.0, 0.8],      # Tip
    [0.125, 0.5],    # Right top tier outer
    [0.0625, 0.5],   # Right top tier inner
    [0.2, 0.25],     # Right mid tier outer
    [0.1, 0.25],     # Right mid tier inner
    [0.35, 0.0],     # Right base outer
    [0.075, 0.0],    # Right trunk
    [0.075, -0.2],   # Right trunk bottom
    [-0.075, -0.2],  # Left trunk bottom
    [-0.075, 0.0],   # Left trunk
    [-0.35, 0.0],    # Left base outer
    [-0.1, 0.25],    # Left mid tier inner
    [-0.2, 0.25],    # Left mid tier outer
    [-0.0625, 0.5],  # Left top tier inner
    [-0.125, 0.5],   # Left top tier outer
], dtype=np.float32)


def rotate_vertices(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vertices by angle in degrees."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return vertices @ rot.T


def get_tree_vertices(x: float, y: float, angle: float) -> np.ndarray:
    """Get tree vertices at position (x, y) with rotation angle."""
    rotated = rotate_vertices(TREE_VERTICES, angle)
    return rotated + np.array([x, y])


def segments_intersect(a1: np.ndarray, a2: np.ndarray,
                       b1: np.ndarray, b2: np.ndarray) -> bool:
    """Check if segment (a1,a2) intersects segment (b1,b2)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    d1 = ccw(a1, a2, b1)
    d2 = ccw(a1, a2, b2)
    d3 = ccw(b1, b2, a1)
    d4 = ccw(b1, b2, a2)

    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i

    return inside


def polygons_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    """Check if two polygons overlap."""
    n1, n2 = len(verts1), len(verts2)

    # Check edge intersections
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            l = (k + 1) % n2
            if segments_intersect(verts1[i], verts1[j], verts2[k], verts2[l]):
                return True

    # Check containment
    if point_in_polygon(verts1[0], verts2):
        return True
    if point_in_polygon(verts2[0], verts1):
        return True

    return False


class TreePackingEnv(gym.Env):
    """
    Environment for sequential tree packing.

    The agent places one tree at a time, trying to minimize the final bounding square.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, n_trees: int = 5, max_coord: float = 3.0,
                 render_mode: Optional[str] = None):
        """
        Initialize environment.

        Args:
            n_trees: Number of trees to pack
            max_coord: Maximum coordinate value (defines placement range)
            render_mode: Rendering mode
        """
        super().__init__()

        self.n_trees = n_trees
        self.max_coord = max_coord
        self.render_mode = render_mode

        # Action space: (x, y, angle) normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space:
        # - For each tree: (x, y, angle, placed) - 4 values
        # - Plus: current bbox side, trees remaining
        obs_dim = n_trees * 4 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.positions: List[Tuple[float, float]] = []
        self.angles: List[float] = []
        self.current_tree = 0
        self.current_bbox = 0.0

    def _get_obs(self) -> np.ndarray:
        """Get observation from current state."""
        obs = np.zeros(self.n_trees * 4 + 2, dtype=np.float32)

        # Tree positions and angles (normalized)
        for i, (pos, angle) in enumerate(zip(self.positions, self.angles)):
            obs[i * 4] = pos[0] / self.max_coord
            obs[i * 4 + 1] = pos[1] / self.max_coord
            obs[i * 4 + 2] = angle / 360.0
            obs[i * 4 + 3] = 1.0  # Placed flag

        # Current bbox (normalized)
        obs[-2] = self.current_bbox / (self.max_coord * 2)

        # Trees remaining (normalized)
        obs[-1] = (self.n_trees - self.current_tree) / self.n_trees

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        return {
            "trees_placed": self.current_tree,
            "current_bbox": self.current_bbox,
            "positions": list(self.positions),
            "angles": list(self.angles),
        }

    def _compute_bbox(self) -> float:
        """Compute current bounding square side length."""
        if not self.positions:
            return 0.0

        all_verts = []
        for pos, angle in zip(self.positions, self.angles):
            verts = get_tree_vertices(pos[0], pos[1], angle)
            all_verts.append(verts)

        all_verts = np.vstack(all_verts)
        min_x, min_y = all_verts.min(axis=0)
        max_x, max_y = all_verts.max(axis=0)

        return max(max_x - min_x, max_y - min_y)

    def _check_overlap(self, x: float, y: float, angle: float) -> bool:
        """Check if new tree at (x, y, angle) overlaps with existing trees."""
        new_verts = get_tree_vertices(x, y, angle)

        for pos, ang in zip(self.positions, self.angles):
            existing_verts = get_tree_vertices(pos[0], pos[1], ang)
            if polygons_overlap(new_verts, existing_verts):
                return True

        return False

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.positions = []
        self.angles = []
        self.current_tree = 0
        self.current_bbox = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action to place next tree.

        Args:
            action: [x, y, angle] normalized to [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Denormalize action
        x = action[0] * self.max_coord
        y = action[1] * self.max_coord
        angle = (action[2] + 1) * 180  # Map [-1, 1] to [0, 360]

        # Check for overlap
        if self._check_overlap(x, y, angle):
            # Invalid placement - large penalty
            reward = -10.0
            terminated = True
            truncated = False
        else:
            # Valid placement
            old_bbox = self.current_bbox

            self.positions.append((x, y))
            self.angles.append(angle)
            self.current_tree += 1
            self.current_bbox = self._compute_bbox()

            # Reward: negative change in bbox (smaller is better)
            delta_bbox = self.current_bbox - old_bbox
            reward = -delta_bbox

            # Bonus for completing the packing
            if self.current_tree == self.n_trees:
                # Final bonus inversely proportional to bbox
                final_bonus = 1.0 / (1.0 + self.current_bbox)
                reward += final_bonus
                terminated = True
            else:
                terminated = False

            truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_human(self):
        """Print current state."""
        print(f"Trees placed: {self.current_tree}/{self.n_trees}")
        print(f"Current bbox: {self.current_bbox:.4f}")
        for i, (pos, angle) in enumerate(zip(self.positions, self.angles)):
            print(f"  Tree {i}: ({pos[0]:.3f}, {pos[1]:.3f}) @ {angle:.1f}°")

    def _render_rgb(self) -> np.ndarray:
        """Render to RGB array."""
        # Simple visualization (could be enhanced)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        fig, ax = plt.subplots(figsize=(6, 6))

        for pos, angle in zip(self.positions, self.angles):
            verts = get_tree_vertices(pos[0], pos[1], angle)
            poly = Polygon(verts, fill=True, alpha=0.5, edgecolor='green', facecolor='lightgreen')
            ax.add_patch(poly)

        ax.set_xlim(-self.max_coord, self.max_coord)
        ax.set_ylim(-self.max_coord, self.max_coord)
        ax.set_aspect('equal')
        ax.grid(True)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return img


# Register environment
gym.register(
    id="TreePacking-v0",
    entry_point="python.rl.tree_packing_env:TreePackingEnv",
)


if __name__ == "__main__":
    # Quick test
    env = TreePackingEnv(n_trees=5)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Random episode
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        print(f"Step {info['trees_placed']}: reward={reward:.4f}, bbox={info['current_bbox']:.4f}")

    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Final bbox: {info['current_bbox']:.4f}")
