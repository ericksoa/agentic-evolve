"""
4-Bar Linkage Simulator

A 4-bar linkage consists of:
- Ground link (fixed, length L1)
- Crank (input, length L2) - rotates around fixed pivot
- Coupler (length L3) - connects crank to follower
- Follower/Rocker (length L4) - rotates around second fixed pivot

The tracer point is on the coupler, offset from the crank-coupler joint.

Parameters:
- L1: Ground length (distance between fixed pivots)
- L2: Crank length
- L3: Coupler length
- L4: Follower length
- Px, Py: Tracer point position relative to coupler (local coords)

ASCII diagram:
                    Tracer Point (P)
                         *
                        /
    Crank (L2)    Coupler (L3)
        O-----------O-----------O
       /                         \
      / θ (input)                  \ Follower (L4)
     O                              O
   Fixed                          Fixed
   Pivot A                        Pivot B
     |<-------- L1 (Ground) ------->|

"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Linkage4Bar:
    """4-bar linkage parameters."""
    L1: float  # Ground length
    L2: float  # Crank length
    L3: float  # Coupler length
    L4: float  # Follower length
    Px: float  # Tracer point X offset on coupler (from crank-coupler joint)
    Py: float  # Tracer point Y offset on coupler (perpendicular to coupler)

    def is_valid(self) -> Tuple[bool, str]:
        """Check if linkage parameters are physically valid."""
        # All lengths must be positive
        if any(x <= 0 for x in [self.L1, self.L2, self.L3, self.L4]):
            return False, "All link lengths must be positive"

        # Check Grashof condition for full rotation
        links = sorted([self.L1, self.L2, self.L3, self.L4])
        s, l = links[0], links[3]  # shortest and longest
        p, q = links[1], links[2]  # intermediate

        # For the linkage to move at all: s + l <= p + q
        if s + l > p + q + 0.001:  # small tolerance
            return False, "Linkage cannot move (violates Grashof inequality)"

        return True, "Valid"

    def can_complete_rotation(self) -> bool:
        """Check if crank can complete full rotation (Grashof crank-rocker or double-crank)."""
        links = sorted([self.L1, self.L2, self.L3, self.L4])
        s = links[0]  # shortest
        l = links[3]  # longest

        # Grashof condition: shortest + longest <= sum of others
        if s + l <= links[1] + links[2]:
            # Check if crank (L2) is the shortest link for full rotation
            return self.L2 == s or self.L1 == s
        return False

    def solve_position(self, theta2: float) -> Optional[Tuple[float, float, float, float]]:
        """
        Solve linkage position for given crank angle.

        Args:
            theta2: Crank angle in radians (0 = along positive X from pivot A)

        Returns:
            (Bx, By, Cx, Cy) - positions of crank-coupler joint (B) and coupler-follower joint (C)
            or None if position is invalid
        """
        # Fixed pivot positions
        Ax, Ay = 0.0, 0.0  # Crank pivot at origin
        Dx, Dy = self.L1, 0.0  # Follower pivot on X-axis

        # Crank-coupler joint position (B)
        Bx = Ax + self.L2 * math.cos(theta2)
        By = Ay + self.L2 * math.sin(theta2)

        # Solve for coupler-follower joint (C) using circle intersection
        # C is at distance L3 from B and distance L4 from D

        # Distance from D to B
        BD = math.sqrt((Bx - Dx)**2 + (By - Dy)**2)

        # Check if solution exists
        if BD > self.L3 + self.L4 or BD < abs(self.L3 - self.L4):
            return None  # No valid configuration

        if BD < 0.0001:
            return None  # B and D coincide

        # Using law of cosines to find angle
        # L4^2 = BD^2 + L3^2 - 2*BD*L3*cos(angle_DBC)
        cos_angle = (BD**2 + self.L3**2 - self.L4**2) / (2 * BD * self.L3)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp for numerical stability

        angle_DBC = math.acos(cos_angle)

        # Angle from B to D
        angle_BD = math.atan2(Dy - By, Dx - Bx)

        # Two solutions (above or below line BD), we typically want the one that
        # gives continuous motion - use the "above" solution (counter-clockwise)
        angle_BC = angle_BD + angle_DBC  # Could also use angle_BD - angle_DBC

        Cx = Bx + self.L3 * math.cos(angle_BC)
        Cy = By + self.L3 * math.sin(angle_BC)

        return Bx, By, Cx, Cy

    def get_tracer_point(self, theta2: float) -> Optional[Tuple[float, float]]:
        """
        Get the position of the tracer point for given crank angle.

        The tracer point is defined in local coupler coordinates:
        - Px: distance along coupler from B toward C
        - Py: perpendicular distance (positive = left of BC direction)
        """
        pos = self.solve_position(theta2)
        if pos is None:
            return None

        Bx, By, Cx, Cy = pos

        # Coupler direction vector (B to C)
        coupler_len = math.sqrt((Cx - Bx)**2 + (Cy - By)**2)
        if coupler_len < 0.0001:
            return None

        # Unit vectors along and perpendicular to coupler
        ux = (Cx - Bx) / coupler_len  # Along coupler
        uy = (Cy - By) / coupler_len

        # Perpendicular (90° counter-clockwise)
        vx = -uy
        vy = ux

        # Tracer point in world coordinates
        Tx = Bx + self.Px * ux + self.Py * vx
        Ty = By + self.Px * uy + self.Py * vy

        return Tx, Ty

    def trace_path(self, num_samples: int = 360, angle_range: Tuple[float, float] = None) -> List[Tuple[float, float]]:
        """
        Trace the coupler curve by sampling crank positions.

        Args:
            num_samples: Number of samples around the rotation
            angle_range: (start, end) in radians, or None for full rotation attempt

        Returns:
            List of (x, y) points traced by the tracer point
        """
        if angle_range is None:
            angle_range = (0, 2 * math.pi)

        points = []
        start, end = angle_range

        for i in range(num_samples):
            theta = start + (end - start) * i / num_samples
            pt = self.get_tracer_point(theta)
            if pt is not None:
                points.append(pt)

        return points


def load_linkage(filepath: str) -> Linkage4Bar:
    """Load linkage parameters from a Python file."""
    namespace = {}
    with open(filepath) as f:
        exec(f.read(), namespace)

    required = ['L1', 'L2', 'L3', 'L4', 'Px', 'Py']
    for var in required:
        if var not in namespace:
            raise ValueError(f"Linkage file must define {var}")

    return Linkage4Bar(
        L1=namespace['L1'],
        L2=namespace['L2'],
        L3=namespace['L3'],
        L4=namespace['L4'],
        Px=namespace['Px'],
        Py=namespace['Py']
    )


# Test
if __name__ == "__main__":
    # Classic crank-rocker example
    linkage = Linkage4Bar(
        L1=4.0,   # Ground
        L2=1.0,   # Crank (shortest for full rotation)
        L3=3.0,   # Coupler
        L4=3.0,   # Follower
        Px=1.5,   # Tracer halfway along coupler
        Py=1.0    # Offset perpendicular to coupler
    )

    valid, msg = linkage.is_valid()
    print(f"Linkage valid: {valid} ({msg})")
    print(f"Can complete rotation: {linkage.can_complete_rotation()}")

    path = linkage.trace_path(100)
    print(f"Traced {len(path)} points")
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        print(f"X range: {min(xs):.2f} to {max(xs):.2f}")
        print(f"Y range: {min(ys):.2f} to {max(ys):.2f}")
