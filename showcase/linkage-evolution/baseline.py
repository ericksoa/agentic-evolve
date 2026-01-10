"""
Baseline 4-bar linkage - a simple starting point for evolution.

This is a classic crank-rocker configuration that produces
a somewhat arbitrary coupler curve. Evolution should be able
to significantly improve on this for any specific target shape.

The parameters are intentionally not optimized for any particular
target, giving evolution plenty of room to improve.
"""

# Linkage dimensions (all in arbitrary units, will be normalized)
L1 = 4.0    # Ground link (distance between fixed pivots)
L2 = 1.5    # Crank (input link that rotates)
L3 = 3.5    # Coupler (connects crank to follower)
L4 = 3.0    # Follower/Rocker

# Tracer point position on coupler
# Px: distance along coupler from crank-coupler joint toward follower
# Py: perpendicular offset (positive = "above" coupler when facing from crank to follower)
Px = 1.75   # Halfway along coupler
Py = 1.0    # Offset perpendicular to coupler

# Notes for evolution:
# - L2 should typically be the shortest link for full crank rotation (Grashof)
# - Px and Py dramatically affect the shape of the traced curve
# - Small changes to link lengths can produce very different curves
# - The coupler curve is surprisingly complex even for simple linkages
