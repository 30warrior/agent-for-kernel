Suggestions:
- Fuse Elementwise Operations and Reductions When Possible
- Use NEON Vectorized Access with Alignment and Stride Checks
- Use Multiple Independent Accumulators for Reductions and FMA
- Tile Loops and Reuse Data in Registers and Cache
- Hoist Broadcasting, Shape and Address Calculations out of Hot Loops
- Handle Vector Tails Safely and Preserve Numerical Accuracy
- Measure Packing and Layout Conversion Costs within the Model Boundary

Focus:
- NEON is fixed-width; do not use SVE, SVE2, SME or other unsupported instructions.
- Choose the vectorization axis from actual strides; avoid unnecessary gathers and scalar work per lane.
- Consider register blocking, loop unrolling, load/store traffic and dependency chains.
- Remain single-threaded; do not use OpenMP or external threaded libraries.
- Evaluate every fixed case and do not optimize by recognizing particular input values.
