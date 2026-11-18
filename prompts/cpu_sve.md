Suggestions:
- Fuse Elementwise Operations and Reductions to Reduce Intermediate Memory Traffic
- Choose NEON, SVE, or Mixed Paths According to Shapes, Strides, and Measured Performance
- Prefer Contiguous Vector Access and Broadcasts; Consider SVE Gather/Scatter or Packing When Their Costs Are Justified
- Use Independent Accumulators, Loop Unrolling, and FMA to Reduce Dependency Chains
- Tile Loops and Reuse Data in Registers and Cache; Balance Reuse Against Register Pressure
- Hoist Invariant Broadcasts, Shape and Address Calculations; Minimize Lane Extraction and Shuffles
- Handle NEON Tails Safely and Use SVE Predication for Partial Vectors
- After Unrolled SVE Loops, Process Every Remaining Full Vector Before the Final Predicated Tail, Including Noncontiguous Reduction Paths
- Consider Prefetching for Large Workloads; Include Packing and Layout Conversion Costs within the Model Boundary
- When Multiple Threads Are Configured, Parallelize Independent Outer Work with OpenMP and Use NEON/SVE within Each Thread; Avoid Parallel Overhead on Small Inputs

Focus:
- Allow NEON and SVE in the same implementation; mixing is optional and must justify its overhead. A NEON-only or SVE-only implementation is also valid.
- Use only instructions and intrinsics supported by the supplied compiler flags and toolchain; do not assume SVE implies SVE2, SME, optional extensions, or Neon-SVE bridge intrinsic availability.
- Keep SVE vector-length agnostic; use element-appropriate runtime lane counts and predicates, not a hard-coded observed vector length.
- Preserve inactive SVE accumulator lanes when needed, and include all valid partial sums in the final reduction; never consume unspecified inactive values.
- If the main loop advances by U * svcntw() (or the dtype-appropriate lane count), its remainder can exceed one vector. Use complete cleanup loops for all remaining elements on every stride path; a single svwhilelt load is insufficient for a multi-vector remainder. Check lengths below, at and above V, 2V and 4V and remainders between V and 2V.
- Choose vectorization axes from actual strides; avoid unnecessary gathers, scalar work per lane, spills, and excessive unrolling.
- Preserve numerical accuracy and memory safety; use FMA or approximate math only within the required tolerance.
- Follow the configured thread budget and CPU affinity. OpenMP is allowed only when multiple threads are configured; do not change thread limits or binding, use nested parallel regions, or call external numerical libraries.
- Evaluate every fixed case. Shape/stride-based specialization is allowed; recognizing particular input values or hard-coding outputs is not.
