Suggestions:
- Fuse Elementwise Operations and Reductions to Reduce Intermediate Memory Traffic
- Choose NEON Loop Structures According to Shapes, Strides, and Measured Performance
- Prefer Contiguous Vector Access and Broadcasts; Consider Packing When Its Costs Are Justified
- Use Independent Accumulators, Loop Unrolling, and FMA to Reduce Dependency Chains
- Tile Loops and Reuse Data in Registers and Cache; Balance Reuse Against Register Pressure
- Hoist Invariant Broadcasts, Shape and Address Calculations; Minimize Lane Extraction and Shuffles
- Handle Fixed-Width Vector Tails Safely and Check Strides and Alignment
- Consider Prefetching for Large Workloads; Include Packing and Layout Conversion Costs within the Model Boundary
- When Multiple Threads Are Configured, Parallelize Independent Outer Work with OpenMP and Use NEON within Each Thread; Avoid Parallel Overhead on Small Inputs

Focus:
- This target guarantees NEON only. Do not use SVE, SVE2, SME or optional extensions not enabled by the supplied compiler flags, even if the host supports them.
- Choose vectorization axes from actual strides; avoid scalar work per lane, spills, and excessive unrolling.
- Preserve numerical accuracy and memory safety; use FMA or approximate math only within the required tolerance.
- Follow the configured thread budget and CPU affinity. OpenMP is allowed only when multiple threads are configured; do not change thread limits or binding, use nested parallel regions, or call external numerical libraries.
- Evaluate every fixed case. Shape/stride-based specialization is allowed; recognizing particular input values or hard-coding outputs is not.
