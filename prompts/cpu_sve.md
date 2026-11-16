Suggestions:
- Fuse Elementwise Operations and Reductions When Possible
- Use Predicated SVE Loads and Stores with Runtime Vector Lengths
- Use Multiple Independent Accumulators for Reductions and FMA
- Tile Loops and Reuse Data in Registers and Cache
- Hoist Broadcasting, Shape and Address Calculations out of Hot Loops
- Handle Vector Tails with Predicates and Preserve Numerical Accuracy
- Measure Packing and Layout Conversion Costs within the Model Boundary

Focus:
- Generate vector-length-agnostic SVE using runtime lane counts; do not assume the observed vector length is universal.
- Use only the SVE instructions enabled by the supplied compiler flags; do not assume SVE2 or SME.
- Prefer contiguous vector access; evaluate gather/scatter and address-generation costs where necessary.
- Consider register blocking, loop unrolling, load/store traffic and dependency chains.
- Remain single-threaded; do not use OpenMP or external threaded libraries.
- Evaluate every fixed case and do not optimize by recognizing particular input values.
