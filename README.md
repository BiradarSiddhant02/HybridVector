# Hybrid Vector Quantization Performance Benchmark

This project implements and benchmarks a hybrid vector quantization approach for Euclidean distance computation. The hybrid vector splits input data into two halves: one half stored as full-precision floating-point values and the other half quantized to 8-bit unsigned integers.

## Overview

### Hybrid Vector Structure
- **Floating-point half**: First half of the vector stored as `float` values
- **Quantized half**: Second half compressed to `uint8_t` using min/max scaling
- **Quantization scheme**: Linear scaling with offset to map `[min, max]` range to `[0, 255]`

### Key Features
- **Memory efficiency**: Reduces storage by ~37.5% (half float + half uint8)
- **SIMD optimization**: Vectorized distance computation using AVX2 instructions
- **Instruction-level parallelism**: Simultaneous dispatch of integer and floating-point operations on separate CPU execution units
- **OpenMP parallelization**: Multi-threaded execution for performance
- **Accuracy preservation**: Maintains reasonable precision for distance calculations

## Implementation Details

The hybrid vector uses a quantization scheme:
```cpp
scale = (fp_max - fp_min) / (255 - 0)
offset = 0 - (fp_min / scale)
quantized_value = (original_value / scale) + offset
```

Distance computation processes both halves:
- Floating-point half: Standard squared difference
- Quantized half: Dequantized squared difference with scale correction

## Technical Notes

The implementation leverages:
- **AVX2 vectorization**: `vpmovzxbd` for efficient uint8→float conversion
- **Fused multiply-add**: `vfmadd` instructions for optimal throughput  
- **Data independence**: Both vector halves can be processed in parallel without dependencies

## Performance Results

The benchmark demonstrates significant performance improvements with the hybrid quantization approach:

![Speedup Analysis](speedup_analsys.png)

Results show:
- **Vector size**: 4,096 dimensions
- **Test vectors**: 1,000 vectors per run
- **Iterations**: 100 distance calculations per run
- **Runs**: 50 independent benchmark runs

### Key Metrics
- **Average speedup**: Hybrid approach vs. full floating-point
- **Accuracy**: Relative error between hybrid and reference implementations
- **Consistency**: Performance variance across multiple runs

## Files

- `benchmark_euclidean.cpp`: Main benchmark implementation
- `hybrid_vector.hpp`: HybridVector class template
- `speedup_results.csv`: Detailed per-run results
- `speedup_stats.csv`: Summary statistics
- `plot_speedup.py`: Visualization script

## Building and Running

```bash
# Compile with optimization
clang++ -O3 -march=native -fopenmp benchmark_euclidean.cpp -o benchmark_euclidean -lgomp

# Run benchmark
./benchmark_euclidean

# Generate plots
python plot_speedup.py
```

This hybrid approach demonstrates practical quantization techniques for high-dimensional vector operations while maintaining computational accuracy.