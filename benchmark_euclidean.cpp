#include "hybrid_vector.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

using namespace std;
using namespace std::chrono;

// Euclidean distance using HybridVector - mathematically accurate version
template<typename fpT, typename qT>
fpT euclidean_distance_hybrid(const HybridVector<fpT, qT>& a, const HybridVector<fpT, qT>& b) {
    return sqrt(a.squared_distance_to(b));
}

// Regular euclidean distance for comparison
template<typename fpT>
fpT euclidean_distance_regular(const vector<fpT>& a, const vector<fpT>& b) {
    fpT sum = 0;
    
#pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < a.size(); i++) {
        fpT diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int main() {
    const int num_vectors = 1000;
    const int vector_size = 4096;
    const int num_iterations = 100;
    const int num_runs = 500;
    
    // Random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-10.0f, 10.0f);
    
    // Generate test vectors
    vector<vector<double>> test_vectors(num_vectors);
    for (int i = 0; i < num_vectors; i++) {
        test_vectors[i].resize(vector_size);
        for (int j = 0; j < vector_size; j++) {
            test_vectors[i][j] = dis(gen);
        }
    }
    
    // Create HybridVectors
    vector<HybridVector<double, uint8_t>> hybrid_vectors;
    for (int i = 0; i < num_vectors; i++) {
        hybrid_vectors.emplace_back(test_vectors[i]);
    }
    
    cout << "Benchmarking Euclidean Distance Calculation" << endl;
    cout << "Vector size: " << vector_size << endl;
    cout << "Number of vectors: " << num_vectors << endl;
    cout << "Iterations: " << num_iterations << endl;
    cout << "Number of runs: " << num_runs << endl << endl;
    
    vector<double> speedups;
    vector<double> errors;
    
    for (int run = 0; run < num_runs; run++) {
        cout << "Run " << (run + 1) << "/" << num_runs << "..." << endl;
        
        // Benchmark HybridVector approach
        auto start_hybrid = high_resolution_clock::now();
        float total_distance_hybrid = 0;
        
        for (int iter = 0; iter < num_iterations; iter++) {
            for (int i = 0; i < num_vectors - 1; i++) {
                total_distance_hybrid += euclidean_distance_hybrid(hybrid_vectors[i], hybrid_vectors[i + 1]);
            }
        }
        
        auto end_hybrid = high_resolution_clock::now();
        auto duration_hybrid = duration_cast<microseconds>(end_hybrid - start_hybrid);
        
        // Benchmark regular approach
        auto start_regular = high_resolution_clock::now();
        float total_distance_regular = 0;
        
        for (int iter = 0; iter < num_iterations; iter++) {
            for (int i = 0; i < num_vectors - 1; i++) {
                total_distance_regular += euclidean_distance_regular(test_vectors[i], test_vectors[i + 1]);
            }
        }
        
        auto end_regular = high_resolution_clock::now();
        auto duration_regular = duration_cast<microseconds>(end_regular - start_regular);
        
        double speedup = (double)duration_regular.count() / duration_hybrid.count();
        speedups.push_back(speedup);
        
        // Calculate relative error
        double relative_error = abs(total_distance_hybrid - total_distance_regular) / total_distance_regular;
        errors.push_back(relative_error);
        
        // cout << "  HybridVector: " << duration_hybrid.count() << " us" << endl;
        // cout << "  Regular: " << duration_regular.count() << " us" << endl;
        // cout << "  Speedup: " << speedup << "x" << endl;
        
        // if (run == 0) {
        //     cout << "  Total distances (sanity check):" << endl;
        //     cout << "    HybridVector: " << total_distance_hybrid << endl;
        //     cout << "    Regular: " << total_distance_regular << endl;
        //     cout << "    Difference: " << abs(total_distance_hybrid - total_distance_regular) << endl;
        // }
        // cout << endl;
    }
    
    // Calculate statistics
    double sum = 0;
    for (double speedup : speedups) {
        sum += speedup;
    }
    double avg_speedup = sum / num_runs;
    
    double error_sum = 0;
    for (double error : errors) {
        error_sum += error;
    }
    double avg_error = error_sum / num_runs;
    
    double min_speedup = *min_element(speedups.begin(), speedups.end());
    double max_speedup = *max_element(speedups.begin(), speedups.end());
    double min_error = *min_element(errors.begin(), errors.end());
    double max_error = *max_element(errors.begin(), errors.end());
    
    cout << "=== FINAL RESULTS ===" << endl;
    cout << "Average speedup: " << avg_speedup << "x" << endl;
    cout << "Min speedup: " << min_speedup << "x" << endl;
    cout << "Max speedup: " << max_speedup << "x" << endl;
    cout << "Average relative error: " << avg_error * 100 << "%" << endl;
    cout << "Min relative error: " << min_error * 100 << "%" << endl;
    cout << "Max relative error: " << max_error * 100 << "%" << endl;
    
    // Write CSV data
    ofstream csv_file("speedup_results.csv");
    csv_file << "run,speedup,relative_error" << endl;
    for (int i = 0; i < speedups.size(); i++) {
        csv_file << (i + 1) << "," << speedups[i] << "," << errors[i] << endl;
    }
    csv_file.close();
    
    // Write summary stats
    ofstream stats_file("speedup_stats.csv");
    stats_file << "metric,value" << endl;
    stats_file << "avg_speedup," << avg_speedup << endl;
    stats_file << "min_speedup," << min_speedup << endl;
    stats_file << "max_speedup," << max_speedup << endl;
    stats_file << "avg_error," << avg_error << endl;
    stats_file << "min_error," << min_error << endl;
    stats_file << "max_error," << max_error << endl;
    stats_file << "num_runs," << num_runs << endl;
    stats_file.close();
    
    cout << "Data written to speedup_results.csv and speedup_stats.csv" << endl;
    
    return 0;
}
