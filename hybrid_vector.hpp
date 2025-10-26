#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <cassert>
#include <memory>
#include <omp.h>

#ifndef N_DIM
#define N_DIM 1024
#endif

using u64 = std::uint64_t;

template <typename fpT, typename qT>
class HybridVector {
private:
    size_t m_size;

    std::vector<fpT> m_fp_half;
    std::vector<qT> m_q_half;

    fpT m_fp_min;
    fpT m_fp_max;

    qT m_q_min = static_cast<qT>(0);
    qT m_q_max = std::numeric_limits<qT>::max();

    fpT m_scale;
    fpT m_offset;

    qT m_quantize_fp(const fpT x) {
        if (m_fp_max == m_fp_min) {
            return static_cast<qT>(0);  // All values are the same
        }
        return static_cast<qT>((x / m_scale) + m_offset);
    }

    fpT m_dequantize_q(const qT x) const {
        if (m_fp_max == m_fp_min) {
            return m_fp_min;  // All values are the same, return the constant value
        }
        return (static_cast<fpT>(x) - m_offset) * m_scale;
    }

public:

    HybridVector(const std::vector<fpT> &vec) {
        auto it_min = std::min_element(vec.begin(), vec.end());
        m_fp_min = *it_min;

        auto it_max = std::max_element(vec.begin(), vec.end());
        m_fp_max = *it_max;

        m_scale = (m_fp_max - m_fp_min) / (m_q_max - m_q_min);

        // Handle edge case where all values are the same (zero range)
        if (m_fp_max == m_fp_min) {
            m_scale = static_cast<fpT>(1.0);  // Avoid division by zero
            m_offset = static_cast<fpT>(0.0);
        } else {
            m_offset = m_q_min - (m_fp_min / m_scale);
        }

        std::vector<fpT> working_vec = vec;
        if (vec.size() % 2 == 0) {
            working_vec.push_back(static_cast<fpT>(0));
        }

        m_size = working_vec.size();

        size_t half_size = m_size / 2;

        m_fp_half.resize(half_size);
        m_q_half.resize(half_size);

        for (size_t i = 0; i < half_size; i++) {
            m_fp_half[i] = working_vec[i];
        }

#pragma omp simd
        for (size_t i = 0; i < half_size; i++) {
            m_q_half[i] = m_quantize_fp(working_vec[i + half_size]);
        }
    }

    HybridVector& operator+=(const HybridVector& other) {
        assert(m_fp_half.size() == other.m_fp_half.size());
        assert(m_q_half.size() == other.m_q_half.size());
        
#pragma omp simd
        for (size_t i = 0; i < m_fp_half.size(); i++) {
            m_fp_half[i] += other.m_fp_half[i];
            m_q_half[i] += other.m_q_half[i];
        }
        
        return *this;
    }

    HybridVector& operator-=(const HybridVector& other) {
        assert(m_fp_half.size() == other.m_fp_half.size());
        assert(m_q_half.size() == other.m_q_half.size());
        
#pragma omp simd
        for (size_t i = 0; i < m_fp_half.size(); i++) {
            m_fp_half[i] -= other.m_fp_half[i];
            m_q_half[i] -= other.m_q_half[i];
        }
        
        return *this;
    }

    HybridVector& operator*=(const HybridVector& other) {
        assert(m_fp_half.size() == other.m_fp_half.size());
        assert(m_q_half.size() == other.m_q_half.size());
        
#pragma omp simd
        for (size_t i = 0; i < m_fp_half.size(); i++) {
            m_fp_half[i] *= other.m_fp_half[i];
            m_q_half[i] *= other.m_q_half[i];
        }
        
        return *this;
    }

    fpT accumulate() {
        fpT sum = 0;
        
#pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < m_fp_half.size(); i++) {
            sum += m_fp_half[i];
            sum += m_dequantize_q(m_q_half[i]);
        }
        
        return sum;
    }

    fpT squared_distance_to(const HybridVector& other) const {
        assert(m_fp_half.size() == other.m_fp_half.size());
        assert(m_q_half.size() == other.m_q_half.size());
        
        fpT sum = 0;
        
        // Handle special case where all values are the same (zero range)
        if (m_fp_max == m_fp_min) {
            // All quantized values are the same, so difference is always 0
#pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < m_fp_half.size(); i++) {
                fpT fp_diff = m_fp_half[i] - other.m_fp_half[i];
                sum += fp_diff * fp_diff;
                // q_half contribution is 0 since all values are identical
            }
        } else {
            // Normal case: linearized quantized computation
            // (dequantize(a) - dequantize(b))² = scale² * (a - b)²
            fpT scale_squared = m_scale * other.m_scale;
            
#pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < m_fp_half.size(); i++) {
                // For fp_half: compute difference and square directly
                fpT fp_diff = m_fp_half[i] - other.m_fp_half[i];
                sum += fp_diff * fp_diff;
                
                // For q_half: linearized computation (a - b)² * scale²
                fpT q_diff = static_cast<fpT>(m_q_half[i]) - static_cast<fpT>(other.m_q_half[i]);
                sum += q_diff * q_diff * scale_squared;
            }
        }
        
        return sum;
    }

    HybridVector operator+(const HybridVector& other) const {
        HybridVector result = *this;
        result += other;
        return result;
    }

    HybridVector operator-(const HybridVector& other) const {
        HybridVector result = *this;
        result -= other;
        return result;
    }

    HybridVector operator*(const HybridVector& other) const {
        HybridVector result = *this;
        result *= other;
        return result;
    }

};