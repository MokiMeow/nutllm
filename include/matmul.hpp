#pragma once
#include "tensor.hpp"

/* Three implementations of C = A x B (A: MxK, B: KxN, C: MxN).
 *
 * They compute the same thing and differ only in how they touch memory and
 * whether they use SIMD — which is the whole lesson: on modern CPUs, matmul
 * performance is a memory-access problem before it is an arithmetic problem. */

/* Textbook triple loop (i, j, k). Correct, and the slowest: the inner loop
 * strides down a column of B, touching a new cache line every iteration. */
void matmul_naive(const Matrix &a, const Matrix &b, Matrix &c);

/* Loop reorder (i, k, j) plus cache blocking. The inner loop now walks a row
 * of B contiguously, and tiling keeps the working set inside L1/L2. */
void matmul_blocked(const Matrix &a, const Matrix &b, Matrix &c);

/* Blocked layout plus AVX2 + FMA: eight floats per instruction, one fused
 * multiply-add instead of a separate multiply and add. Falls back to the
 * blocked version when built without AVX2. */
void matmul_simd(const Matrix &a, const Matrix &b, Matrix &c);

/* True when this build actually has AVX2/FMA compiled in. */
bool simd_available();
