#pragma once

#include <cstddef>

#include "tensor.hpp"

void softmax_reference(const float *input, float *output,
                       size_t rows, size_t cols);
void softmax_inplace(Matrix &matrix);

void rmsnorm_reference(const float *input, const float *weight, float *output,
                       size_t size, float epsilon);
void rmsnorm(const float *input, const float *weight, float *output,
             size_t size, float epsilon);

float silu(float value);
void swiglu_reference(const Matrix &input, const Matrix &gate_weight,
                      const Matrix &up_weight, Matrix &output);
void swiglu(const Matrix &input, const Matrix &gate_weight,
            const Matrix &up_weight, Matrix &output);

void rope_reference(float *query, float *key, size_t sequence,
                    size_t dimensions, size_t position_offset,
                    float theta = 10000.0f);
void rope(float *query, float *key, size_t sequence, size_t dimensions,
          size_t position_offset, float theta = 10000.0f);

void add_inplace(float *destination, const float *source, size_t size);

void matvec_reference(const Matrix &matrix, const float *vector, float *output);
void matvec(const Matrix &matrix, const float *vector, float *output);
