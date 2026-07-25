#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "matmul.hpp"
#include "ops.hpp"
#include "perplexity.hpp"
#include "quant.hpp"
#include "selftest.hpp"

namespace {

void fill_random(Matrix &matrix, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < matrix.size(); index++)
        matrix.data()[index] = distribution(rng);
}

bool roundtrip_within_bound(const Matrix &source,
                            const QuantizedMatrix &quantized,
                            float &worst_ratio) {
    Matrix restored = dequantize(quantized);
    worst_ratio = 0.0f;
    for (size_t row = 0; row < source.rows(); row++) {
        for (size_t column = 0; column < source.cols(); column++) {
            const float error =
                std::fabs(source.at(row, column) -
                          restored.at(row, column));
            const float bound = quantized.error_bound(row, column);
            if (error > bound + 1e-7f)
                return false;
            if (bound > 0.0f)
                worst_ratio = std::max(worst_ratio, error / bound);
        }
    }
    return true;
}

bool matmul_within_bound(const Matrix &left, const Matrix &right,
                         const QuantizedMatrix &quantized,
                         float &worst_ratio) {
    Matrix reference(left.rows(), right.cols());
    Matrix candidate(left.rows(), right.cols());
    matmul_naive(left, right, reference);
    matmul_quantized(left, quantized, candidate);
    worst_ratio = 0.0f;
    for (size_t row = 0; row < left.rows(); row++) {
        for (size_t column = 0; column < right.cols(); column++) {
            double bound = 0.0;
            for (size_t inner = 0; inner < left.cols(); inner++)
                bound += std::fabs(double(left.at(row, inner))) *
                         double(quantized.error_bound(inner, column));
            const double error =
                std::fabs(double(reference.at(row, column)) -
                          double(candidate.at(row, column)));
            if (error > bound + 1e-5)
                return false;
            if (bound > 0.0)
                worst_ratio =
                    std::max(worst_ratio, float(error / bound));
        }
    }
    return true;
}

} // namespace

bool run_quant_selftests() {
    try {
        Matrix weights(5, 37);
        fill_random(weights, 505);
        const QuantizedMatrix int8 =
            QuantizedMatrix::quantize(weights, QuantType::int8);
        const QuantizedMatrix int4 =
            QuantizedMatrix::quantize(weights, QuantType::int4);
        Matrix zero_weights(1, 33);
        zero_weights.zero();
        const QuantizedMatrix zero_int4 =
            QuantizedMatrix::quantize(zero_weights, QuantType::int4);
        float int8_ratio = 0.0f;
        float int4_ratio = 0.0f;
        float zero_ratio = 0.0f;
        const bool roundtrip_ok =
            roundtrip_within_bound(weights, int8, int8_ratio) &&
            roundtrip_within_bound(weights, int4, int4_ratio) &&
            roundtrip_within_bound(zero_weights, zero_int4, zero_ratio);

        Matrix packing_source(1, 4);
        packing_source.at(0, 0) = -7.0f;
        packing_source.at(0, 1) = -1.0f;
        packing_source.at(0, 2) = 2.0f;
        packing_source.at(0, 3) = 7.0f;
        const QuantizedMatrix packing =
            QuantizedMatrix::quantize(packing_source, QuantType::int4);
        const bool packing_ok =
            packing.packed_byte(0) == 0xf9 &&
            packing.packed_byte(1) == 0x72;

        Matrix left(3, 5);
        fill_random(left, 606);
        float int8_matmul_ratio = 0.0f;
        float int4_matmul_ratio = 0.0f;
        const bool matmul_ok =
            matmul_within_bound(left, weights, int8, int8_matmul_ratio) &&
            matmul_within_bound(left, weights, int4, int4_matmul_ratio);

        std::vector<float> vector(weights.cols(), 0.25f);
        std::vector<float> reference(weights.rows());
        std::vector<float> candidate(weights.rows());
        matvec(weights, vector.data(), reference.data());
        matvec_quantized(int8, vector.data(), candidate.data());
        bool matvec_ok = true;
        for (size_t row = 0; row < weights.rows(); row++) {
            double bound = 0.0;
            for (size_t column = 0; column < weights.cols(); column++)
                bound += 0.25 * int8.error_bound(row, column);
            matvec_ok =
                matvec_ok &&
                std::fabs(double(reference[row] - candidate[row])) <=
                    bound + 1e-5;
        }

        const size_t blocks = (weights.size() + 31) / 32;
        const bool storage_ok =
            int8.storage_bytes() == weights.size() + blocks * 2 &&
            int4.storage_bytes() ==
                (weights.size() + 1) / 2 + blocks * 2;

        Matrix sample_activations(12, 5);
        fill_random(sample_activations, 707);
        Matrix fp32_logits(12, 37);
        Matrix int8_logits(12, 37);
        Matrix int4_logits(12, 37);
        matmul_naive(sample_activations, weights, fp32_logits);
        matmul_quantized(sample_activations, int8, int8_logits);
        matmul_quantized(sample_activations, int4, int4_logits);
        std::vector<size_t> targets(sample_activations.rows());
        for (size_t row = 0; row < fp32_logits.rows(); row++) {
            size_t best = 0;
            for (size_t column = 1; column < fp32_logits.cols(); column++)
                if (fp32_logits.at(row, column) >
                    fp32_logits.at(row, best))
                    best = column;
            targets[row] = best;
        }
        const double fp32_perplexity = perplexity(fp32_logits, targets);
        const double int8_perplexity = perplexity(int8_logits, targets);
        const double int4_perplexity = perplexity(int4_logits, targets);
        const bool perplexity_ok =
            std::isfinite(fp32_perplexity) &&
            std::isfinite(int8_perplexity) &&
            std::isfinite(int4_perplexity);
        std::printf(
            "  %s INT8/INT4 roundtrip bounds ratios=%.3f/%.3f\n",
            roundtrip_ok ? "ok" : "FAIL", double(int8_ratio),
            double(int4_ratio));
        std::printf(
            "  %s quantized matmul bounds    ratios=%.3f/%.3f\n",
            matmul_ok && matvec_ok ? "ok" : "FAIL",
            double(int8_matmul_ratio), double(int4_matmul_ratio));
        std::printf(
            "  %s INT4 packing/storage       bytes fp32=%zu i8=%zu i4=%zu\n",
            packing_ok && storage_ok ? "ok" : "FAIL",
            weights.size() * sizeof(float), int8.storage_bytes(),
            int4.storage_bytes());
        std::printf(
            "  %s synthetic perplexity       fp32=%.3f i8=%.3f i4=%.3f\n",
            perplexity_ok ? "ok" : "FAIL", fp32_perplexity,
            int8_perplexity, int4_perplexity);
        return roundtrip_ok && packing_ok && matmul_ok && matvec_ok &&
               storage_ok && perplexity_ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL quantization selftest: %s\n", error.what());
        return false;
    }
}
