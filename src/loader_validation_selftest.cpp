/* Network-free loader boundary regressions. Rejections check diagnostics so
 * unrelated errors cannot hide a regression in validation order. */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "llama2.hpp"
#include "loader.hpp"
#include "selftest.hpp"

namespace {

class Fixture {
public:
    Fixture() {
        char pattern[] = "build/loader-validation-XXXXXX";
        const int descriptor = mkstemp(pattern);
        if (descriptor < 0)
            throw std::runtime_error("cannot create loader validation fixture");
        close(descriptor);
        path_ = pattern;
    }
    ~Fixture() { std::remove(path_.c_str()); }
    const std::string &path() const { return path_; }
    void write(const std::string &bytes) const {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), std::streamsize(bytes.size()));
        output.close();
        if (!output)
            throw std::runtime_error("cannot write loader validation fixture");
    }
private:
    std::string path_;
};

std::string integer_bytes(uint64_t value, size_t count) {
    std::string bytes(count, '\0');
    for (size_t byte = 0; byte < count; byte++)
        bytes[byte] = char((value >> (byte * 8)) & 0xff);
    return bytes;
}

std::string container(std::string header, size_t payload = 4) {
    while (header.size() % 8 != 0)
        header += ' ';
    return integer_bytes(header.size(), 8) + header + std::string(payload, '\0');
}

std::string descriptor(const std::string &shape, const std::string &offsets) {
    return "{\"dtype\":\"F32\",\"shape\":" + shape +
           ",\"data_offsets\":" + offsets + "}";
}

bool rejected(const char *name, const std::string &expected,
              const std::function<void()> &function) {
    try {
        function();
    } catch (const std::exception &error) {
        const bool pass = std::string(error.what()).find(expected) != std::string::npos;
        if (!pass)
            std::printf("  FAIL %s: unexpected error: %s\n", name, error.what());
        return pass;
    }
    std::printf("  FAIL %s: malformed input accepted\n", name);
    return false;
}

std::string read_fixture(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read loader fixture");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string changed_metadata(const std::string &key, const std::string &value) {
    const std::string bytes = read_fixture("tests/fixtures/tiny.safetensors");
    uint64_t length = 0;
    for (size_t byte = 0; byte < 8; byte++)
        length |= uint64_t(static_cast<unsigned char>(bytes[byte])) << (byte * 8);
    std::string header = bytes.substr(8, size_t(length));
    const size_t key_start = header.find("\"nutllm." + key + "\"");
    if (key_start == std::string::npos)
        throw std::runtime_error("missing fixture metadata");
    const size_t start = header.find('"', header.find(':', key_start)) + 1;
    const size_t end = header.find('"', start);
    header.replace(start, end - start, value);
    while (header.size() % 8 != 0)
        header += ' ';
    return integer_bytes(header.size(), 8) + header + bytes.substr(8 + size_t(length));
}

} // namespace

bool run_loader_validation_selftests() {
    try {
        Fixture fixture;
        bool ok = true;
        size_t checks = 0;
        const auto reject_container = [&](const char *name, const std::string &bytes,
                                           const std::string &diagnostic) {
            fixture.write(bytes);
            ok = rejected(name, diagnostic, [&]() {
                (void)SafeTensorFile::load(fixture.path());
            }) && ok;
            checks++;
        };
        const std::string one = descriptor("[1]", "[0,4]");
        reject_container("short prefix", "abc", "truncated length");
        reject_container("oversized header", integer_bytes(100000001, 8), "header exceeds size limit");
        reject_container("truncated header", integer_bytes(1024, 8) + "{}", "truncated header");
        reject_container("leading header whitespace", container(" {\"x\":" + one + "}"), "header must start");
        reject_container("deep arrays", container("{\"x\":" + std::string(64, '[') +
                         "0" + std::string(64, ']') + "}"), "nesting limit");
        std::string deep = "0";
        for (size_t level = 0; level < 64; level++)
            deep = "{\"x\":" + deep + "}";
        reject_container("deep objects", container(deep), "nesting limit");
        reject_container("duplicate keys", container("{\"x\":" + one + ",\"x\":" + one + "}"), "duplicate object key");
        reject_container("escaped duplicate", container("{\"x\":" + one + ",\"\\u0078\":" + one + "}"), "duplicate object key");
        reject_container("overlap", container("{\"x\":" + one + ",\"y\":" + one + "}"), "overlapping or gapped");
        reject_container("initial gap", container("{\"x\":" + descriptor("[1]", "[4,8]") + "}", 8), "overlapping or gapped");
        reject_container("interior gap", container("{\"x\":" + one + ",\"y\":" + descriptor("[1]", "[8,12]") + "}", 12), "overlapping or gapped");
        reject_container("trailing payload", container("{\"x\":" + one + "}", 8), "unindexed trailing");
        reject_container("leading zero", container("{\"x\":" + descriptor("[01]", "[0,4]") + "}"), "leading zero");
        reject_container("integer overflow", container("{\"x\":" + descriptor("[18446744073709551616]", "[0,4]") + "}"), "integer overflow");
        reject_container("shape overflow", container("{\"x\":" + descriptor("[18446744073709551615,2]", "[0,4]") + "}"), "invalid tensor shape");
        reject_container("misaligned extent", container("{\"x\":" + descriptor("[1]", "[1,5]") + "}", 5), "invalid tensor extent");
        reject_container("out of bounds", container("{\"x\":" + descriptor("[2]", "[0,8]") + "}"), "invalid tensor extent");
        for (const std::string &invalid : {std::string("\xc0\xaf"), std::string("\xed\xa0\x80"),
                                           std::string("\xf4\x90\x80\x80"), std::string("\xe2x")})
            reject_container("invalid UTF-8", container("{\"" + invalid + "\":" + one + "}"), "invalid UTF-8");
        for (const std::string invalid : {"\\udc00", "\\ud800x", "\\ud800\\u0041"})
            reject_container("unpaired surrogate", container("{\"" + invalid + "\":" + one + "}"), "surrogate");
        reject_container("invalid unicode escape", container("{\"\\uxyz0\":" + one + "}"), "unicode escape");

        // Physical order can differ from key order. Escaped and literal UTF-8
        // names must resolve identically, including a supplementary codepoint.
        fixture.write(container("{\"a\":" + descriptor("[1]", "[4,8]") +
                       ",\"\\u00e9\\ud83c\\udf30\":" + one + "}", 8));
        {
            const SafeTensorFile valid = SafeTensorFile::load(fixture.path());
            ok = valid.tensor_count() == 2 &&
                 valid.tensor("\xc3\xa9\xf0\x9f\x8c\xb0").elements == 1 && ok;
        }
        checks++;
        fixture.write(container("{\"\xc3\xa9\xf0\x9f\x8c\xb0\":" + one + "}"));
        ok = SafeTensorFile::load(fixture.path()).tensor_count() == 1 && ok;
        checks++;
        for (const auto &test : std::vector<std::pair<std::string, std::string>>{
                 {"layers", "1000000000"}, {"vocab_size", "2147483647"}, {"ffn_dim", "1000000000000"}}) {
            fixture.write(changed_metadata(test.first, test.second));
            ok = rejected("metadata before allocation", test.first == "layers" ? "missing layer tensors" : "wrong shape",
                          [&]() { (void)load_model(fixture.path()); }) && ok;
            checks++;
        }

        const std::string tokenizer = read_fixture("tests/fixtures/tiny-llama2.tokenizer");
        const size_t vocabulary = load_llama2c_model("tests/fixtures/tiny-llama2.bin").config.vocab_size;
        const auto reject_tokenizer = [&](const char *name, const std::string &bytes,
                                           size_t count, const std::string &diagnostic) {
            fixture.write(bytes);
            ok = rejected(name, diagnostic, [&]() {
                (void)Llama2Tokenizer::load(fixture.path(), count);
            }) && ok;
            checks++;
        };
        reject_tokenizer("tokenizer minimum vocab", tokenizer, 1, "invalid vocabulary");
        reject_tokenizer("tokenizer count before reserve", tokenizer, size_t(std::numeric_limits<int>::max()), "truncated vocabulary");
        std::string huge_token = tokenizer;
        huge_token.replace(0, 4, integer_bytes(2147483647, 4));
        huge_token.replace(8, 4, integer_bytes(2147483647, 4));
        reject_tokenizer("token bytes before allocation", huge_token, vocabulary, "truncated token");
        reject_tokenizer("tokenizer truncation", tokenizer.substr(0, tokenizer.size() - 1), vocabulary, "truncated token");
        reject_tokenizer("tokenizer trailing bytes", tokenizer + "x", vocabulary, "trailing data");
        std::string duplicates = integer_bytes(1, 4);
        for (size_t id = 0; id < 3; id++)
            duplicates += integer_bytes(0, 4) + integer_bytes(1, 4) + "x";
        reject_tokenizer("duplicate vocabulary", duplicates, 3, "duplicate token");
        std::printf("  %s loader boundary regressions (%zu checks)\n", ok ? "ok" : "FAIL", checks);
        return ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL loader validation selftest: %s\n", error.what());
        return false;
    }
}
