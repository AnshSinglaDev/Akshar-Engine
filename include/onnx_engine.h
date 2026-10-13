#pragma once
#include <string>
#include <vector>
#include <memory>

namespace byt5 {

struct OnnxConfig {
    std::string model_dir;       // Directory containing quantized ONNX models
    std::string device = "cpu";  // "cpu" (AVX2 SIMD) or "dml" (DirectML)
    int n_threads = 0;           // 0 = auto-detect hardware concurrency
    int n_predict = 256;         // Max tokens per chunk
    int n_layers = 4;            // Number of decoder layers (4 for Small, 12 for Base, 24 for Large)
    size_t chunk_bytes = 750;    // Smart paragraph chunking threshold (bytes)
};

class OnnxEngine {
public:
    explicit OnnxEngine(const OnnxConfig& config);
    ~OnnxEngine();

    // Disable copies, allow moves
    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;
    OnnxEngine(OnnxEngine&&) noexcept;
    OnnxEngine& operator=(OnnxEngine&&) noexcept;

    // Generates translated/transliterated Hinglish output from input text
    std::string generate(const std::string& prompt, size_t max_tokens = 0);

    // High-throughput batch generation for a collection of inputs
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts, size_t max_tokens = 0);

    const std::string& get_device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace byt5
