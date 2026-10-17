#pragma once
#include <string>
#include <vector>
#include <memory>

namespace byt5 {

enum class DeviceMode {
    Auto,
    Vulkan,
    CPU
};

struct GenerationConfig {
    bool auto_terminate = true;        // Appends '।' if missing to eliminate IT hallucinations
    float length_ratio = 1.4f;         // Max output chars = max(32, input chars * length_ratio)
    int no_repeat_ngram = 3;           // Suppresses 3+ consecutive word repetition loops
    std::vector<std::string> hotwords; // Exact casing/spelling overrides for creator/brand names
    std::string case_style = "none";   // "none", "lower", "title", "sentence"
};

struct EngineConfig {
    std::string model_path;
    DeviceMode device = DeviceMode::Auto;
    int n_gpu_layers = 99;      // 99 = full Vulkan GPU offload, 0 = pure CPU
    int n_ctx = 1024;          // Context window (1024 accommodates full dense paragraphs)
    int n_predict = 256;       // Default generation limit
    int n_threads = 0;         // 0 = auto-detect hardware concurrency
    int n_parallel = 4;        // Number of concurrent execution slots (default: 4)
    size_t chunk_bytes = 750;  // Smart paragraph chunking threshold (bytes)
    GenerationConfig gen_config; // Decoding guardrails & hyperparameters
};

class ByT5Engine {
public:
    explicit ByT5Engine(const EngineConfig& config);
    ~ByT5Engine();

    // Disable copies, allow moves
    ByT5Engine(const ByT5Engine&) = delete;
    ByT5Engine& operator=(const ByT5Engine&) = delete;
    ByT5Engine(ByT5Engine&&) noexcept;
    ByT5Engine& operator=(ByT5Engine&&) noexcept;

    // Generates translated/transliterated Hinglish output from input text
    std::string generate(const std::string& prompt, size_t max_tokens = 0);

    // High-throughput batch generation for a collection of inputs
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts, size_t max_tokens = 0);

    void set_generation_config(const GenerationConfig& cfg);
    const GenerationConfig& get_generation_config() const;

    bool is_gpu_active() const;
    const std::string& get_device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace byt5
