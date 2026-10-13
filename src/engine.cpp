#include "engine.h"
#include "text_utils.h"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#include <llama.h>

namespace byt5 {

constexpr int32_t BYT5_PAD_TOKEN = 0;
constexpr int32_t BYT5_EOS_TOKEN = 1;
constexpr int32_t BYT5_BYTE_OFFSET = 3;
constexpr int32_t BYT5_BYTE_MAX = 258;

struct WorkerSlot {
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;
    llama_batch_ext* batch = nullptr;
};

struct ByT5Engine::Impl {
    EngineConfig config;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    std::vector<WorkerSlot> slots;
    std::string device_name;
    bool gpu_active = false;

    // Slot checkout synchronization for safe concurrent multi-threaded access
    std::mutex slots_mutex_;
    std::condition_variable slots_cv_;
    std::vector<bool> slot_in_use_;

    int acquire_slot() {
        std::unique_lock<std::mutex> lock(slots_mutex_);
        slots_cv_.wait(lock, [this]() {
            for (size_t i = 0; i < slot_in_use_.size(); ++i) {
                if (!slot_in_use_[i]) return true;
            }
            return false;
        });
        for (size_t i = 0; i < slot_in_use_.size(); ++i) {
            if (!slot_in_use_[i]) {
                slot_in_use_[i] = true;
                return static_cast<int>(i);
            }
        }
        return 0;
    }

    void release_slot(int slot_id) {
        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            if (slot_id >= 0 && slot_id < static_cast<int>(slot_in_use_.size())) {
                slot_in_use_[slot_id] = false;
            }
        }
        slots_cv_.notify_one();
    }

    explicit Impl(const EngineConfig& cfg) : config(cfg) {
        init();
    }

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        for (auto& slot : slots) {
            if (slot.batch) {
                llama_batch_ext_free(slot.batch);
                slot.batch = nullptr;
            }
            if (slot.smpl) {
                llama_sampler_free(slot.smpl);
                slot.smpl = nullptr;
            }
            if (slot.ctx) {
                llama_free(slot.ctx);
                slot.ctx = nullptr;
            }
        }
        slots.clear();

        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }

    void init() {
        // Silence internal ggml and llama backend debug logs to keep console clean
        ggml_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
        llama_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
        llama_backend_init();

        int target_ngl = 0;
        if (config.device == DeviceMode::Vulkan || config.device == DeviceMode::Auto) {
            target_ngl = (config.n_gpu_layers > 0) ? config.n_gpu_layers : 99;
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = target_ngl;

        int num_slots = (config.n_parallel > 0) ? config.n_parallel : 4;

        std::cerr << "[ByT5Engine] Loading model: " << config.model_path 
                  << " (Requested GPU layers: " << target_ngl << ", Parallel slots: " << num_slots << ")..." << std::endl;

        model = llama_model_load_from_file(config.model_path.c_str(), mparams);

        // Graceful fallback: If GPU fails in Auto mode, fall back to pure CPU
        if (!model && config.device == DeviceMode::Auto && target_ngl > 0) {
            std::cerr << "[ByT5Engine Warning] Vulkan GPU initialization failed. Falling back to CPU SIMD..." << std::endl;
            mparams.n_gpu_layers = 0;
            target_ngl = 0;
            model = llama_model_load_from_file(config.model_path.c_str(), mparams);
        }

        if (!model) {
            throw std::runtime_error("Failed to load GGUF model from: " + config.model_path);
        }

        vocab = llama_model_get_vocab(model);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = config.n_ctx;
        cparams.n_batch = config.n_ctx;
        cparams.n_ubatch = config.n_ctx;
        cparams.no_perf = true;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        unsigned int hw_threads = std::thread::hardware_concurrency();
        int total_threads = (hw_threads > 0) ? static_cast<int>(hw_threads) : 8;

        if (config.n_threads > 0) {
            cparams.n_threads = config.n_threads;
            cparams.n_threads_batch = config.n_threads;
        } else if (target_ngl == 0) {
            // Distribute CPU threads evenly among slots to prevent oversubscription
            int threads_per_slot = std::max(1, total_threads / num_slots);
            cparams.n_threads = threads_per_slot;
            cparams.n_threads_batch = threads_per_slot;
        } else {
            // GPU handles matrix computation; 1 host thread per slot eliminates thread-pool contention and driver mutex stalls
            cparams.n_threads = 1;
            cparams.n_threads_batch = 1;
        }

        slots.resize(num_slots);
        slot_in_use_.assign(num_slots, false);
        for (int i = 0; i < num_slots; ++i) {
            slots[i].ctx = llama_init_from_model(model, cparams);
            if (!slots[i].ctx) {
                cleanup();
                throw std::runtime_error("Failed to initialize llama context for slot " + std::to_string(i));
            }

            auto sparams = llama_sampler_chain_default_params();
            sparams.no_perf = true;
            slots[i].smpl = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(slots[i].smpl, llama_sampler_init_greedy());

            slots[i].batch = llama_batch_ext_init(slots[i].ctx);
        }

        if (target_ngl > 0) {
            gpu_active = true;
            device_name = "Vulkan GPU (Matrix Cores, " + std::to_string(num_slots) + " parallel streams)";
            std::cerr << "[ByT5Engine] Active backend: Pure Vulkan GPU Acceleration (" 
                      << num_slots << " parallel compute slots, FlashAttention active)." << std::endl;
        } else {
            gpu_active = false;
            device_name = "CPU SIMD (AVX2/AVX-512, " + std::to_string(num_slots) + " slots)";
            std::cerr << "[ByT5Engine] Active backend: Multi-threaded AVX2 CPU SIMD (" 
                      << num_slots << " slots x " << cparams.n_threads << " threads)." << std::endl;
        }
    }

    static void batch_set_tokens(llama_batch_ext* b, const llama_token* tokens, int32_t n_tokens, llama_pos pos_0, bool all_outputs = false) {
        llama_batch_ext_clear(b);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const int32_t idx = llama_batch_ext_add_token(b, 0, tokens[i]);
            const llama_pos pos = pos_0 + i;
            llama_batch_ext_set_pos(b, idx, &pos);
            if (all_outputs || i == n_tokens - 1) {
                llama_batch_ext_set_output_logits(b, idx, true);
            }
        }
    }

    std::string generate_single(const std::string& prompt, size_t max_tokens, int slot_id = 0) {
        if (prompt.empty()) return "";
        auto& slot = slots[slot_id];

        // Reset memory and sampler state (data=false resets KV heads without triggering heavy GPU VRAM memset barriers)
        llama_memory_clear(llama_get_memory(slot.ctx), false);
        llama_sampler_reset(slot.smpl);

        // Fast O(1) ByT5 Byte Tokenizer (Zero string queries, zero two-pass overhead)
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.reserve(prompt.size() + 1);
        for (unsigned char b : prompt) {
            prompt_tokens.push_back(static_cast<llama_token>(b) + BYT5_BYTE_OFFSET);
        }
        prompt_tokens.push_back(BYT5_EOS_TOKEN); // ByT5 mandatory EOS for encoder

        int n_tokens = static_cast<int>(prompt_tokens.size());
        batch_set_tokens(slot.batch, prompt_tokens.data(), n_tokens, 0, true);

        // Encoder pass
        if (llama_model_has_encoder(model)) {
            if (llama_process(slot.ctx, LLAMA_PROCESS_TYPE_ENCODE, slot.batch) != 0) {
                return "";
            }

            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }
            batch_set_tokens(slot.batch, &decoder_start_token_id, 1, 0, false);
            n_tokens = 1;
        }

        // Decoder generation loop
        std::string result;
        result.reserve(prompt.size() * 2 + 16);
        int limit = (max_tokens > 0) ? static_cast<int>(max_tokens) : config.n_predict;

        const llama_token eos_token_id = llama_vocab_eos(vocab);

        for (int n_pos = 0; n_pos + n_tokens < config.n_ctx && n_pos < limit; ) {
            if (llama_process(slot.ctx, LLAMA_PROCESS_TYPE_DECODE, slot.batch) != 0) {
                break;
            }

            n_pos += n_tokens;
            llama_token new_token_id = llama_sampler_sample(slot.smpl, slot.ctx, -1);

            // ByT5 EOS: Token 1, Vocab EOS, or EOG
            if (new_token_id == BYT5_EOS_TOKEN || new_token_id == eos_token_id || llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // Fast O(1) ByT5 Byte De-tokenizer (Zero vocab queries)
            if (new_token_id >= BYT5_BYTE_OFFSET && new_token_id <= BYT5_BYTE_MAX) {
                result.push_back(static_cast<char>(new_token_id - BYT5_BYTE_OFFSET));
            }

            batch_set_tokens(slot.batch, &new_token_id, 1, n_pos, false);
            n_tokens = 1;
        }

        return TextUtils::sanitize_utf8(result);
    }

    std::string generate_with_slot(const std::string& prompt, size_t max_tokens, int slot_id) {
        if (prompt.empty()) return "";

        size_t limit = (config.chunk_bytes > 0) ? config.chunk_bytes : 750;

        // If prompt exceeds chunk limit, cleanly chunk on sentence or word boundaries
        if (prompt.size() > limit) {
            std::vector<std::string> chunks = TextUtils::chunk_text(prompt, limit);
            if (chunks.size() > 1) {
                std::string combined;
                for (const auto& chunk : chunks) {
                    std::string part = generate_single(chunk, max_tokens, slot_id);
                    if (!part.empty()) {
                        if (!combined.empty() && combined.back() != ' ') combined.push_back(' ');
                        combined += part;
                    }
                }
                return combined;
            }
        }

        return generate_single(prompt, max_tokens, slot_id);
    }

    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts, size_t max_tokens) {
        if (prompts.empty()) return {};
        if (prompts.size() == 1 || slots.size() <= 1) {
            std::vector<std::string> res;
            res.reserve(prompts.size());
            for (const auto& p : prompts) {
                int slot_id = acquire_slot();
                try {
                    res.push_back(generate_with_slot(p, max_tokens, slot_id));
                    release_slot(slot_id);
                } catch (...) {
                    release_slot(slot_id);
                    throw;
                }
            }
            return res;
        }

        std::vector<std::string> results(prompts.size());
        std::atomic<size_t> next_idx{0};
        size_t num_workers = std::min(slots.size(), prompts.size());

        // Lock/acquire worker slots safely
        for (size_t i = 0; i < num_workers; ++i) {
            std::unique_lock<std::mutex> lock(slots_mutex_);
            slots_cv_.wait(lock, [this, i]() { return !slot_in_use_[i]; });
            slot_in_use_[i] = true;
        }

        std::vector<std::thread> workers;
        workers.reserve(num_workers);

        for (size_t slot_id = 0; slot_id < num_workers; ++slot_id) {
            workers.emplace_back([this, &prompts, &results, &next_idx, max_tokens, slot_id]() {
                while (true) {
                    size_t idx = next_idx.fetch_add(1);
                    if (idx >= prompts.size()) break;
                    results[idx] = generate_with_slot(prompts[idx], max_tokens, static_cast<int>(slot_id));
                }
            });
        }

        for (auto& w : workers) {
            w.join();
        }

        // Release worker slots safely
        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            for (size_t i = 0; i < num_workers; ++i) {
                slot_in_use_[i] = false;
            }
        }
        slots_cv_.notify_all();

        return results;
    }
};

ByT5Engine::ByT5Engine(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

ByT5Engine::~ByT5Engine() = default;
ByT5Engine::ByT5Engine(ByT5Engine&&) noexcept = default;
ByT5Engine& ByT5Engine::operator=(ByT5Engine&&) noexcept = default;

std::string ByT5Engine::generate(const std::string& prompt, size_t max_tokens) {
    int slot_id = impl_->acquire_slot();
    std::string result;
    try {
        result = impl_->generate_with_slot(prompt, max_tokens, slot_id);
    } catch (...) {
        impl_->release_slot(slot_id);
        throw;
    }
    impl_->release_slot(slot_id);
    return result;
}

std::vector<std::string> ByT5Engine::generate_batch(const std::vector<std::string>& prompts, size_t max_tokens) {
    return impl_->generate_batch(prompts, max_tokens);
}

bool ByT5Engine::is_gpu_active() const {
    return impl_->gpu_active;
}

const std::string& ByT5Engine::get_device_name() const {
    return impl_->device_name;
}

} // namespace byt5
