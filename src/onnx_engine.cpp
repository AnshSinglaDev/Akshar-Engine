#include "onnx_engine.h"
#include "text_utils.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace byt5 {

static inline int64_t fast_argmax(const float* logits, size_t size) {
    int64_t best_idx = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_idx = static_cast<int64_t>(i);
        }
    }
    return best_idx;
}

static std::string locate_model_file(const fs::path& dir, const std::string& preferred, const std::string& fallback) {
    if (fs::exists(dir / preferred)) return (dir / preferred).string();
    if (fs::exists(dir / fallback)) return (dir / fallback).string();
    throw std::runtime_error("Could not locate ONNX model file: " + preferred + " (or " + fallback + ") in " + dir.string());
}

struct OnnxEngine::Impl {
    OnnxConfig config;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> enc_session_;
    std::unique_ptr<Ort::Session> dec_init_session_;
    std::unique_ptr<Ort::Session> dec_step_session_;

    std::vector<std::string> in_names_str_;
    std::vector<const char*> step_in_names_;
    std::vector<std::string> out_names_str_;
    std::vector<const char*> step_out_names_;

    std::string device_name_;
    int active_threads_ = 0;

    explicit Impl(const OnnxConfig& cfg)
        : config(cfg), env_(ORT_LOGGING_LEVEL_WARNING, "ByT5_ONNX_Engine") {
        
        fs::path dir(config.model_dir);
        if (!fs::exists(dir)) {
            throw std::runtime_error("ONNX model directory does not exist: " + config.model_dir);
        }

        std::string enc_path = locate_model_file(dir, "encoder_model_quantized.onnx", "encoder_model.onnx");
        std::string dec_init_path = locate_model_file(dir, "decoder_model_quantized.onnx", "decoder_model.onnx");
        std::string dec_step_path = locate_model_file(dir, "decoder_with_past_model_quantized.onnx", "decoder_with_past_model.onnx");

        int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        active_threads_ = (config.n_threads > 0) ? config.n_threads : (hw_threads > 0 ? hw_threads : 8);

        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(active_threads_);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        bool dml_active = false;
        if (config.device == "dml" || config.device == "directml" || config.device == "gpu") {
            const OrtDmlApi* dml_api = nullptr;
            if (Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api)) == nullptr && dml_api != nullptr) {
                opts.DisableMemPattern();
                opts.DisableCpuMemArena();
                dml_api->SessionOptionsAppendExecutionProvider_DML(opts, 0);
                dml_active = true;
                device_name_ = "DirectML GPU (DirectX 12)";
            } else {
                std::cerr << "[ONNX Engine Warning] DirectML execution provider unavailable, falling back to CPU." << std::endl;
            }
        }

        if (!dml_active) {
            opts.EnableCpuMemArena();
            opts.EnableMemPattern();
            device_name_ = "CPU AVX2 SIMD (" + std::to_string(active_threads_) + " threads)";
        }

        int num_layers = (config.n_layers > 0) ? config.n_layers : 4;

#ifdef _WIN32
        std::wstring w_enc = fs::path(enc_path).wstring();
        std::wstring w_dec_init = fs::path(dec_init_path).wstring();
        std::wstring w_dec_step = fs::path(dec_step_path).wstring();

        enc_session_ = std::make_unique<Ort::Session>(env_, w_enc.c_str(), opts);
        dec_init_session_ = std::make_unique<Ort::Session>(env_, w_dec_init.c_str(), opts);
        dec_step_session_ = std::make_unique<Ort::Session>(env_, w_dec_step.c_str(), opts);
#else
        enc_session_ = std::make_unique<Ort::Session>(env_, enc_path.c_str(), opts);
        dec_init_session_ = std::make_unique<Ort::Session>(env_, dec_init_path.c_str(), opts);
        dec_step_session_ = std::make_unique<Ort::Session>(env_, dec_step_path.c_str(), opts);
#endif

        // Pre-build input/output names for step decoder execution dynamically based on layer count
        in_names_str_ = {"input_ids", "encoder_attention_mask"};
        for (int i = 0; i < num_layers; ++i) {
            in_names_str_.push_back("past_key_values." + std::to_string(i) + ".decoder.key");
            in_names_str_.push_back("past_key_values." + std::to_string(i) + ".decoder.value");
            in_names_str_.push_back("past_key_values." + std::to_string(i) + ".encoder.key");
            in_names_str_.push_back("past_key_values." + std::to_string(i) + ".encoder.value");
        }
        for (const auto& s : in_names_str_) step_in_names_.push_back(s.c_str());

        out_names_str_ = {"logits"};
        for (int i = 0; i < num_layers; ++i) {
            out_names_str_.push_back("present." + std::to_string(i) + ".decoder.key");
            out_names_str_.push_back("present." + std::to_string(i) + ".decoder.value");
        }
        for (const auto& s : out_names_str_) step_out_names_.push_back(s.c_str());
    }

    std::string generate_single_chunk(const std::string& text, size_t max_tokens) {
        if (text.empty()) return "";

        int num_layers = (config.n_layers > 0) ? config.n_layers : 4;
        constexpr int64_t BYT5_PAD_TOKEN = 0;
        constexpr int64_t BYT5_EOS_TOKEN = 1;
        constexpr int64_t BYT5_BYTE_OFFSET = 3;
        constexpr int64_t BYT5_BYTE_MAX = 258;
        constexpr size_t BYT5_VOCAB_SIZE = 384;

        std::vector<int64_t> input_tokens;
        input_tokens.reserve(text.size() + 1);
        for (unsigned char b : text) {
            input_tokens.push_back(static_cast<int64_t>(b) + BYT5_BYTE_OFFSET);
        }
        input_tokens.push_back(BYT5_EOS_TOKEN);

        int64_t seq_len = static_cast<int64_t>(input_tokens.size());
        std::vector<int64_t> input_shape = {1, seq_len};
        std::vector<int64_t> attention_mask(seq_len, 1);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, input_tokens.data(), input_tokens.size(), input_shape.data(), input_shape.size());
        Ort::Value attn_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

        const char* enc_in_names[] = {"input_ids", "attention_mask"};
        Ort::Value enc_inputs[] = {std::move(input_ids_tensor), std::move(attn_mask_tensor)};
        const char* enc_out_names[] = {"last_hidden_state"};

        // 1. Run Encoder
        auto enc_outputs = enc_session_->Run(
            Ort::RunOptions{nullptr}, enc_in_names, enc_inputs, 2, enc_out_names, 1);
        Ort::Value& enc_hidden_states = enc_outputs[0];

        // 2. Decoder Initial Step (token = 0)
        int64_t start_token = BYT5_PAD_TOKEN;
        int64_t dec_init_shape[] = {1, 1};
        Ort::Value dec_input_ids = Ort::Value::CreateTensor<int64_t>(
            mem_info, &start_token, 1, dec_init_shape, 2);
        Ort::Value dec_attn_mask = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

        const char* dec_init_in_names[] = {"input_ids", "encoder_attention_mask", "encoder_hidden_states"};
        Ort::Value dec_init_inputs[] = {
            std::move(dec_input_ids),
            std::move(dec_attn_mask),
            std::move(enc_hidden_states)
        };

        size_t total_init_outs = 1 + static_cast<size_t>(num_layers * 4);
        std::vector<const char*> dec_init_out_names(total_init_outs);
        dec_init_out_names[0] = "logits";
        std::vector<std::string> init_out_str_names;
        init_out_str_names.reserve(num_layers * 4);
        for (int i = 0; i < num_layers; ++i) {
            init_out_str_names.push_back("present." + std::to_string(i) + ".decoder.key");
            init_out_str_names.push_back("present." + std::to_string(i) + ".decoder.value");
            init_out_str_names.push_back("present." + std::to_string(i) + ".encoder.key");
            init_out_str_names.push_back("present." + std::to_string(i) + ".encoder.value");
        }
        for (size_t i = 0; i < init_out_str_names.size(); ++i) {
            dec_init_out_names[1 + i] = init_out_str_names[i].c_str();
        }

        auto dec_init_outputs = dec_init_session_->Run(
            Ort::RunOptions{nullptr}, dec_init_in_names, dec_init_inputs, 3, dec_init_out_names.data(), total_init_outs);

        // Fast Argmax
        float* logits = dec_init_outputs[0].GetTensorMutableData<float>();
        int64_t next_tok = fast_argmax(logits, BYT5_VOCAB_SIZE);

        std::vector<int64_t> generated;
        generated.reserve(max_tokens);
        generated.push_back(BYT5_PAD_TOKEN);
        if (next_tok == BYT5_EOS_TOKEN) return "";
        generated.push_back(next_tok);

        // Separate decoder KV and encoder KV
        std::vector<Ort::Value> dec_kv;
        dec_kv.reserve(num_layers * 2);
        std::vector<Ort::Value> enc_kv;
        enc_kv.reserve(num_layers * 2);
        for (int i = 0; i < num_layers; ++i) {
            dec_kv.push_back(std::move(dec_init_outputs[1 + i * 4 + 0]));
            dec_kv.push_back(std::move(dec_init_outputs[1 + i * 4 + 1]));
            enc_kv.push_back(std::move(dec_init_outputs[1 + i * 4 + 2]));
            enc_kv.push_back(std::move(dec_init_outputs[1 + i * 4 + 3]));
        }

        // 3. Autoregressive Loop with Raw C-API KV-Cache Pointer Preservation
        size_t total_step_inputs = 2 + static_cast<size_t>(num_layers * 4);
        size_t total_step_outputs = 1 + static_cast<size_t>(num_layers * 2);
        std::vector<const OrtValue*> c_inputs(total_step_inputs, nullptr);
        std::vector<OrtValue*> c_outputs(total_step_outputs, nullptr);

        for (size_t step = 1; step < max_tokens; ++step) {
            Ort::Value cur_tok_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, &next_tok, 1, dec_init_shape, 2);
            Ort::Value cur_attn_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

            std::vector<Ort::Value> input_copies;
            input_copies.reserve(total_step_inputs);
            input_copies.push_back(std::move(cur_tok_tensor));
            input_copies.push_back(std::move(cur_attn_tensor));
            for (int i = 0; i < num_layers; ++i) {
                input_copies.push_back(std::move(dec_kv[i * 2 + 0]));
                input_copies.push_back(std::move(dec_kv[i * 2 + 1]));
                input_copies.push_back(Ort::Value{nullptr});
                input_copies.push_back(Ort::Value{nullptr});
            }

            c_inputs[0] = static_cast<const OrtValue*>(input_copies[0]);
            c_inputs[1] = static_cast<const OrtValue*>(input_copies[1]);
            for (int i = 0; i < num_layers; ++i) {
                c_inputs[2 + i * 4 + 0] = static_cast<const OrtValue*>(input_copies[2 + i * 4 + 0]);
                c_inputs[2 + i * 4 + 1] = static_cast<const OrtValue*>(input_copies[2 + i * 4 + 1]);
                c_inputs[2 + i * 4 + 2] = static_cast<const OrtValue*>(enc_kv[i * 2 + 0]);
                c_inputs[2 + i * 4 + 3] = static_cast<const OrtValue*>(enc_kv[i * 2 + 1]);
            }

            std::fill(c_outputs.begin(), c_outputs.end(), nullptr);
            Ort::ThrowOnError(Ort::GetApi().Run(
                *dec_step_session_,
                nullptr,
                step_in_names_.data(),
                c_inputs.data(),
                total_step_inputs,
                step_out_names_.data(),
                total_step_outputs,
                c_outputs.data()
            ));

            std::vector<Ort::Value> step_outputs;
            step_outputs.reserve(total_step_outputs);
            for (size_t i = 0; i < total_step_outputs; ++i) {
                step_outputs.push_back(Ort::Value(c_outputs[i]));
            }

            float* s_logits = step_outputs[0].GetTensorMutableData<float>();
            next_tok = fast_argmax(s_logits, BYT5_VOCAB_SIZE);

            if (next_tok == BYT5_EOS_TOKEN) break;
            generated.push_back(next_tok);

            // Update decoder KV caches for next step
            dec_kv.clear();
            for (size_t i = 1; i < step_outputs.size(); ++i) {
                dec_kv.push_back(std::move(step_outputs[i]));
            }
        }

        std::string result;
        result.reserve(generated.size());
        for (size_t i = 1; i < generated.size(); ++i) {
            int64_t id = generated[i];
            if (id >= BYT5_BYTE_OFFSET && id <= BYT5_BYTE_MAX) {
                result.push_back(static_cast<char>(id - BYT5_BYTE_OFFSET));
            }
        }
        return result;
    }

    std::string generate(const std::string& prompt, size_t max_tokens) {
        if (prompt.empty()) return "";
        size_t token_limit = (max_tokens > 0) ? max_tokens : config.n_predict;

        std::vector<std::string> chunks = TextUtils::chunk_text(prompt, config.chunk_bytes);
        if (chunks.size() <= 1) {
            return generate_single_chunk(prompt, token_limit);
        }

        std::string combined;
        for (size_t i = 0; i < chunks.size(); ++i) {
            std::string part = generate_single_chunk(chunks[i], token_limit);
            TextUtils::trim(part);
            if (!part.empty()) {
                if (!combined.empty()) combined.push_back(' ');
                combined += part;
            }
        }

        return TextUtils::sanitize_utf8(combined);
    }
};

OnnxEngine::OnnxEngine(const OnnxConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

OnnxEngine::~OnnxEngine() = default;

OnnxEngine::OnnxEngine(OnnxEngine&&) noexcept = default;
OnnxEngine& OnnxEngine::operator=(OnnxEngine&&) noexcept = default;

std::string OnnxEngine::generate(const std::string& prompt, size_t max_tokens) {
    return impl_->generate(prompt, max_tokens);
}

std::vector<std::string> OnnxEngine::generate_batch(const std::vector<std::string>& prompts, size_t max_tokens) {
    std::vector<std::string> results;
    results.reserve(prompts.size());
    for (const auto& p : prompts) {
        results.push_back(generate(p, max_tokens));
    }
    return results;
}

const std::string& OnnxEngine::get_device_name() const {
    return impl_->device_name_;
}

} // namespace byt5
