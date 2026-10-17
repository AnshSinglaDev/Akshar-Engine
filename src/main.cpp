#include "engine.h"
#include "onnx_engine.h"
#include "text_utils.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <memory>
#include <clocale>
#include <iomanip>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

// Generic abstract engine interface for unified CLI dispatch
class IEngine {
public:
    virtual ~IEngine() = default;
    virtual std::string generate(const std::string& text, size_t max_tokens) = 0;
    virtual std::vector<std::string> generate_batch(const std::vector<std::string>& texts, size_t max_tokens) = 0;
    virtual std::string get_device_name() const = 0;
    virtual void set_generation_config(const byt5::GenerationConfig& cfg) {}
};

class VulkanEngineAdapter : public IEngine {
    byt5::ByT5Engine engine_;
public:
    explicit VulkanEngineAdapter(const byt5::EngineConfig& cfg) : engine_(cfg) {}
    std::string generate(const std::string& text, size_t max_tokens) override {
        return engine_.generate(text, max_tokens);
    }
    std::vector<std::string> generate_batch(const std::vector<std::string>& texts, size_t max_tokens) override {
        return engine_.generate_batch(texts, max_tokens);
    }
    std::string get_device_name() const override {
        return engine_.get_device_name();
    }
    void set_generation_config(const byt5::GenerationConfig& cfg) override {
        engine_.set_generation_config(cfg);
    }
};

class OnnxEngineAdapter : public IEngine {
    byt5::OnnxEngine engine_;
    byt5::GenerationConfig gen_config_;
public:
    explicit OnnxEngineAdapter(const byt5::OnnxConfig& cfg) : engine_(cfg) {}
    void set_generation_config(const byt5::GenerationConfig& cfg) override {
        gen_config_ = cfg;
    }
    std::string generate(const std::string& text, size_t max_tokens) override {
        std::string effective = gen_config_.auto_terminate ? byt5::TextUtils::ensure_sentence_termination(text) : text;
        std::string out = engine_.generate(effective, max_tokens);
        if (gen_config_.auto_terminate) byt5::TextUtils::clean_generated_output(out);
        if (gen_config_.no_repeat_ngram > 0) out = byt5::TextUtils::deduplicate_words(out, gen_config_.no_repeat_ngram - 1);
        if (!gen_config_.hotwords.empty()) out = byt5::TextUtils::apply_hotwords(out, gen_config_.hotwords);
        if (gen_config_.case_style != "none" && !gen_config_.case_style.empty()) out = byt5::TextUtils::format_case(out, gen_config_.case_style);
        return out;
    }
    std::vector<std::string> generate_batch(const std::vector<std::string>& texts, size_t max_tokens) override {
        std::vector<std::string> res;
        res.reserve(texts.size());
        for (const auto& t : texts) res.push_back(generate(t, max_tokens));
        return res;
    }
    std::string get_device_name() const override {
        return engine_.get_device_name();
    }
};

static fs::path get_executable_directory() {
#if defined(_WIN32)
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return fs::path(path).parent_path();
    }
#endif
    return fs::current_path();
}

static std::string resolve_gguf_path(const std::string& user_path) {
    if (!user_path.empty() && fs::exists(user_path) && !fs::is_directory(user_path)) {
        return user_path;
    }
    const char* env_path = std::getenv("AKSHAR_MODEL_PATH");
    if (!env_path) env_path = std::getenv("BYT5_GGUF_MODEL");
    if (env_path && fs::exists(env_path) && !fs::is_directory(env_path)) {
        return env_path;
    }

    fs::path exe_dir = get_executable_directory();
    fs::path cwd = fs::current_path();

    const std::vector<fs::path> candidates = {
        exe_dir / "hindi-to-hinglish-pro.q8_0.gguf",
        exe_dir / "models" / "hindi-to-hinglish-pro.q8_0.gguf",
        cwd / "hindi-to-hinglish-pro.q8_0.gguf",
        cwd / "models" / "hindi-to-hinglish-pro.q8_0.gguf",
        exe_dir / "byt5_v3_120k_q8_0.gguf",
        cwd / "byt5_v3_120k_q8_0.gguf"
    };

    for (const auto& c : candidates) {
        if (fs::exists(c) && !fs::is_directory(c)) {
            return c.string();
        }
    }
    return "";
}

static std::string resolve_onnx_dir(const std::string& user_path) {
    if (!user_path.empty() && fs::exists(user_path) && fs::is_directory(user_path)) {
        return user_path;
    }
    const char* env_dir = std::getenv("AKSHAR_ONNX_DIR");
    if (!env_dir) env_dir = std::getenv("BYT5_ONNX_DIR");
    if (env_dir && fs::exists(env_dir) && fs::is_directory(env_dir)) {
        return env_dir;
    }

    fs::path exe_dir = get_executable_directory();
    fs::path cwd = fs::current_path();

    const std::vector<fs::path> candidates = {
        exe_dir / "onnx",
        exe_dir / "models" / "onnx",
        cwd / "onnx",
        cwd / "models" / "onnx",
        exe_dir / "byt5_onnx_q8",
        cwd / "byt5_onnx_q8"
    };

    for (const auto& c : candidates) {
        if (fs::exists(c) && fs::is_directory(c)) {
            return c.string();
        }
    }
    return "";
}

static bool download_file_with_curl_or_ps(const std::string& url, const fs::path& dest_file, size_t min_bytes) {
    std::string cmd = "curl.exe -L --progress-bar -o \"" + dest_file.string() + "\" \"" + url + "\"";
    int ret = std::system(cmd.c_str());

    if (ret == 0 && fs::exists(dest_file) && fs::file_size(dest_file) >= min_bytes) {
        return true;
    }

    // Fallback via PowerShell
    std::cout << "[Notice] Retrying via PowerShell WebClient...\n";
    std::string ps_cmd = "powershell -Command \"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('" + url + "', '" + dest_file.string() + "')\"";
    ret = std::system(ps_cmd.c_str());

    if (ret == 0 && fs::exists(dest_file) && fs::file_size(dest_file) >= min_bytes) {
        return true;
    }

    if (fs::exists(dest_file)) {
        std::error_code ec;
        fs::remove(dest_file, ec);
    }
    return false;
}

static bool download_gguf_model(const fs::path& target_file) {
    const std::string url = "https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/resolve/main/gguf/hindi-to-hinglish-pro.q8_0.gguf";

    std::cout << "\n------------------------------------------------------------------------\n"
              << "  DOWNLOADING OFFICIAL GGUF MODEL FROM HUGGING FACE\n"
              << "------------------------------------------------------------------------\n"
              << "  Model  : hindi-to-hinglish-pro.q8_0.gguf (~303 MB)\n"
              << "  Source : " << url << "\n"
              << "  Save to: " << target_file.string() << "\n"
              << "------------------------------------------------------------------------\n\n";

    if (download_file_with_curl_or_ps(url, target_file, 100 * 1024 * 1024)) {
        std::cout << "\n[OK] Model successfully downloaded (" << (fs::file_size(target_file) / (1024 * 1024)) << " MB)!\n"
                  << "Saved to: " << target_file.string() << "\n\n";
        return true;
    }

    std::cerr << "\n[Error] GGUF download failed. Please check internet connection.\n";
    return false;
}

static bool download_onnx_model(const fs::path& target_dir) {
    std::error_code ec;
    fs::create_directories(target_dir, ec);

    struct OnnxFileInfo {
        std::string filename;
        std::string desc;
        size_t min_bytes;
    };

    const std::vector<OnnxFileInfo> files = {
        {"encoder_model_quantized.onnx", "Encoder INT8 (~218 MB)", 100 * 1024 * 1024},
        {"decoder_model_quantized.onnx", "Decoder Init INT8 (~83 MB)", 40 * 1024 * 1024},
        {"decoder_with_past_model_quantized.onnx", "Decoder KV-Cache INT8 (~78 MB)", 40 * 1024 * 1024}
    };

    std::cout << "\n------------------------------------------------------------------------\n"
              << "  DOWNLOADING OFFICIAL ONNX MODEL SUITE FROM HUGGING FACE\n"
              << "------------------------------------------------------------------------\n"
              << "  Suite  : ByT5 Quantized INT8 ONNX (3 models, ~380 MB total)\n"
              << "  Target : " << target_dir.string() << "\n"
              << "------------------------------------------------------------------------\n\n";

    const std::string base_url = "https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/resolve/main/onnx/";
    int step = 1;
    for (const auto& item : files) {
        fs::path dest = target_dir / item.filename;
        if (fs::exists(dest) && fs::file_size(dest) >= item.min_bytes) {
            std::cout << "  [" << step << "/3] " << item.filename << " already present, skipping.\n";
            step++;
            continue;
        }
        std::cout << "  [" << step << "/3] Downloading " << item.desc << "...\n";
        std::string file_url = base_url + item.filename;
        if (!download_file_with_curl_or_ps(file_url, dest, item.min_bytes)) {
            std::cerr << "\n[Error] Failed to download " << item.filename << ".\n";
            return false;
        }
        std::cout << "  ✔ " << item.filename << " saved.\n\n";
        step++;
    }

    std::cout << "\n[OK] All ONNX model files successfully downloaded to:\n"
              << target_dir.string() << "\n\n";
    return true;
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    std::setlocale(LC_ALL, ".UTF-8");
    SetConsoleTitleW(L"AKSHAR ENGINE (अक्षर) — Hindi to Hinglish Neural Runtime");
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif

    std::vector<std::string> args;
#if defined(_WIN32)
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        for (int i = 0; i < wargc; ++i) {
            int size = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
            if (size > 1) {
                std::string s(size - 1, '\0');
                WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], size, nullptr, nullptr);
                args.push_back(s);
            } else {
                args.push_back("");
            }
        }
        LocalFree(wargv);
    } else {
        for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
    }
#else
    for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
#endif

    std::string user_model_path;
    std::string dev_arg = "auto";
    int n_threads = 0;
    int n_predict = 256;
    int n_parallel = 4;
    std::string batch_input_file;
    std::string batch_output_file;
    std::string single_input_text;
    std::string download_target;
    bool quiet_mode = false;
    byt5::GenerationConfig gen_config;

    for (size_t i = 1; i < args.size(); ++i) {
        std::string arg = args[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < args.size()) {
            user_model_path = args[++i];
        } else if ((arg == "--device" || arg == "-d" || arg == "--backend") && i + 1 < args.size()) {
            dev_arg = args[++i];
        } else if ((arg == "--max-tokens" || arg == "-t") && i + 1 < args.size()) {
            n_predict = std::stoi(args[++i]);
        } else if ((arg == "--threads" || arg == "-j") && i + 1 < args.size()) {
            n_threads = std::stoi(args[++i]);
        } else if ((arg == "--parallel" || arg == "-p") && i + 1 < args.size()) {
            n_parallel = std::stoi(args[++i]);
        } else if ((arg == "--batch-file" || arg == "-b") && i + 1 < args.size()) {
            batch_input_file = args[++i];
        } else if ((arg == "--output-file" || arg == "-o") && i + 1 < args.size()) {
            batch_output_file = args[++i];
        } else if ((arg == "--text" || arg == "-s" || arg == "--input" || arg == "-i") && i + 1 < args.size()) {
            single_input_text = args[++i];
        } else if (arg == "--download" && i + 1 < args.size()) {
            download_target = args[++i];
        } else if (arg == "--quiet" || arg == "-q") {
            quiet_mode = true;
        } else if (arg == "--auto-terminate") {
            gen_config.auto_terminate = true;
        } else if (arg == "--no-auto-terminate") {
            gen_config.auto_terminate = false;
        } else if ((arg == "--length-ratio" || arg == "-l") && i + 1 < args.size()) {
            gen_config.length_ratio = std::stof(args[++i]);
        } else if ((arg == "--no-repeat-ngram" || arg == "-n") && i + 1 < args.size()) {
            gen_config.no_repeat_ngram = std::stoi(args[++i]);
        } else if ((arg == "--hotwords" || arg == "-w") && i + 1 < args.size()) {
            std::string hw_str = args[++i];
            size_t start = 0;
            while (start < hw_str.size()) {
                size_t comma = hw_str.find(',', start);
                std::string token = (comma == std::string::npos) ? hw_str.substr(start) : hw_str.substr(start, comma - start);
                byt5::TextUtils::trim(token);
                if (!token.empty()) gen_config.hotwords.push_back(token);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        } else if ((arg == "--case" || arg == "-c") && i + 1 < args.size()) {
            gen_config.case_style = args[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\n========================================================================\n"
                      << "  AKSHAR ENGINE (अक्षर) — HINDI TO HINGLISH NEURAL RUNTIME\n"
                      << "  Sub-15ms Real-Time Inference • Pure Vulkan GPU & CPU SIMD\n"
                      << "========================================================================\n\n"
                      << "Usage: akshar_engine [options] [model_path]\n\n"
                      << "Execution Options:\n"
                      << "  -d, --device <backend>   Target backend: 'auto', 'vulkan' (GPU), 'onnx' (CPU)\n"
                      << "  -m, --model <path>       Model path (.gguf file for Vulkan, directory for ONNX)\n"
                      << "  -s, --text <string>      Single sentence input (translates to stdout and exits)\n"
                      << "  -b, --batch-file <file>  Batch process lines from a file\n"
                      << "  -o, --output-file <file> Output destination for batch results (default: stdout)\n"
                      << "  -j, --threads <int>      CPU compute threads (default: hardware concurrency)\n"
                      << "  -p, --parallel <int>     GPU parallel compute slots for Vulkan (default: 4)\n"
                      << "  -t, --max-tokens <int>   Maximum generation tokens (default: 256)\n"
                      << "  --download <gguf|onnx>   Download official model weights from Hugging Face\n"
                      << "  -q, --quiet              Suppress status logs (clean stdout for app piping)\n"
                      << "  -h, --help               Show this help message\n\n"
                      << "Decoding & Guardrail Options:\n"
                      << "  --auto-terminate         Auto-terminate with '।' to kill hallucinations (default: ON)\n"
                      << "  --no-auto-terminate      Disable auto-termination guardrail\n"
                      << "  -l, --length-ratio <f>   Clamp runaway output length ratio (default: 1.4)\n"
                      << "  -n, --no-repeat-ngram <i> Suppress 3+ word repetition loops (default: 3, 0=off)\n"
                      << "  -w, --hotwords <csv>     Prioritize brand/creator spellings (e.g. \"RenderCaption,Ansh\")\n"
                      << "  -c, --case <style>       Output formatting: 'none', 'lower', 'title', 'sentence'\n";
            return 0;
        } else if (arg[0] != '-') {
            user_model_path = arg;
        }
    }

    // Direct download requested via CLI
    if (!download_target.empty()) {
        fs::path app_dir = get_executable_directory();
        std::transform(download_target.begin(), download_target.end(), download_target.begin(), ::tolower);
        if (download_target == "onnx" || download_target == "cpu") {
            fs::path onnx_dir = app_dir / "onnx";
            if (download_onnx_model(onnx_dir)) {
                std::cout << "[OK] ONNX models downloaded to: " << onnx_dir.string() << "\n";
                return 0;
            }
            return 1;
        } else {
            fs::path gguf_file = app_dir / "hindi-to-hinglish-pro.q8_0.gguf";
            if (download_gguf_model(gguf_file)) {
                std::cout << "[OK] GGUF model downloaded to: " << gguf_file.string() << "\n";
                return 0;
            }
            return 1;
        }
    }

    bool interactive_gui_mode = (argc == 1 && single_input_text.empty() && batch_input_file.empty());
    if (interactive_gui_mode) {
        std::cout << "\033[1;36m"
                  << "    _    _  ______ _   _    _    ____   \n"
                  << "   / \\  | |/ / ___| | | |  / \\  |  _ \\  \n"
                  << "  / _ \\ | ' /\\___ \\| |_| |/ _ \\ | |_) | \n"
                  << " / ___ \\| . \\ ___) |  _  / ___ \\|  _ <  \n"
                  << "/_/   \\_\\_|\\_\\____/|_| |_/_/   \\_\\_| \\_\\ \n"
                  << "\033[0m"
                  << "========================================================================\n"
                  << "\033[1m  AKSHAR ENGINE (अक्षर) — Hindi to Hinglish Neural Runtime\033[0m\n"
                  << "  Sub-15ms Real-Time Inference • Pure Vulkan GPU & CPU SIMD\n"
                  << "========================================================================\n\n";
    }

    // Determine backend
    std::string selected_backend = dev_arg;
    std::transform(selected_backend.begin(), selected_backend.end(), selected_backend.begin(), ::tolower);

    bool use_onnx = false;
    bool use_vulkan = false;
    bool use_ggml_cpu = false;

    if (selected_backend == "ggml-cpu" || selected_backend == "vulkan-cpu" || selected_backend == "v2-cpu" || selected_backend == "ggml") {
        use_ggml_cpu = true;
    } else if (selected_backend == "onnx" || selected_backend == "cpu" || selected_backend == "ort") {
        use_onnx = true;
    } else if (selected_backend == "vulkan" || selected_backend == "vk" || selected_backend == "gpu") {
        use_vulkan = true;
    } else if (selected_backend == "auto") {
        if (!user_model_path.empty() && fs::is_directory(user_model_path)) {
            use_onnx = true;
        } else {
            std::string candidate_gguf = resolve_gguf_path(user_model_path);
            if (!candidate_gguf.empty()) {
                user_model_path = candidate_gguf;
                use_vulkan = true;
            } else {
                std::string candidate_onnx = resolve_onnx_dir(user_model_path);
                if (!candidate_onnx.empty()) {
                    user_model_path = candidate_onnx;
                    use_onnx = true;
                } else if (interactive_gui_mode) {
                    fs::path app_dir = get_executable_directory();
                    std::cout << "------------------------------------------------------------------------\n"
                              << "  MODEL SETUP: Hindi-to-Hinglish Model Not Found\n"
                              << "------------------------------------------------------------------------\n"
                              << "  Target folder: " << app_dir.string() << "\n\n"
                              << "  [1] Download GGUF model (~303 MB) — Vulkan GPU & CPU SIMD [RECOMMENDED]\n"
                              << "  [2] Download ONNX model suite (~380 MB) — Pure CPU / DirectML\n"
                              << "  [3] Drag & Drop an existing .gguf file or onnx folder into this window\n"
                              << "  [4] Exit\n"
                              << "------------------------------------------------------------------------\n"
                              << "  Select option [Press ENTER for 1] > " << std::flush;

                    std::string choice;
                    std::getline(std::cin, choice);
                    byt5::TextUtils::trim(choice);

                    if (choice.empty() || choice == "1") {
                        fs::path dest_file = app_dir / "hindi-to-hinglish-pro.q8_0.gguf";
                        if (download_gguf_model(dest_file)) {
                            user_model_path = dest_file.string();
                            use_vulkan = true;
                        }
                    } else if (choice == "2") {
                        fs::path dest_dir = app_dir / "onnx";
                        if (download_onnx_model(dest_dir)) {
                            user_model_path = dest_dir.string();
                            use_onnx = true;
                        }
                    } else if (choice == "3") {
                        std::cout << "\n  Drag & drop your .gguf file or onnx folder here and press ENTER > " << std::flush;
                        std::string input_path;
                        if (std::getline(std::cin, input_path)) {
                            byt5::TextUtils::trim(input_path);
                            if (input_path.size() >= 2 && input_path.front() == '"' && input_path.back() == '"') {
                                input_path = input_path.substr(1, input_path.size() - 2);
                            }
                            if (fs::exists(input_path)) {
                                user_model_path = input_path;
                                if (fs::is_directory(input_path)) use_onnx = true;
                                else use_vulkan = true;
                            }
                        }
                    } else {
                        return 0;
                    }
                }

                if (!use_vulkan && !use_onnx) {
                    std::cerr << "\n[Error] No valid model provided or downloaded.\n"
                              << "You can download weights manually from:\n"
                              << "https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro\n\n";

                    if (interactive_gui_mode) {
                        std::cout << "Press Enter to exit...";
                        std::string dummy;
                        std::getline(std::cin, dummy);
                    }
                    return 1;
                }
            }
        }
    } else {
        std::cerr << "[Warning] Unrecognized device '" << dev_arg << "'. Defaulting to auto detection.\n";
        use_vulkan = true;
    }

    std::unique_ptr<IEngine> engine;
    auto t0 = std::chrono::high_resolution_clock::now();

    try {
        if (use_ggml_cpu) {
            byt5::EngineConfig cfg;
            cfg.model_path = resolve_gguf_path(user_model_path);
            cfg.n_predict = n_predict;
            cfg.n_threads = n_threads;
            cfg.n_parallel = (n_parallel > 0) ? n_parallel : 1;
            cfg.device = byt5::DeviceMode::CPU;
            cfg.n_gpu_layers = 0; // Pure CPU AVX2 SIMD

            if (!interactive_gui_mode && !quiet_mode && single_input_text.empty()) {
                std::cerr << "========================================================\n"
                          << " Akshar Engine (अक्षर) [GGML CPU AVX2 Backend]\n"
                          << "========================================================\n"
                          << " Model:   " << cfg.model_path << "\n"
                          << " Threads: " << (cfg.n_threads > 0 ? std::to_string(cfg.n_threads) : "Auto") << "\n"
                          << " Slots:   " << cfg.n_parallel << " CPU streams\n"
                          << " Context: " << cfg.n_ctx << " tokens\n"
                          << "========================================================\n";
            }

            engine = std::make_unique<VulkanEngineAdapter>(cfg);
        }

        if (use_vulkan && !engine) {
            byt5::EngineConfig cfg;
            cfg.model_path = resolve_gguf_path(user_model_path);
            cfg.n_predict = n_predict;
            cfg.n_threads = n_threads;
            cfg.n_parallel = n_parallel;
            cfg.device = byt5::DeviceMode::Vulkan;
            cfg.n_gpu_layers = 99;

            if (!interactive_gui_mode && !quiet_mode && single_input_text.empty()) {
                std::cerr << "========================================================\n"
                          << " Akshar Engine (अक्षर) [Vulkan GPU Backend]\n"
                          << "========================================================\n"
                          << " Model:   " << cfg.model_path << "\n"
                          << " Slots:   " << cfg.n_parallel << " GPU concurrent streams\n"
                          << " Context: " << cfg.n_ctx << " tokens\n"
                          << "========================================================\n";
            }

            try {
                engine = std::make_unique<VulkanEngineAdapter>(cfg);
            } catch (const std::exception& e) {
                if (dev_arg == "auto") {
                    if (!quiet_mode) {
                        std::cerr << "[Auto Fallback] Vulkan initialization failed (" << e.what() 
                                  << "). Automatically falling back to ONNX CPU SIMD backend...\n";
                    }
                    use_onnx = true;
                } else {
                    throw;
                }
            }
        }

        if (use_onnx && !engine) {
            byt5::OnnxConfig cfg;
            cfg.model_dir = resolve_onnx_dir(user_model_path);
            cfg.device = (selected_backend == "dml") ? "dml" : "cpu";
            cfg.n_threads = n_threads;
            cfg.n_predict = n_predict;

            if (!interactive_gui_mode && !quiet_mode && single_input_text.empty()) {
                std::cerr << "========================================================\n"
                          << " Akshar Engine (अक्षर) [ONNX Runtime CPU Backend]\n"
                          << "========================================================\n"
                          << " Model Dir: " << cfg.model_dir << "\n"
                          << " Mode:      " << cfg.device << "\n"
                          << " Threads:   " << (cfg.n_threads > 0 ? std::to_string(cfg.n_threads) : "Auto (Hardware)") << "\n"
                          << "========================================================\n";
            }

            engine = std::make_unique<OnnxEngineAdapter>(cfg);
        }

        if (!engine) {
            std::cerr << "[Fatal Error] Failed to initialize any engine backend.\n";
            return 1;
        }

        engine->set_generation_config(gen_config);

        auto t1 = std::chrono::high_resolution_clock::now();
        double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 1. Single sentence execution (CLI or subprocess pipe)
        if (!single_input_text.empty()) {
            std::string output = engine->generate(single_input_text, n_predict);
            std::cout << output << "\n";
            return 0;
        }

        if (interactive_gui_mode) {
            std::string active_path = user_model_path.empty() ? resolve_gguf_path("") : user_model_path;
            std::string model_fname = fs::path(active_path).filename().string();
            if (model_fname.empty()) model_fname = "hindi-to-hinglish-pro.q8_0.gguf";

            std::cout << "------------------------------------------------------------------------\n"
                      << "  SYSTEM STATUS\n"
                      << "------------------------------------------------------------------------\n"
                      << "  Model    : " << model_fname << "\n"
                      << "  Backend  : " << engine->get_device_name() << "\n"
                      << "  Ready in : " << static_cast<int>(init_ms) << " ms\n"
                      << "------------------------------------------------------------------------\n\n"
                      << "  Type any Hindi sentence and press ENTER (or type 'EXIT' to quit).\n"
                      << "------------------------------------------------------------------------\n";
        } else if (!quiet_mode) {
            std::cerr << "[Akshar Engine] Ready in " << static_cast<int>(init_ms) 
                      << " ms! Active: " << engine->get_device_name() << "\n"
                      << "[Akshar Engine] Listening on STDIO (type 'EXIT' or 'QUIT' to terminate)...\n";
        }

        // Batch processing mode
        if (!batch_input_file.empty()) {
            std::ifstream fin(batch_input_file);
            if (!fin.is_open()) {
                std::cerr << "[Error] Failed to open batch file: " << batch_input_file << std::endl;
                return 1;
            }

            std::vector<std::string> lines;
            std::string l;
            size_t total_in_bytes = 0;
            while (std::getline(fin, l)) {
                byt5::TextUtils::trim(l);
                lines.push_back(l);
                total_in_bytes += l.size();
            }

            std::cerr << "[Akshar Engine] Processing " << lines.size() << " sentences (" 
                      << total_in_bytes << " bytes)...\n";

            auto b0 = std::chrono::high_resolution_clock::now();
            std::vector<std::string> outputs = engine->generate_batch(lines, n_predict);
            auto b1 = std::chrono::high_resolution_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(b1 - b0).count();

            size_t total_out_chars = 0;
            for (const auto& out : outputs) total_out_chars += out.size();

            std::ostream* out_stream = &std::cout;
            std::ofstream fout;
            if (!batch_output_file.empty()) {
                fout.open(batch_output_file);
                if (fout.is_open()) out_stream = &fout;
            }

            for (const auto& out : outputs) {
                *out_stream << out << "\n";
            }

            double chars_per_sec = (batch_ms > 0) ? (total_out_chars / (batch_ms / 1000.0)) : 0;
            double sents_per_sec = (batch_ms > 0) ? (lines.size() / (batch_ms / 1000.0)) : 0;

            std::cerr << "========================================================\n"
                      << " Batch Finished: " << lines.size() << " sentences (" << total_out_chars << " chars)\n"
                      << " Elapsed Time:   " << static_cast<int>(batch_ms) << " ms\n"
                      << " Throughput:     " << static_cast<int>(chars_per_sec) << " chars/sec ("
                      << static_cast<int>(sents_per_sec) << " sentences/sec)\n"
                      << "========================================================\n";
            return 0;
        }

        std::string line;
        while (true) {
            if (interactive_gui_mode) {
                std::cout << "\033[1;36m[Hindi]    > \033[0m" << std::flush;
            }
            if (!std::getline(std::cin, line)) break;
            byt5::TextUtils::trim(line);
            if (line.empty()) continue;
            if (line == "EXIT" || line == "QUIT" || line == "exit" || line == "quit") break;

            auto start = std::chrono::high_resolution_clock::now();
            std::string output = engine->generate(line, n_predict);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            if (interactive_gui_mode) {
                double cps = (ms > 0) ? (output.size() / (ms / 1000.0)) : 0;
                std::cout << "\033[1;32m[Hinglish] > \033[1;37m" << output << "\033[0m\n"
                          << "  \033[90m[⚡ " << std::fixed << std::setprecision(1) << ms 
                          << " ms  •  " << output.size() << " chars  •  " 
                          << static_cast<int>(cps) << " chars/sec]\033[0m\n"
                          << "------------------------------------------------------------------------\n";
            } else {
                if (!quiet_mode) {
                    std::cerr << "[Akshar Engine] '" << line << "' -> '" << output 
                              << "' (" << static_cast<int>(ms) << " ms)\n";
                }
                std::cout << output << std::endl;
            }
        }

        if (interactive_gui_mode) {
            std::cout << "\n[Akshar Engine] Closed. Press Enter to exit...";
            std::string dummy;
            std::getline(std::cin, dummy);
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[Fatal Error] " << e.what() << "\n" << std::endl;
        if (interactive_gui_mode) {
            std::cout << "\nPress Enter to exit...";
            std::string dummy;
            std::getline(std::cin, dummy);
        }
        return 1;
    }

    return 0;
}
