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
};

class OnnxEngineAdapter : public IEngine {
    byt5::OnnxEngine engine_;
public:
    explicit OnnxEngineAdapter(const byt5::OnnxConfig& cfg) : engine_(cfg) {}
    std::string generate(const std::string& text, size_t max_tokens) override {
        return engine_.generate(text, max_tokens);
    }
    std::vector<std::string> generate_batch(const std::vector<std::string>& texts, size_t max_tokens) override {
        return engine_.generate_batch(texts, max_tokens);
    }
    std::string get_device_name() const override {
        return engine_.get_device_name();
    }
};

static std::string resolve_gguf_path(const std::string& user_path) {
    if (!user_path.empty() && fs::exists(user_path) && !fs::is_directory(user_path)) {
        return user_path;
    }
    const char* env_path = std::getenv("BYT5_GGUF_MODEL");
    if (env_path && fs::exists(env_path) && !fs::is_directory(env_path)) {
        return env_path;
    }
    const std::vector<std::string> candidates = {
        "hindi-to-hinglish-pro.q8_0.gguf",
        "models/hindi-to-hinglish-pro.q8_0.gguf",
        "../models/hindi-to-hinglish-pro.q8_0.gguf",
        "../../models/hindi-to-hinglish-pro.q8_0.gguf",
        "../../../models/hindi-to-hinglish-pro.q8_0.gguf",
        "byt5_v3_120k_q8_0.gguf",
        "models/byt5_v3_120k_q8_0.gguf",
        "../models/byt5_v3_120k_q8_0.gguf",
        "../../models/byt5_v3_120k_q8_0.gguf",
        "../../../models/byt5_v3_120k_q8_0.gguf",
        "hinglish_research/byt5_v3_120k_q8_0.gguf",
        "../hinglish_research/byt5_v3_120k_q8_0.gguf",
        "../../hinglish_research/byt5_v3_120k_q8_0.gguf",
        "../../../hinglish_research/byt5_v3_120k_q8_0.gguf",
        "../../../../hinglish_research/byt5_v3_120k_q8_0.gguf",
        "hinglish_research/byt5_v3_120k.gguf",
        "../hinglish_research/byt5_v3_120k.gguf",
        "../../hinglish_research/byt5_v3_120k.gguf",
        "../../../hinglish_research/byt5_v3_120k.gguf",
        "../../../../hinglish_research/byt5_v3_120k.gguf"
    };
    for (const auto& c : candidates) {
        if (!c.empty() && fs::exists(c) && !fs::is_directory(c)) {
            return c;
        }
    }
    return user_path;
}

static std::string resolve_onnx_dir(const std::string& user_path) {
    if (!user_path.empty() && fs::exists(user_path) && fs::is_directory(user_path)) {
        return user_path;
    }
    const char* env_dir = std::getenv("BYT5_ONNX_DIR");
    if (env_dir && fs::exists(env_dir) && fs::is_directory(env_dir)) {
        return env_dir;
    }
    const std::vector<std::string> candidates = {
        "onnx",
        "models/onnx",
        "../models/onnx",
        "../../models/onnx",
        "byt5_onnx_q8",
        "models/byt5_onnx_q8",
        "../models/byt5_onnx_q8",
        "../../models/byt5_onnx_q8",
        "../../../models/byt5_onnx_q8",
        "hinglish_research/byt5_onnx_q8",
        "../hinglish_research/byt5_onnx_q8",
        "../../hinglish_research/byt5_onnx_q8",
        "../../../hinglish_research/byt5_onnx_q8",
        "../../../../hinglish_research/byt5_onnx_q8"
    };
    for (const auto& c : candidates) {
        if (!c.empty() && fs::exists(c) && fs::is_directory(c)) {
            return c;
        }
    }
    return user_path;
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    std::setlocale(LC_ALL, ".UTF-8");
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif

    std::string user_model_path;
    std::string dev_arg = "auto";
    int n_threads = 0;
    int n_predict = 256;
    int n_parallel = 4;
    std::string batch_input_file;
    std::string batch_output_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            user_model_path = argv[++i];
        } else if ((arg == "--device" || arg == "-d" || arg == "--backend") && i + 1 < argc) {
            dev_arg = argv[++i];
        } else if ((arg == "--max-tokens" || arg == "-t") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if ((arg == "--threads" || arg == "-j") && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if ((arg == "--parallel" || arg == "-p") && i + 1 < argc) {
            n_parallel = std::stoi(argv[++i]);
        } else if ((arg == "--batch-file" || arg == "-b") && i + 1 < argc) {
            batch_input_file = argv[++i];
        } else if ((arg == "--output-file" || arg == "-o") && i + 1 < argc) {
            batch_output_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\033[1;36m"
                      << "================================================================================\n"
                      << "           HINDI TO HINGLISH NEURAL TRANSLATION RUNTIME\n"
                      << "           Akshar Engine (अक्षर) — ByT5 Vulkan GPU & AVX2 SIMD\n"
                      << "================================================================================\033[0m\n\n"
                      << "\033[1;33mUsage:\033[0m akshar_engine [options] [model_path]\n\n"
                      << "Options:\n"
                      << "  -d, --device <backend>   Target backend: 'auto', 'vulkan' (GPU), 'onnx' (CPU), 'dml' (DirectML)\n"
                      << "  -m, --model <path>       Model path (.gguf file for Vulkan, directory for ONNX)\n"
                      << "  -j, --threads <int>      CPU compute threads (default: hardware concurrency)\n"
                      << "  -p, --parallel <int>     GPU parallel compute slots for Vulkan (default: 4)\n"
                      << "  -t, --max-tokens <int>   Maximum generation tokens (default: 256)\n"
                      << "  -b, --batch-file <file>  Batch process lines from a file\n"
                      << "  -o, --output-file <file> Output file destination for batch results (default: stdout)\n"
                      << "  -h, --help               Show this help message\n";
            return 0;
        } else if (arg[0] != '-') {
            user_model_path = arg;
        }
    }

    bool interactive_gui_mode = (argc == 1);
    if (interactive_gui_mode) {
        std::cout << "\n\033[1;36m"
                  << " ╔═══════════════════════════════════════════════════════════════════════════╗\n"
                  << " ║\033[1;37m                 HINDI TO HINGLISH NEURAL TRANSLATION                      \033[1;36m║\n"
                  << " ║\033[1;33m              Akshar Engine (अक्षर) • Ultra-Fast ByT5 Runtime              \033[1;36m║\n"
                  << " ║                                                                           ║\n"
                  << " ║\033[0m   ⚡ \033[1;32mPure Vulkan GPU & AVX2 SIMD\033[0m      ⏱️  \033[1;32mSub-15ms Real-Time Latency\033[0m     \033[1;36m║\n"
                  << " ║\033[0m   📦 \033[90mModel: Singla0009/Hindi-to-Hinglish-Pro\033[0m • \033[33mCC-BY-NC 4.0 Non-Commercial\033[0m \033[1;36m║\n"
                  << " ╚═══════════════════════════════════════════════════════════════════════════╝\033[0m\n\n";
    }

    // Determine backend
    std::string selected_backend = dev_arg;
    std::transform(selected_backend.begin(), selected_backend.end(), selected_backend.begin(), ::tolower);

    bool use_onnx = false;
    bool use_vulkan = false;
    bool use_ggml_cpu = false;

    if (selected_backend == "ggml-cpu" || selected_backend == "vulkan-cpu" || selected_backend == "v2-cpu" || selected_backend == "ggml") {
        use_ggml_cpu = true;
    } else if (selected_backend == "onnx" || selected_backend == "cpu" || selected_backend == "ort" || selected_backend == "dml") {
        use_onnx = true;
    } else if (selected_backend == "vulkan" || selected_backend == "vk" || selected_backend == "gpu") {
        use_vulkan = true;
    } else if (selected_backend == "auto") {
        // Auto: If user explicitly passed an ONNX directory or if .gguf doesn't exist, try ONNX; otherwise try Vulkan
        if (!user_model_path.empty() && fs::is_directory(user_model_path)) {
            use_onnx = true;
        } else {
            std::string candidate_gguf = resolve_gguf_path(user_model_path);
            if (fs::exists(candidate_gguf) && !fs::is_directory(candidate_gguf)) {
                use_vulkan = true;
            } else {
                std::string candidate_onnx = resolve_onnx_dir(user_model_path);
                if (fs::exists(candidate_onnx) && fs::is_directory(candidate_onnx)) {
                    use_onnx = true;
                } else if (interactive_gui_mode) {
                    // Friendly drag & drop model prompt for double-clickers
                    std::cout << "\033[1;33m ╭─ [ Model Setup ] ─────────────────────────────────────────────────────────╮\n"
                              << " │  📂 Drag & Drop your 'hindi-to-hinglish-pro.q8_0.gguf' here & press ENTER  │\n"
                              << " │  🌐 Free Weights: huggingface.co/Singla0009/Hindi-to-Hinglish-Pro         │\n"
                              << " ╰───────────────────────────────────────────────────────────────────────────╯\033[0m\n"
                              << " \033[1;36mModel File Path ❯\033[0m " << std::flush;
                    std::string input_path;
                    if (std::getline(std::cin, input_path)) {
                        byt5::TextUtils::trim(input_path);
                        if (!input_path.empty()) {
                            if (input_path.size() >= 2 && input_path.front() == '"' && input_path.back() == '"') {
                                input_path = input_path.substr(1, input_path.size() - 2);
                            }
                            if (fs::exists(input_path)) {
                                user_model_path = input_path;
                                if (fs::is_directory(input_path)) use_onnx = true;
                                else use_vulkan = true;
                            }
                        }
                    }
                }

                // If still not found after interactive prompt
                if (!use_vulkan && !use_onnx) {
                    std::cout << "\n\033[1;31m❌ Hindi to Hinglish model not found!\033[0m\n\n"
                              << "📥 \033[1;37mDownload official weights from Hugging Face:\033[0m\n"
                              << "   \033[1;36mhuggingface-cli download Singla0009/Hindi-to-Hinglish-Pro gguf/hindi-to-hinglish-pro.q8_0.gguf --local-dir models/\033[0m\n\n"
                              << "   Or launch via terminal with explicit model path:\n"
                              << "   \033[90makshar_engine.exe -m <path_to_model.gguf>\033[0m\n\n";

                    if (interactive_gui_mode) {
                        std::cout << "\033[90mPress Enter to exit...\033[0m";
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

            if (!interactive_gui_mode) {
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

            if (!interactive_gui_mode) {
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
                    std::cerr << "[Auto Fallback] Vulkan initialization failed (" << e.what() 
                              << "). Automatically falling back to ONNX CPU SIMD backend...\n";
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

            if (!interactive_gui_mode) {
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

        auto t1 = std::chrono::high_resolution_clock::now();
        double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (interactive_gui_mode) {
            std::string active_path = user_model_path.empty() ? resolve_gguf_path("") : user_model_path;
            std::string model_fname = fs::path(active_path).filename().string();
            if (model_fname.empty()) model_fname = "hindi-to-hinglish-pro.q8_0.gguf";

            std::cout << "\033[1;32m ┌─ System Status ───────────────────────────────────────────────────────────┐\n"
                      << " │  ✔ Neural Model   : " << model_fname << "\n"
                      << " │  ✔ Hardware Target: " << engine->get_device_name() << "\n"
                      << " │  ✔ Initialization : Ready in " << static_cast<int>(init_ms) << " ms\n"
                      << " └───────────────────────────────────────────────────────────────────────────┘\033[0m\n\n"
                      << "\033[1;37m 💡 Enter any Hindi text below and press [ENTER] to translate to Hinglish.\033[0m\n"
                      << "    Type \033[1;31mEXIT\033[0m or \033[1;31mQUIT\033[0m to terminate the engine.\n"
                      << "\033[90m ───────────────────────────────────────────────────────────────────────────\033[0m\n";
        } else {
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
                std::cout << "\n\033[1;36m[ हिन्दी In ]  ❯ \033[0m" << std::flush;
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
                std::cout << "\033[1;32m[ Hinglish Out] ❯ \033[1;37m" << output << "\033[0m\n"
                          << "  \033[90m⚡ " << std::fixed << std::setprecision(1) << ms 
                          << " ms  •  " << output.size() << " chars  •  " 
                          << static_cast<int>(cps) << " chars/sec\033[0m\n";
            } else {
                std::cerr << "[Akshar Engine] '" << line << "' -> '" << output 
                          << "' (" << static_cast<int>(ms) << " ms)\n";
                std::cout << output << std::endl;
            }
        }

        if (interactive_gui_mode) {
            std::cout << "\n\033[90mPress Enter to close this window...\033[0m";
            std::string dummy;
            std::getline(std::cin, dummy);
        }

    } catch (const std::exception& e) {
        std::cerr << "\n\033[1;31m[Fatal Error] " << e.what() << "\033[0m\n" << std::endl;
        if (interactive_gui_mode) {
            std::cout << "\n\033[90mPress Enter to close this window...\033[0m";
            std::string dummy;
            std::getline(std::cin, dummy);
        }
        return 1;
    }

    return 0;
}
