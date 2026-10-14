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

static bool download_gguf_model(const fs::path& target_file) {
    const std::string url = "https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/resolve/main/gguf/hindi-to-hinglish-pro.q8_0.gguf";

    std::cout << "\n------------------------------------------------------------------------\n"
              << "  DOWNLOADING OFFICIAL AI MODEL FROM HUGGING FACE\n"
              << "------------------------------------------------------------------------\n"
              << "  Model  : hindi-to-hinglish-pro.q8_0.gguf (~303 MB)\n"
              << "  Source : " << url << "\n"
              << "  Save to: " << target_file.string() << "\n"
              << "------------------------------------------------------------------------\n\n";

    // Attempt download using native Windows curl.exe with live progress bar
    std::string cmd = "curl.exe -L --progress-bar -o \"" + target_file.string() + "\" \"" + url + "\"";
    int ret = std::system(cmd.c_str());

    if (ret == 0 && fs::exists(target_file) && fs::file_size(target_file) > 100 * 1024 * 1024) {
        std::cout << "\n[OK] Model successfully downloaded (" << (fs::file_size(target_file) / (1024 * 1024)) << " MB)!\n"
                  << "Saved to: " << target_file.string() << "\n\n";
        return true;
    }

    // Fallback via PowerShell Invoke-WebRequest if curl.exe is missing
    std::cout << "\n[Notice] Standard downloader failed. Retrying via PowerShell WebClient...\n";
    std::string ps_cmd = "powershell -Command \"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('" + url + "', '" + target_file.string() + "')\"";
    ret = std::system(ps_cmd.c_str());

    if (ret == 0 && fs::exists(target_file) && fs::file_size(target_file) > 100 * 1024 * 1024) {
        std::cout << "\n[OK] Fallback download completed successfully (" << (fs::file_size(target_file) / (1024 * 1024)) << " MB)!\n\n";
        return true;
    }

    std::cerr << "\n[Error] Download failed or file was incomplete. Please check your internet connection.\n";
    if (fs::exists(target_file)) {
        std::error_code ec;
        fs::remove(target_file, ec);
    }
    return false;
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
            std::cout << "\n========================================================================\n"
                      << "  AKSHAR ENGINE (अक्षर) — HINDI TO HINGLISH NEURAL RUNTIME\n"
                      << "  Sub-15ms Real-Time Inference • Pure Vulkan GPU & CPU SIMD\n"
                      << "========================================================================\n\n"
                      << "Usage: akshar_engine [options] [model_path]\n\n"
                      << "Options:\n"
                      << "  -d, --device <backend>   Target backend: 'auto', 'vulkan' (GPU), 'onnx' (CPU)\n"
                      << "  -m, --model <path>       Model path (.gguf file for Vulkan, directory for ONNX)\n"
                      << "  -j, --threads <int>      CPU compute threads (default: hardware concurrency)\n"
                      << "  -p, --parallel <int>     GPU parallel compute slots for Vulkan (default: 4)\n"
                      << "  -t, --max-tokens <int>   Maximum generation tokens (default: 256)\n"
                      << "  -b, --batch-file <file>  Batch process lines from a file\n"
                      << "  -o, --output-file <file> Output destination for batch results (default: stdout)\n"
                      << "  -h, --help               Show this help message\n";
            return 0;
        } else if (arg[0] != '-') {
            user_model_path = arg;
        }
    }

    bool interactive_gui_mode = (argc == 1);
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
                              << "  [1] Download official model from Hugging Face (~303 MB) [RECOMMENDED]\n"
                              << "  [2] Drag & Drop an existing .gguf file into this window\n"
                              << "  [3] Exit\n"
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
                        std::cout << "\n  Drag & drop your .gguf file here and press ENTER > " << std::flush;
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

            std::cout << "------------------------------------------------------------------------\n"
                      << "  SYSTEM STATUS\n"
                      << "------------------------------------------------------------------------\n"
                      << "  Model    : " << model_fname << "\n"
                      << "  Backend  : " << engine->get_device_name() << "\n"
                      << "  Ready in : " << static_cast<int>(init_ms) << " ms\n"
                      << "------------------------------------------------------------------------\n\n"
                      << "  Type any Hindi sentence and press ENTER (or type 'EXIT' to quit).\n"
                      << "------------------------------------------------------------------------\n";
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
                std::cerr << "[Akshar Engine] '" << line << "' -> '" << output 
                          << "' (" << static_cast<int>(ms) << " ms)\n";
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
