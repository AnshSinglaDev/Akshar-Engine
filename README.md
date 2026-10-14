<div align="center">

# 🏛️ Akshar Engine (अक्षर)
### Real-Time Hindi to Hinglish AI Translator — 10ms Latency, Zero Setup, Works on Any PC

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Vulkan](https://img.shields.io/badge/Vulkan-Any_GPU-C41E3A?style=for-the-badge&logo=vulkan&logoColor=white)](https://www.vulkan.org/)
[![CPU](https://img.shields.io/badge/CPU-Any_Laptop-FF6F00?style=for-the-badge&logo=intel&logoColor=white)](https://onnxruntime.ai/)
[![Hugging Face](https://img.shields.io/badge/Model-Hindi--to--Hinglish--Pro-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-9370DB?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Akshar Engine (अक्षर)</b> converts Hindi (Devanagari script) into natural, everyday Hinglish (Roman English script).<br>
  It handles conversational words, slangs, and English loanwords like <i>Doctor, Hospital, Computer, Mobile</i> with 100% accurate spelling.<br>
  Runs <b>100% offline</b> on your own machine in <b>~10 to 15 milliseconds</b> per sentence — no internet or cloud API required.
</p>

**TL;DR — Type Hindi → get natural Hinglish in 10 milliseconds. Works on NVIDIA, AMD, Intel graphics cards or regular laptop CPUs. Free forever for non-commercial use.**

</div>

---

## ⚡ Quick Start (No Coding Required)

### Step 1: Download the Pre-Built App
Download the Windows app bundle (**19 MB**) from the **[Releases Page](https://github.com/AnshSinglaDev/Akshar-Engine/releases/latest)**:
👉 **[Download akshar-engine-v1.0.0-win-x64.zip](https://github.com/AnshSinglaDev/Akshar-Engine/releases/download/v1.0.0/akshar-engine-v1.0.0-win-x64.zip)**

### Step 2: Download the AI Model
Download the official AI model file (**303 MB**) from Hugging Face:
👉 **[Download hindi-to-hinglish-pro.q8_0.gguf](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/resolve/main/gguf/hindi-to-hinglish-pro.q8_0.gguf)**

### Step 3: Run
1. Unzip the downloaded folder.
2. Place the `hindi-to-hinglish-pro.q8_0.gguf` file inside that folder.
3. **Double-click `akshar_engine.exe`**.

That's it! Type any Hindi sentence and press **Enter** to see the instant Hinglish translation.

*(Note: If you run it without placing the model first, the app will politely ask you to drag & drop the model file right into the window.)*

---

## 🎬 Live Terminal Preview

```text
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                 HINDI TO HINGLISH NEURAL TRANSLATION                      ║
 ║              Akshar Engine (अक्षर) • Ultra-Fast ByT5 Runtime              ║
 ║                                                                           ║
 ║   ⚡ Pure Vulkan GPU & AVX2 SIMD      ⏱️  Sub-15ms Real-Time Latency     ║
 ║   📦 Model: Singla0009/Hindi-to-Hinglish-Pro • CC-BY-NC 4.0 Non-Commercial ║
 ╚═══════════════════════════════════════════════════════════════════════════╝

 ┌─ System Status ───────────────────────────────────────────────────────────┐
 │  ✔ Neural Model   : hindi-to-hinglish-pro.q8_0.gguf
 │  ✔ Hardware Target: Vulkan GPU (Matrix Cores, 4 parallel streams)
 │  ✔ Initialization : Ready in 358 ms
 └───────────────────────────────────────────────────────────────────────────┘

 💡 Enter any Hindi text below and press [ENTER] to translate to Hinglish.
    Type EXIT or QUIT to terminate the engine.
 ───────────────────────────────────────────────────────────────────────────

[ हिन्दी In ]   ❯ आज रात का मौसम बहुत सुहाना है और मुझे चाय पीने का मन कर रहा है।
[ Hinglish Out ] ❯ Aaj raat ka mausam bahut suhana hai aur mujhe chai peene ka man kar raha hai.
  ⚡ 14.8 ms  •  77 chars  •  5,202 chars/sec

[ हिन्दी In ]   ❯ डॉक्टर ने मरीज को तुरंत अस्पताल ले जाने और ब्लड टेस्ट कराने की सलाह दी।
[ Hinglish Out ] ❯ Doctor ne maries ko turant hospital le jane aur blood test karane ki salah di.
  ⚡ 17.2 ms  •  83 chars  •  4,825 chars/sec
```

---

## 🌟 Why Akshar Engine? (In Plain Words)

| Traditional Translators / Old Tools | Akshar Engine (अक्षर) |
| :--- | :--- |
| **Butchers English words:** Words like "डॉक्टर" or "हॉस्पिटल" get butchered into *"daktar"* or *"aspataal"*. | **Keeps correct English:** Automatically outputs `Doctor`, `Hospital`, `Computer`, `Investment`, and `Police`. |
| **Slow Python scripts:** Takes 300 to 500 ms per line and needs a huge 15 GB CUDA setup. | **Instant C++ speed:** Takes **10–15 milliseconds** per sentence and uses under 870 MB of memory. |
| **Requires NVIDIA only:** Locked to expensive CUDA GPUs. | **Works on any graphics card:** NVIDIA, AMD Radeon, Intel Arc, or regular laptop CPUs. |
| **Requires Internet / Cloud:** Sends your private text and data to remote servers. | **100% Offline & Private:** Everything runs locally on your PC. No data ever leaves your computer. |

---

## 🤖 Available AI Models (Hugging Face)

All official model weights are hosted on Hugging Face at [**`Singla0009/Hindi-to-Hinglish-Pro`**](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro):

| Model Format | Best For | File to Download | Size | Link |
| :--- | :--- | :--- | :---: | :--- |
| **GGUF (Q8_0)** ⭐ | **Most Users (GPU & Laptop)** | `hindi-to-hinglish-pro.q8_0.gguf` | 303 MB | [Direct Download](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/resolve/main/gguf/hindi-to-hinglish-pro.q8_0.gguf) |
| **ONNX (INT8)** | Budget Cloud VPS / CPU servers | `onnx/` folder | ~380 MB | [Browse ONNX Files](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/tree/main/onnx) |
| **CTranslate2** | Offline C++ batch jobs | `ct2/` folder | ~360 MB | [Browse CT2 Files](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/tree/main/ct2) |
| **PyTorch SafeTensors**| Python developers & fine-tuning | `model.safetensors` | 1.2 GB | [Browse PyTorch Files](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro/tree/main) |

### Download via Hugging Face CLI (Optional):
```bash
# Recommended for most users:
huggingface-cli download Singla0009/Hindi-to-Hinglish-Pro gguf/hindi-to-hinglish-pro.q8_0.gguf --local-dir models/
```

---

## 📊 Real-World Speed Test

Measured on a standard desktop computer:

| Sentence Type | Hindi Input | Hinglish Output | Speed |
| :--- | :--- | :--- | :---: |
| **Short Greeting** | *नमस्ते दुनिया* | *Namaste dunia* | **10 ms** |
| **Daily Conversation** | *आप आज शाम को क्या कर रहे हैं?* | *Aap aaj shaam ko kya kar rahe hain?* | **28 ms** |
| **Technical / Business** | *आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग बहुत जरूरी हैं।* | *Artificial Intelligence aur machine learning bahut zaroori hain.* | **72 ms** |
| **Medical / Health** | *डॉक्टर ने मरीज को तुरंत अस्पताल ले जाने की सलाह दी।* | *Doctor ne maries ko turant hospital le jane ki salah di.* | **94 ms** |
| **Full Audio Paragraph** | *50 continuous sentences (~15 minutes of speech)* | *5,830 characters transliterated* | **6.1 seconds** (948 chars/sec) |

---

## 💻 Developer & Command Line Usage

If you prefer using the Command Prompt or PowerShell:

### 1. Interactive Mode
```bash
akshar_engine.exe -m models/hindi-to-hinglish-pro.q8_0.gguf
```

### 2. Process a Text File (Batch)
Translate thousands of sentences at once from a file:
```bash
akshar_engine.exe -b input_hindi.txt -o output_hinglish.txt
```

### 3. Force Pure CPU Mode
If running on a server without a graphics card:
```bash
akshar_engine.exe -d onnx -j 8 -m models/onnx
```

---

## 🛠️ Build from Source (For Developers)

If you want to compile the C++ code yourself:

### Requirements
- **CMake** 3.20 or higher
- **C++17 Compiler** (MSVC on Windows, GCC/Clang on Linux)
- **Vulkan SDK** (Optional, for GPU mode)

### Compilation Steps
```bash
# Clone the repository
git clone https://github.com/AnshSinglaDev/Akshar-Engine.git
cd Akshar-Engine

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
The compiled executable will be placed in `./build/Release/akshar_engine.exe`.

---

## 🌐 Works With RenderCaption

**Akshar Engine** is designed to power the real-time Hindi-to-Hinglish subtitle feature inside [**RenderCaption**](https://github.com/AnshSinglaDev/Rust_Indic). You can combine it with any Speech-to-Text model for an end-to-end live subtitling workflow.

---

## 📄 License & Terms

* **License:** [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](LICENSE)
* **What you CAN do:** Free to use for personal projects, research, educational study, content creation, and open-source hobby workflows.
* **What you CANNOT do:** You may **not** put this engine behind a paid paywall, sell it as a paid product, or charge users a subscription without explicit written permission.
* **Creator:** [Ansh Singla (AnshSinglaDev)](https://github.com/AnshSinglaDev)
* **Neural Model:** [Hindi-to-Hinglish-Pro on Hugging Face](https://huggingface.co/Singla0009/Hindi-to-Hinglish-Pro)
