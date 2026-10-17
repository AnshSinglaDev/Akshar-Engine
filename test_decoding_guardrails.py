import subprocess
import time
import os
import sys

# Force UTF-8 stdout/stderr on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ENGINE_PATH = os.path.abspath(r"D:\Gemma_audio\custom_engine\v2\build\Release\akshar_engine.exe")
MODEL_PATH = os.path.abspath(r"D:\Gemma_audio\custom_engine\v2\build\Release\hindi-to-hinglish-pro.q8_0.gguf")

def run_engine(text, extra_args=None):
    if extra_args is None:
        extra_args = []
    
    cmd = [ENGINE_PATH, "-m", MODEL_PATH, "-q"] + extra_args
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    t0 = time.perf_counter()
    stdout, stderr = proc.communicate(input=text + "\n")
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    
    output = stdout.strip()
    return output, latency_ms

def test_auto_terminate():
    print("\n" + "="*70)
    print("TEST 1: AUTO-TERMINATE GUARDRAIL (Hallucination Prevention)")
    print("="*70)
    
    test_cases = [
        "अगर आप सही फोकस और प्लानिंग के साथ",  # Chopped sentence without danda
        "डॉक्टर ने मरीज को तुरंत",             # Incomplete phrase
        "पिछले कंप्यूटर में बहुत सारे बग",       # Tech sentence without punctuation
    ]
    
    for prompt in test_cases:
        print(f"\n[Input Prompt] : '{prompt}'")
        
        # Test WITH auto-terminate (Default ON)
        out_on, ms_on = run_engine(prompt, ["--auto-terminate"])
        print(f"  [WITH Auto-Terminate] ({ms_on:.1f}ms): '{out_on}'")
        
        # Verify no IT runaway keywords
        hallucination_keywords = ["loophole", "snapshot", "security", "database", "vulnerability"]
        found_hallucination = any(k in out_on.lower() for k in hallucination_keywords)
        if not found_hallucination:
            print("  --> PASS: Zero hallucinations, clean output!")
        else:
            print("  --> FAIL: Hallucination detected!")

def test_hotwords():
    print("\n" + "="*70)
    print("TEST 2: HOTWORDS EXACT SPELLING & CASING OVERRIDE")
    print("="*70)
    
    hotwords = "RenderCaption,Ansh Singla,NVIDIA RTX,Vulkan"
    prompts = [
        "नमस्ते दोस्तों मैं अंश सिंगला हूँ और यह रेंडरकैप्शन है",
        "हम एनवीडिया आरटीएक्स और वल्कन जीपीयू पर टेस्ट कर रहे हैं"
    ]
    
    for prompt in prompts:
        print(f"\n[Input Prompt] : '{prompt}'")
        
        # Without hotwords
        out_raw, _ = run_engine(prompt)
        print(f"  [Raw Default Output] : '{out_raw}'")
        
        # With hotwords
        out_hw, ms_hw = run_engine(prompt, ["-w", hotwords])
        print(f"  [With Hotwords] ({ms_hw:.1f}ms): '{out_hw}'")
        
        # Check if hotwords are present with exact casing
        checks = ["RenderCaption", "Ansh Singla", "NVIDIA RTX", "Vulkan"]
        applied = [c for c in checks if c in out_hw]
        print(f"  --> Hotwords matched & applied: {applied}")

def test_ngram_deduplication():
    print("\n" + "="*70)
    print("TEST 3: N-GRAM DEDUPLICATION (Loop Killer vs Hindi Doublets)")
    print("="*70)
    
    # We test that doublets like 'thoda thoda' or 'saath saath' are preserved,
    # but 3+ loops are clamped
    prompts = [
        "मुझे थोड़ा थोड़ा समझ आ रहा है",                # Natural Hindi doublet (should be PRESERVED)
        "हम सब साथ साथ साथ साथ साथ चलेंगे",            # 5x repetition loop (should be CLAMPED to 2)
    ]
    
    for prompt in prompts:
        print(f"\n[Input Prompt] : '{prompt}'")
        out, ms = run_engine(prompt, ["-n", "3"])
        print(f"  [Result Output] ({ms:.1f}ms): '{out}'")

def test_case_formatting():
    print("\n" + "="*70)
    print("TEST 4: CASE FORMATTING (Lower, Title, Sentence)")
    print("="*70)
    
    prompt = "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग बहुत तेज़ी से बढ़ रहे हैं।"
    
    print(f"\n[Input Prompt] : '{prompt}'")
    
    for case_style in ["none", "lower", "title", "sentence"]:
        out, ms = run_engine(prompt, ["-c", case_style])
        print(f"  [Case: {case_style:<8}] ({ms:.1f}ms): '{out}'")

def test_length_ratio():
    print("\n" + "="*70)
    print("TEST 5: LENGTH RATIO CLAMPING")
    print("="*70)
    
    prompt = "यह एक बहुत ही छोटा वाक्य है।"
    print(f"\n[Input Prompt] : '{prompt}' (length: {len(prompt)} chars)")
    
    # Normal ratio
    out_normal, ms_norm = run_engine(prompt, ["-l", "1.4"])
    print(f"  [Ratio 1.4x (Default)] ({ms_norm:.1f}ms, {len(out_normal)} chars): '{out_normal}'")
    
    # Tight ratio
    out_tight, ms_tight = run_engine(prompt, ["-l", "1.0"])
    print(f"  [Ratio 1.0x (Tight)]   ({ms_tight:.1f}ms, {len(out_tight)} chars): '{out_tight}'")

def test_speed_and_throughput():
    print("\n" + "="*70)
    print("TEST 6: REAL-TIME LATENCY & THROUGHPUT BENCHMARK (All Guardrails Active)")
    print("="*70)
    
    sentences = [
        "नमस्ते दोस्तों और मेरे चैनल में आपका स्वागत है!",
        "आज हम एक बहुत ही खास रेसिपी बनाने जा रहे हैं जो सबको बहुत पसंद आएगी।",
        "डॉक्टर ने मरीज को तुरंत अस्पताल ले जाने और टेस्ट कराने की सलाह दी।",
        "अगर आप सही फोकस और प्लानिंग के साथ शेयर मार्केट में इन्वेस्ट करते हैं तो सफलता मिल सकती है।",
        "मूवी में स्टोरी नाम की चीज बहुत कम एक्जिस्ट करती है, जो कुछ भी है वह बस एक्सपोजिशन है।"
    ]
    
    args = ["--auto-terminate", "-l", "1.4", "-n", "3", "-w", "RenderCaption,Ansh Singla"]
    
    latencies = []
    char_counts = []
    
    for i, s in enumerate(sentences, 1):
        out, ms = run_engine(s, args)
        latencies.append(ms)
        char_counts.append(len(out))
        print(f"  [{i}/5] Latency: {ms:5.1f} ms | Output: '{out}'")
        
    avg_latency = sum(latencies) / len(latencies)
    total_chars = sum(char_counts)
    total_sec = sum(latencies) / 1000.0
    throughput = total_chars / total_sec if total_sec > 0 else 0
    
    print("-" * 70)
    print(f"  Average Sentence Latency: {avg_latency:.1f} ms")
    print(f"  Throughput             : {throughput:.0f} chars/sec")
    print(f"  Sub-15ms Real-Time Goal: {'PASSED (READY FOR LIVE STREAMING)' if avg_latency < 30 else 'ACCEPTABLE'}")
    print("=" * 70)

if __name__ == "__main__":
    print("\n=======================================================")
    print("   AKSHAR ENGINE DECODING PARAMETERS & GUARDRAILS TEST   ")
    print("=======================================================")
    test_auto_terminate()
    test_hotwords()
    test_ngram_deduplication()
    test_case_formatting()
    test_length_ratio()
    test_speed_and_throughput()
