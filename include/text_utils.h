#pragma once
#include <string>
#include <vector>

namespace byt5 {

class TextUtils {
public:
    // Strips UTF-8 BOM, trailing whitespace, and CR/LF characters
    static inline void trim(std::string& line) {
        if (line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line = line.substr(3);
        }
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
    }

    // Filters and sanitizes UTF-8 byte streams, stripping invalid byte fragments
    static inline std::string sanitize_utf8(const std::string& str) {
        std::string out;
        out.reserve(str.size());
        for (size_t i = 0; i < str.size(); ) {
            unsigned char c = static_cast<unsigned char>(str[i]);
            if (c < 0x80) {
                out.push_back(str[i++]);
            } else if ((c & 0xE0) == 0xC0) {
                if (i + 1 < str.size() && (static_cast<unsigned char>(str[i + 1]) & 0xC0) == 0x80) {
                    out.append(str, i, 2);
                    i += 2;
                } else {
                    i++;
                }
            } else if ((c & 0xF0) == 0xE0) {
                if (i + 2 < str.size() &&
                    (static_cast<unsigned char>(str[i + 1]) & 0xC0) == 0x80 &&
                    (static_cast<unsigned char>(str[i + 2]) & 0xC0) == 0x80) {
                    out.append(str, i, 3);
                    i += 3;
                } else {
                    i++;
                }
            } else if ((c & 0xF8) == 0xF0) {
                if (i + 3 < str.size() &&
                    (static_cast<unsigned char>(str[i + 1]) & 0xC0) == 0x80 &&
                    (static_cast<unsigned char>(str[i + 2]) & 0xC0) == 0x80 &&
                    (static_cast<unsigned char>(str[i + 3]) & 0xC0) == 0x80) {
                    out.append(str, i, 4);
                    i += 4;
                } else {
                    i++;
                }
            } else {
                i++;
            }
        }
        return out;
    }

    // Splits Devanagari text on punctuation boundaries ('।', '?', '!', '\n')
    static inline std::vector<std::string> split_sentences(const std::string& text) {
        std::vector<std::string> chunks;
        std::string current;
        current.reserve(text.size());

        size_t i = 0;
        while (i < text.size()) {
            // Check for UTF-8 Devanagari danda '।' (0xE0 0xA5 0xA4) or double danda '॥' (0xE0 0xA5 0xA5)
            if (i + 2 < text.size() &&
                static_cast<unsigned char>(text[i]) == 0xE0 &&
                static_cast<unsigned char>(text[i + 1]) == 0xA5 &&
                (static_cast<unsigned char>(text[i + 2]) == 0xA4 || static_cast<unsigned char>(text[i + 2]) == 0xA5)) {
                current.append(text, i, 3);
                i += 3;
                while (i < text.size() && text[i] == ' ') i++;
                if (!current.empty()) {
                    chunks.push_back(current);
                    current.clear();
                }
            } else if (text[i] == '.' || text[i] == '?' || text[i] == '!' || text[i] == '\n' || text[i] == ';') {
                current.push_back(text[i]);
                i++;
                while (i < text.size() && text[i] == ' ') i++;
                if (!current.empty()) {
                    chunks.push_back(current);
                    current.clear();
                }
            } else {
                current.push_back(text[i]);
                i++;
            }
        }
        if (!current.empty()) {
            chunks.push_back(current);
        }
        return chunks;
    }

    // Smart chunker: Packs sentences up to max_bytes (default: 750 bytes).
    // If an individual sentence has no punctuation and exceeds max_bytes,
    // splits cleanly on space boundaries so no UTF-8 character is sliced.
    static inline std::vector<std::string> chunk_text(const std::string& text, size_t max_bytes = 750) {
        if (text.size() <= max_bytes) {
            return {text};
        }
        std::vector<std::string> sentences = split_sentences(text);
        std::vector<std::string> result;
        std::string current_chunk;
        current_chunk.reserve(max_bytes + 64);

        for (const auto& s : sentences) {
            if (s.size() > max_bytes) {
                if (!current_chunk.empty()) {
                    result.push_back(current_chunk);
                    current_chunk.clear();
                }
                size_t start = 0;
                while (start < s.size()) {
                    size_t len = std::min(max_bytes, s.size() - start);
                    if (start + len < s.size()) {
                        size_t last_space = s.rfind(' ', start + len);
                        if (last_space != std::string::npos && last_space > start) {
                            len = last_space - start;
                        }
                    }
                    std::string piece = s.substr(start, len);
                    trim(piece);
                    if (!piece.empty()) result.push_back(piece);
                    start += len;
                    while (start < s.size() && s[start] == ' ') start++;
                }
            } else {
                if (current_chunk.size() + s.size() + 1 > max_bytes) {
                    if (!current_chunk.empty()) {
                        result.push_back(current_chunk);
                        current_chunk.clear();
                    }
                }
                if (!current_chunk.empty()) current_chunk.push_back(' ');
                current_chunk += s;
            }
        }
        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }
        return result;
    }

    // Ensures input ends with a sentence terminator ('।' / '?' / '!' / '.') to prevent unclosed prompt hallucinations
    static inline std::string ensure_sentence_termination(const std::string& input) {
        if (input.empty()) return "";
        std::string s = input;
        trim(s);
        if (s.empty()) return "";

        // Check if last character is already punctuation
        if (s.size() >= 3 &&
            static_cast<unsigned char>(s[s.size() - 3]) == 0xE0 &&
            static_cast<unsigned char>(s[s.size() - 2]) == 0xA5 &&
            (static_cast<unsigned char>(s[s.size() - 1]) == 0xA4 || static_cast<unsigned char>(s[s.size() - 1]) == 0xA5)) {
            return s; // Already ends with '।' or '॥'
        }
        char last = s.back();
        if (last == '.' || last == '?' || last == '!' || last == ';' || last == ':' || last == ',') {
            return s;
        }

        // Append Devanagari danda '।' (0xE0 0xA5 0xA4)
        s.push_back(static_cast<char>(0xE0));
        s.push_back(static_cast<char>(0xA5));
        s.push_back(static_cast<char>(0xA4));
        return s;
    }

    // Strips trailing '।' or '.' from generated output so caller gets clean words
    static inline void clean_generated_output(std::string& text) {
        trim(text);
        while (text.size() >= 3 &&
               static_cast<unsigned char>(text[text.size() - 3]) == 0xE0 &&
               static_cast<unsigned char>(text[text.size() - 2]) == 0xA5 &&
               (static_cast<unsigned char>(text[text.size() - 1]) == 0xA4 || static_cast<unsigned char>(text[text.size() - 1]) == 0xA5)) {
            text.resize(text.size() - 3);
            trim(text);
        }
        while (!text.empty() && (text.back() == '.' || text.back() == ',' || text.back() == ';')) {
            text.pop_back();
            trim(text);
        }
    }

    // Kills 3+ consecutive word repetition loops while preserving natural Hindi doublets (e.g. "thoda thoda")
    static inline std::string deduplicate_words(const std::string& text, int max_consecutive = 2) {
        if (text.empty() || max_consecutive <= 0) return text;
        std::vector<std::string> words;
        std::string current;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n') {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
            } else {
                current.push_back(c);
            }
        }
        if (!current.empty()) words.push_back(current);
        if (words.empty()) return "";

        std::vector<std::string> filtered;
        filtered.reserve(words.size());

        std::string last_word;
        int repeat_count = 0;

        for (const auto& w : words) {
            std::string lower_w = w;
            for (char& ch : lower_w) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

            std::string lower_last = last_word;
            for (char& ch : lower_last) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

            if (lower_w == lower_last && !lower_w.empty()) {
                repeat_count++;
                if (repeat_count <= max_consecutive) {
                    filtered.push_back(w);
                }
            } else {
                repeat_count = 1;
                last_word = w;
                filtered.push_back(w);
            }
        }

        std::string out;
        for (size_t i = 0; i < filtered.size(); ++i) {
            if (i > 0) out.push_back(' ');
            out += filtered[i];
        }
        return out;
    }

    // Replaces case-insensitive matches with exact hotword spelling/casing
    static inline std::string apply_hotwords(const std::string& text, const std::vector<std::string>& hotwords) {
        if (text.empty() || hotwords.empty()) return text;
        std::string result = text;

        for (const auto& hw : hotwords) {
            if (hw.empty()) continue;
            std::string lower_hw = hw;
            for (char& c : lower_hw) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

            size_t pos = 0;
            while (pos < result.size()) {
                std::string lower_res = result.substr(pos);
                for (char& c : lower_res) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

                size_t found = lower_res.find(lower_hw);
                if (found == std::string::npos) break;

                size_t match_idx = pos + found;
                bool start_boundary = (match_idx == 0 || !std::isalnum(static_cast<unsigned char>(result[match_idx - 1])));
                bool end_boundary = (match_idx + lower_hw.size() >= result.size() ||
                                     !std::isalnum(static_cast<unsigned char>(result[match_idx + lower_hw.size()])));

                if (start_boundary && end_boundary) {
                    result.replace(match_idx, lower_hw.size(), hw);
                    pos = match_idx + hw.size();
                } else {
                    pos = match_idx + 1;
                }
            }
        }
        return result;
    }

    // Formats casing (none, lower, title, sentence)
    static inline std::string format_case(const std::string& text, const std::string& case_style) {
        if (text.empty() || case_style == "none" || case_style.empty()) return text;
        std::string out = text;

        if (case_style == "lower") {
            for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (case_style == "title") {
            bool new_word = true;
            for (char& c : out) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    new_word = true;
                } else if (new_word) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    new_word = false;
                }
            }
        } else if (case_style == "sentence") {
            bool start = true;
            for (char& c : out) {
                if (!std::isspace(static_cast<unsigned char>(c)) && start) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    start = false;
                }
            }
        }
        return out;
    }
};

} // namespace byt5
