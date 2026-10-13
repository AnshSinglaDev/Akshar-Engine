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
};

} // namespace byt5
