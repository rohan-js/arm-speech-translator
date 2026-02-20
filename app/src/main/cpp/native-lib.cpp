#include <jni.h>
#include <android/log.h>

#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#else
#error "ONNX Runtime C++ API header not found"
#endif

#if __has_include(<whisper.h>)
#include <whisper.h>
#else
#error "whisper.h not found"
#endif

#if __has_include(<ggml-cpu.h>)
#include <ggml-cpu.h>
#elif __has_include("third_party/whisper.cpp/ggml/include/ggml-cpu.h")
#include "third_party/whisper.cpp/ggml/include/ggml-cpu.h"
#else
#error "ggml-cpu.h not found"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr const char *kTag = "SpeechTranslation";
constexpr int kTrimMemoryRunningCritical = 15;
constexpr int kTrimMemoryComplete = 80;

void logi(const std::string &msg) {
    __android_log_print(ANDROID_LOG_INFO, kTag, "%s", msg.c_str());
}

enum class Lang : int {
    ENG = 0,
    HIN = 1,
};

enum class ThermalMode : int {
    NORMAL = 0,
    THROTTLED = 1,
    EMERGENCY = 2,
    CRITICAL = 3,
};

struct OrtSessionMeta {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_name_storage;
    std::vector<const char *> input_names;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> output_name_storage;
    std::vector<const char *> output_names;
    std::vector<ONNXTensorElementDataType> output_types;
    std::vector<std::vector<int64_t>> output_shapes;
};

struct SentencePiecePiece {
    std::string text;
    int type = 1;
    float score = 0.0f;
};

struct SentencePieceModel {
    bool loaded = false;
    int unk_id = 0;
    std::size_t max_piece_len = 0;
    std::vector<SentencePiecePiece> id_to_piece;
    std::unordered_map<std::string, int> piece_to_id;
    std::unordered_map<std::string, float> piece_score;
    std::vector<std::string> id_to_token;
};

struct NllbConfig {
    int eos_id = 2;
    int pad_id = 1;
    int unk_id = 3;
    int eng_lang_id = -1;
    int hin_lang_id = -1;
    int decoder_start_id = -1;
    bool uses_lang_ids = true;
};

struct NllbState {
    bool loaded = false;
    int threads = 0;
    int last_max_new_tokens = 0;
    int last_input_tokens = 0;
    std::int64_t last_used_ms = 0;
    std::string variant;
    OrtSessionMeta encoder;
    OrtSessionMeta decoder;
    OrtSessionMeta decoder_with_past;
    SentencePieceModel source_tokenizer;
    SentencePieceModel target_tokenizer;
    NllbConfig config;
};

struct PiperState {
    bool loaded = false;
    int loaded_voice = -1;
    std::int64_t last_used_ms = 0;
};

struct SileroVadState {
    bool loaded = false;
    std::string model_path;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_name_storage;
    std::vector<const char *> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<std::string> output_name_storage;
    std::vector<const char *> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<ONNXTensorElementDataType> output_types;
    int audio_input_idx = -1;
    int sr_input_idx = -1;
    std::vector<int> state_input_indices;
    std::vector<std::vector<int64_t>> state_input_shapes;
    std::vector<int> state_output_indices;
};

struct GlobalState {
    std::string models_dir;
    ThermalMode thermal_mode = ThermalMode::NORMAL;
    std::unique_ptr<Ort::Env> ort_env;

    NllbState nllb;
    PiperState piper;
    SileroVadState vad;

    bool fallback_en_hi_loaded = false;
    bool fallback_hi_en_loaded = false;
    bool ggml_features_logged = false;
    std::unordered_map<std::string, std::string> fallback_en_hi;
    std::unordered_map<std::string, std::string> fallback_hi_en;
};

GlobalState g_state;
std::mutex g_mu;

std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::size_t safe_element_count(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        return 1;
    }

    std::size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        if (count > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))) {
            return 0;
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::vector<int64_t> normalize_shape(std::vector<int64_t> shape) {
    if (shape.empty()) {
        return {1};
    }
    for (auto &dim : shape) {
        if (dim <= 0) {
            dim = 1;
        }
    }
    return shape;
}

std::string segments_to_string(const std::vector<std::pair<int, int>> &segments) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << "(" << segments[i].first << "," << segments[i].second << ")";
    }
    os << "]";
    return os.str();
}

bool ensure_ort_env_locked() {
    if (!g_state.ort_env) {
        g_state.ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "speech_translation");
    }
    return g_state.ort_env != nullptr;
}

bool ortSanityCheck(Ort::Env &env, const std::string &model_path, std::string &error) {
    std::ifstream test_file(model_path, std::ios::binary);
    if (!test_file.good()) {
        error = "model missing: " + model_path;
        return false;
    }
    test_file.close();

    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        Ort::Session session(env, model_path.c_str(), session_options);
        if (session.GetInputCount() == 0 || session.GetOutputCount() == 0) {
            error = "invalid model io count";
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);

        if (!input_name || !output_name) {
            error = "unable to read io names";
            return false;
        }

        auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType elem_type = input_info.GetElementType();
        std::vector<int64_t> input_shape = input_info.GetShape();
        if (input_shape.empty()) {
            input_shape = {1};
        } else {
            for (auto &dim : input_shape) {
                if (dim <= 0) {
                    dim = 1;
                }
            }
        }

        const std::size_t input_count = safe_element_count(input_shape);
        if (input_count == 0) {
            error = "invalid input shape";
            return false;
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault
        );

        const char *input_names[] = {input_name.get()};
        const char *output_names[] = {output_name.get()};

        std::vector<Ort::Value> outputs;
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::vector<float> input_values(input_count, 2.0f);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_values.data(),
                input_values.size(),
                input_shape.data(),
                input_shape.size()
            );
            outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            std::vector<int64_t> input_values(input_count, 2);
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info,
                input_values.data(),
                input_values.size(),
                input_shape.data(),
                input_shape.size()
            );
            outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
        } else {
            error = "unsupported sanity input type";
            return false;
        }

        if (outputs.empty() || !outputs.front().IsTensor()) {
            error = "sanity run produced no tensor output";
            return false;
        }

        return true;
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    } catch (const std::exception &ex) {
        error = ex.what();
        return false;
    } catch (...) {
        error = "unknown exception";
        return false;
    }
}

std::string normalize_key(const std::string &text) {
    std::string out;
    out.reserve(text.size());

    bool prev_space = false;
    for (char ch : text) {
        unsigned char u = static_cast<unsigned char>(ch);
        char lower = (u < 128) ? static_cast<char>(std::tolower(u)) : ch;

        bool is_punct = (lower == '.' || lower == ',' || lower == ';' || lower == ':' || lower == '!' || lower == '?');
        if (is_punct) {
            continue;
        }

        if (std::isspace(u)) {
            if (!prev_space) {
                out.push_back(' ');
            }
            prev_space = true;
        } else {
            out.push_back(lower);
            prev_space = false;
        }
    }

    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }

    return out;
}

bool load_phrase_table_locked(bool en_hi) {
    auto &loaded = en_hi ? g_state.fallback_en_hi_loaded : g_state.fallback_hi_en_loaded;
    auto &table = en_hi ? g_state.fallback_en_hi : g_state.fallback_hi_en;
    if (loaded) {
        return true;
    }

    const std::string filename = en_hi ? "phrases_en_hi.tsv" : "phrases_hi_en.tsv";
    std::string path = g_state.models_dir + "/fallback/" + filename;

    std::ifstream input(path);
    if (!input.is_open()) {
        logi("Failed to open fallback table: " + path);
        return false;
    }

    table.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        auto tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }

        std::string key = normalize_key(line.substr(0, tab));
        std::string value = line.substr(tab + 1);
        if (!key.empty() && !value.empty()) {
            table[key] = value;
        }
    }

    loaded = true;
    return true;
}

void unload_nllb_locked() {
    const int last_input_tokens = g_state.nllb.last_input_tokens;
    const int last_max_new_tokens = g_state.nllb.last_max_new_tokens;
    g_state.nllb = NllbState{};
    g_state.nllb.last_input_tokens = last_input_tokens;
    g_state.nllb.last_max_new_tokens = last_max_new_tokens;
}

void unload_piper_locked() {
    g_state.piper.loaded = false;
    g_state.piper.loaded_voice = -1;
}

void unload_silero_locked() {
    g_state.vad = SileroVadState{};
}

void maybe_idle_unload_locked() {
    const auto now = now_ms();
    if (g_state.nllb.loaded && (now - g_state.nllb.last_used_ms) > 60000) {
        unload_nllb_locked();
    }

    if (g_state.piper.loaded && (now - g_state.piper.last_used_ms) > 120000) {
        unload_piper_locked();
    }
}

bool ensure_silero_loaded_locked(std::string &error) {
    if (g_state.vad.loaded && g_state.vad.session) {
        return true;
    }
    if (!ensure_ort_env_locked()) {
        error = "ORT environment unavailable";
        return false;
    }

    std::string model_path = g_state.models_dir + "/silero_vad.onnx";
    std::ifstream direct_file(model_path, std::ios::binary);
    if (!direct_file.good()) {
        model_path = g_state.models_dir + "/silero/silero_vad.onnx";
    } else {
        direct_file.close();
    }

    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file.good()) {
        error = "Silero model missing: " + model_path;
        return false;
    }
    model_file.close();

    try {
        SileroVadState vad;
        vad.model_path = model_path;

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        vad.session = std::make_unique<Ort::Session>(*g_state.ort_env, vad.model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = vad.session->GetInputCount();
        const size_t output_count = vad.session->GetOutputCount();
        vad.input_name_storage.reserve(input_count);
        vad.input_names.reserve(input_count);
        vad.input_shapes.reserve(input_count);
        vad.input_types.reserve(input_count);
        vad.output_name_storage.reserve(output_count);
        vad.output_names.reserve(output_count);
        vad.output_shapes.reserve(output_count);
        vad.output_types.reserve(output_count);

        for (size_t i = 0; i < input_count; ++i) {
            Ort::AllocatedStringPtr input_name = vad.session->GetInputNameAllocated(i, allocator);
            std::string name = input_name ? std::string(input_name.get()) : ("input_" + std::to_string(i));
            vad.input_name_storage.push_back(name);
            vad.input_names.push_back(vad.input_name_storage.back().c_str());

            auto info = vad.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType elem_type = info.GetElementType();
            std::vector<int64_t> shape = info.GetShape();

            vad.input_types.push_back(elem_type);
            vad.input_shapes.push_back(shape);

            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && vad.sr_input_idx < 0) {
                vad.sr_input_idx = static_cast<int>(i);
                continue;
            }
            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                if (vad.audio_input_idx < 0) {
                    vad.audio_input_idx = static_cast<int>(i);
                } else {
                    vad.state_input_indices.push_back(static_cast<int>(i));
                    vad.state_input_shapes.push_back(shape);
                }
            }
        }

        for (size_t i = 0; i < output_count; ++i) {
            Ort::AllocatedStringPtr output_name = vad.session->GetOutputNameAllocated(i, allocator);
            std::string name = output_name ? std::string(output_name.get()) : ("output_" + std::to_string(i));
            vad.output_name_storage.push_back(name);
            vad.output_names.push_back(vad.output_name_storage.back().c_str());

            auto info = vad.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            vad.output_types.push_back(info.GetElementType());
            vad.output_shapes.push_back(info.GetShape());
        }

        if (vad.audio_input_idx < 0 || vad.output_names.empty()) {
            error = "Silero signature unsupported";
            return false;
        }

        for (size_t i = 1; i < vad.output_types.size(); ++i) {
            if (vad.output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                vad.state_output_indices.push_back(static_cast<int>(i));
                if (vad.state_output_indices.size() >= vad.state_input_indices.size()) {
                    break;
                }
            }
        }

        vad.loaded = true;
        g_state.vad = std::move(vad);
        logi("Silero VAD loaded: " + g_state.vad.model_path);
        return true;
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    } catch (const std::exception &ex) {
        error = ex.what();
        return false;
    } catch (...) {
        error = "unknown Silero load error";
        return false;
    }
}

int mode_threads(ThermalMode mode) {
    return mode == ThermalMode::NORMAL ? 2 : 1;
}

int mode_cap(ThermalMode mode) {
    return mode == ThermalMode::NORMAL ? 64 : 48;
}

int whisper_threads_for_mode(ThermalMode mode) {
    return mode == ThermalMode::NORMAL ? 2 : 1;
}

const char * whisper_lang_code(Lang lang) {
    return lang == Lang::HIN ? "hi" : "en";
}

int compute_max_new_tokens(int input_tokens, ThermalMode mode) {
    const int cap = mode_cap(mode);
    const int scaled = static_cast<int>(std::ceil(input_tokens * 1.5));
    return std::min(cap, std::max(1, scaled));
}

int rough_word_token_count(const std::string &text) {
    const std::string norm = normalize_key(text);
    if (norm.empty()) {
        return 1;
    }
    int count = 0;
    bool in_word = false;
    for (char ch : norm) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (in_word) {
                count++;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if (in_word) {
        count++;
    }
    return std::max(1, count);
}

bool file_exists(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

bool read_file_to_string(const std::string &path, std::string &out, std::string &error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        error = "Unable to open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::string trim_ascii(const std::string &value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        start++;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(start, end - start);
}

std::string to_lower_ascii(std::string value) {
    for (char &ch : value) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (u < 128) {
            ch = static_cast<char>(std::tolower(u));
        }
    }
    return value;
}

bool contains_ci(const std::string &value, const std::string &needle) {
    if (needle.empty()) {
        return true;
    }
    const std::string lower_value = to_lower_ascii(value);
    const std::string lower_needle = to_lower_ascii(needle);
    return lower_value.find(lower_needle) != std::string::npos;
}

bool read_json_int(const std::string &json, const std::string &key, int &result) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        pos++;
    }
    bool neg = false;
    if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
        neg = json[pos] == '-';
        pos++;
    }
    if (pos >= json.size() || !std::isdigit(static_cast<unsigned char>(json[pos]))) {
        return false;
    }
    long long value = 0;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
        value = value * 10 + static_cast<long long>(json[pos] - '0');
        pos++;
    }
    result = static_cast<int>(neg ? -value : value);
    return true;
}

bool parse_json_string_token(const std::string &json, std::size_t &pos, std::string &out) {
    if (pos >= json.size() || json[pos] != '"') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < json.size()) {
        const char ch = json[pos++];
        if (ch == '"') {
            return true;
        }
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (pos >= json.size()) {
            return false;
        }
        const char esc = json[pos++];
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > json.size()) {
                    return false;
                }
                auto hex_to_int = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                    return -1;
                };
                const int h0 = hex_to_int(json[pos]);
                const int h1 = hex_to_int(json[pos + 1]);
                const int h2 = hex_to_int(json[pos + 2]);
                const int h3 = hex_to_int(json[pos + 3]);
                if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
                    return false;
                }
                const std::uint32_t cp = static_cast<std::uint32_t>(
                    (h0 << 12) | (h1 << 8) | (h2 << 4) | h3
                );
                if (cp <= 0x7F) {
                    out.push_back(static_cast<char>(cp));
                } else if (cp <= 0x7FF) {
                    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
                pos += 4;
                break;
            }
            default:
                out.push_back(esc);
                break;
        }
    }
    return false;
}

bool load_vocab_json_map(
    const std::string &path,
    std::unordered_map<std::string, int> &token_to_id,
    std::vector<std::string> &id_to_token,
    std::string &error) {
    std::string json;
    if (!read_file_to_string(path, json, error)) {
        return false;
    }

    token_to_id.clear();
    id_to_token.clear();
    std::size_t pos = 0;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    if (pos >= json.size() || json[pos] != '{') {
        error = "invalid vocab json";
        return false;
    }
    ++pos;

    while (pos < json.size()) {
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        if (pos < json.size() && json[pos] == '}') {
            ++pos;
            break;
        }

        std::string token;
        if (!parse_json_string_token(json, pos, token)) {
            error = "invalid vocab key";
            return false;
        }
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        if (pos >= json.size() || json[pos] != ':') {
            error = "invalid vocab separator";
            return false;
        }
        ++pos;
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        bool neg = false;
        if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
            neg = json[pos] == '-';
            ++pos;
        }
        if (pos >= json.size() || !std::isdigit(static_cast<unsigned char>(json[pos]))) {
            error = "invalid vocab id";
            return false;
        }
        int value = 0;
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) {
            value = value * 10 + (json[pos] - '0');
            ++pos;
        }
        if (neg) {
            value = -value;
        }
        if (value >= 0) {
            token_to_id[token] = value;
            if (static_cast<std::size_t>(value) >= id_to_token.size()) {
                id_to_token.resize(static_cast<std::size_t>(value) + 1);
            }
            id_to_token[static_cast<std::size_t>(value)] = token;
        }

        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
            ++pos;
        }
        if (pos < json.size() && json[pos] == ',') {
            ++pos;
        }
    }

    if (token_to_id.empty()) {
        error = "empty vocab map";
        return false;
    }
    return true;
}

int first_json_int_or_default(const std::string &json, const std::vector<std::string> &keys, int fallback) {
    for (const auto &key : keys) {
        int value = 0;
        if (read_json_int(json, key, value)) {
            return value;
        }
    }
    return fallback;
}

bool pb_read_varint(const std::string &buf, std::size_t &pos, std::uint64_t &value) {
    value = 0;
    int shift = 0;
    while (pos < buf.size() && shift <= 63) {
        const unsigned char byte = static_cast<unsigned char>(buf[pos++]);
        value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return true;
        }
        shift += 7;
    }
    return false;
}

bool pb_read_fixed32(const std::string &buf, std::size_t &pos, std::uint32_t &value) {
    if (pos + 4 > buf.size()) {
        return false;
    }
    value = static_cast<std::uint32_t>(static_cast<unsigned char>(buf[pos]))
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(buf[pos + 1])) << 8)
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(buf[pos + 2])) << 16)
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(buf[pos + 3])) << 24);
    pos += 4;
    return true;
}

bool pb_skip_field(const std::string &buf, std::size_t &pos, int wire_type) {
    switch (wire_type) {
        case 0: {
            std::uint64_t ignored = 0;
            return pb_read_varint(buf, pos, ignored);
        }
        case 1: {
            if (pos + 8 > buf.size()) {
                return false;
            }
            pos += 8;
            return true;
        }
        case 2: {
            std::uint64_t len = 0;
            if (!pb_read_varint(buf, pos, len) || pos + len > buf.size()) {
                return false;
            }
            pos += static_cast<std::size_t>(len);
            return true;
        }
        case 5: {
            if (pos + 4 > buf.size()) {
                return false;
            }
            pos += 4;
            return true;
        }
        default:
            return false;
    }
}

bool parse_sentencepiece_entry(const std::string &entry, SentencePiecePiece &piece) {
    std::size_t pos = 0;
    bool has_piece = false;
    while (pos < entry.size()) {
        std::uint64_t key = 0;
        if (!pb_read_varint(entry, pos, key)) {
            return false;
        }
        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x07);
        if (field == 1 && wire == 2) {
            std::uint64_t len = 0;
            if (!pb_read_varint(entry, pos, len) || pos + len > entry.size()) {
                return false;
            }
            piece.text.assign(entry.data() + pos, static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);
            has_piece = true;
        } else if (field == 2 && wire == 5) {
            std::uint32_t bits = 0;
            if (!pb_read_fixed32(entry, pos, bits)) {
                return false;
            }
            float score = 0.0f;
            std::memcpy(&score, &bits, sizeof(float));
            piece.score = score;
        } else if (field == 3 && wire == 0) {
            std::uint64_t v = 0;
            if (!pb_read_varint(entry, pos, v)) {
                return false;
            }
            piece.type = static_cast<int>(v);
        } else {
            if (!pb_skip_field(entry, pos, wire)) {
                return false;
            }
        }
    }
    return has_piece;
}

bool load_sentencepiece_model(const std::string &path, int unk_id, SentencePieceModel &model, std::string &error) {
    std::string blob;
    if (!read_file_to_string(path, blob, error)) {
        return false;
    }

    SentencePieceModel parsed;
    parsed.unk_id = unk_id;
    std::size_t pos = 0;
    while (pos < blob.size()) {
        std::uint64_t key = 0;
        if (!pb_read_varint(blob, pos, key)) {
            error = "SentencePiece parse failed (key)";
            return false;
        }

        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x07);
        if (field == 1 && wire == 2) {
            std::uint64_t len = 0;
            if (!pb_read_varint(blob, pos, len) || pos + len > blob.size()) {
                error = "SentencePiece parse failed (piece length)";
                return false;
            }
            std::string entry(blob.data() + pos, static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);

            SentencePiecePiece piece;
            if (parse_sentencepiece_entry(entry, piece)) {
                const int id = static_cast<int>(parsed.id_to_piece.size());
                parsed.max_piece_len = std::max(parsed.max_piece_len, piece.text.size());
                parsed.piece_to_id[piece.text] = id;
                parsed.piece_score[piece.text] = piece.score;
                parsed.id_to_piece.push_back(std::move(piece));
            }
        } else {
            if (!pb_skip_field(blob, pos, wire)) {
                error = "SentencePiece parse failed (skip)";
                return false;
            }
        }
    }

    if (parsed.id_to_piece.empty()) {
        error = "SentencePiece has no pieces";
        return false;
    }

    int detected_unk_id = -1;
    for (int i = 0; i < static_cast<int>(parsed.id_to_piece.size()); ++i) {
        if (parsed.id_to_piece[static_cast<std::size_t>(i)].type == 2) {
            detected_unk_id = i;
            break;
        }
    }
    if (detected_unk_id >= 0) {
        parsed.unk_id = detected_unk_id;
    } else if (parsed.unk_id < 0 || parsed.unk_id >= static_cast<int>(parsed.id_to_piece.size())) {
        parsed.unk_id = 0;
    }

    parsed.loaded = true;
    model = std::move(parsed);
    return true;
}

std::size_t utf8_codepoint_len(const std::string &text, std::size_t pos) {
    if (pos >= text.size()) {
        return 0;
    }
    const unsigned char c = static_cast<unsigned char>(text[pos]);
    if ((c & 0x80) == 0) {
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        return (pos + 2 <= text.size()) ? 2 : 1;
    }
    if ((c & 0xF0) == 0xE0) {
        return (pos + 3 <= text.size()) ? 3 : 1;
    }
    if ((c & 0xF8) == 0xF0) {
        return (pos + 4 <= text.size()) ? 4 : 1;
    }
    return 1;
}

std::string to_sentencepiece_surface(const std::string &text) {
    static const std::string marker = "\xE2\x96\x81"; // ▁
    std::string out;
    out.reserve(text.size() + 8);
    bool need_marker = true;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t cp_len = utf8_codepoint_len(text, pos);
        const std::string cp = text.substr(pos, cp_len);
        const unsigned char first = static_cast<unsigned char>(cp[0]);
        const bool is_ascii_space = cp_len == 1 && std::isspace(first);
        if (is_ascii_space) {
            need_marker = true;
        } else {
            if (need_marker) {
                out += marker;
                need_marker = false;
            }
            out += cp;
        }
        pos += cp_len;
    }
    return out;
}

std::vector<int64_t> sentencepiece_encode(const SentencePieceModel &model, const std::string &text) {
    std::vector<int64_t> ids;
    if (!model.loaded) {
        return ids;
    }

    std::string surface = to_sentencepiece_surface(text);
    if (surface.empty()) {
        return ids;
    }

    const std::size_t n = surface.size();
    constexpr float kNegInf = -1.0e30f;
    constexpr float kUnkPenalty = -100.0f;
    std::vector<float> best_score(n + 1, kNegInf);
    std::vector<int> best_id(n + 1, -1);
    std::vector<std::size_t> best_next(n + 1, n + 1);
    best_score[n] = 0.0f;

    for (std::size_t pos = n; pos-- > 0;) {
        const std::size_t max_len = std::min(model.max_piece_len, n - pos);
        for (std::size_t len = 1; len <= max_len; ++len) {
            auto it = model.piece_to_id.find(surface.substr(pos, len));
            if (it == model.piece_to_id.end()) {
                continue;
            }
            const int id = it->second;
            if (best_score[pos + len] <= kNegInf / 2.0f) {
                continue;
            }
            float piece_score = 0.0f;
            auto score_it = model.piece_score.find(it->first);
            if (score_it != model.piece_score.end()) {
                piece_score = score_it->second;
            }
            const float score = piece_score + best_score[pos + len];
            if (score > best_score[pos]) {
                best_score[pos] = score;
                best_id[pos] = id;
                best_next[pos] = pos + len;
            }
        }

        const std::size_t cp_len = std::max<std::size_t>(1, utf8_codepoint_len(surface, pos));
        if (pos + cp_len <= n && best_score[pos + cp_len] > kNegInf / 2.0f) {
            const float unk_score = kUnkPenalty + best_score[pos + cp_len];
            if (unk_score > best_score[pos]) {
                best_score[pos] = unk_score;
                best_id[pos] = model.unk_id;
                best_next[pos] = pos + cp_len;
            }
        }
    }

    std::size_t pos = 0;
    while (pos < n) {
        int id = best_id[pos];
        std::size_t next = best_next[pos];
        if (id < 0 || next <= pos || next > n) {
            id = model.unk_id;
            next = pos + std::max<std::size_t>(1, utf8_codepoint_len(surface, pos));
        }
        ids.push_back(static_cast<int64_t>(id));
        pos = next;
    }
    return ids;
}

bool parse_byte_piece(const std::string &piece, char &out_byte) {
    if (piece.size() != 6 || piece.rfind("<0x", 0) != 0 || piece.back() != '>') {
        return false;
    }
    const auto hex = piece.substr(3, 2);
    char *end = nullptr;
    long value = std::strtol(hex.c_str(), &end, 16);
    if (end == nullptr || *end != '\0' || value < 0 || value > 255) {
        return false;
    }
    out_byte = static_cast<char>(value);
    return true;
}

std::string sentencepiece_decode(
    const SentencePieceModel &model,
    const std::vector<int64_t> &tokens,
    const NllbConfig &config) {
    static const std::string marker = "\xE2\x96\x81"; // ▁
    std::string out;
    for (const int64_t token : tokens) {
        if (token == config.eos_id || token == config.pad_id) {
            continue;
        }
        std::string piece_text;
        int piece_type = 1;
        if (!model.id_to_token.empty()) {
            if (token < 0 || token >= static_cast<int64_t>(model.id_to_token.size())) {
                continue;
            }
            piece_text = model.id_to_token[static_cast<std::size_t>(token)];
            if (piece_text.empty()) {
                continue;
            }
            if (!piece_text.empty() && piece_text.front() == '<' && piece_text.back() == '>') {
                continue;
            }
        } else {
            if (token < 0 || token >= static_cast<int64_t>(model.id_to_piece.size())) {
                continue;
            }
            const auto &piece = model.id_to_piece[static_cast<std::size_t>(token)];
            piece_text = piece.text;
            piece_type = piece.type;
        }
        if (piece_type == 3) { // CONTROL
            continue;
        }
        if (piece_text.empty()) {
            continue;
        }

        char byte_value = 0;
        if (parse_byte_piece(piece_text, byte_value)) {
            out.push_back(byte_value);
            continue;
        }

        if (piece_text.rfind(marker, 0) == 0) {
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
            out += piece_text.substr(marker.size());
        } else {
            out += piece_text;
        }
    }
    return trim_ascii(out);
}

bool load_ort_session_meta(
    const std::string &path,
    int threads,
    OrtSessionMeta &meta,
    std::string &error) {
    if (!file_exists(path)) {
        error = "missing model: " + path;
        return false;
    }
    if (!ensure_ort_env_locked()) {
        error = "ORT environment unavailable";
        return false;
    }

    try {
        OrtSessionMeta loaded;
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        loaded.session = std::make_unique<Ort::Session>(*g_state.ort_env, path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t in_count = loaded.session->GetInputCount();
        const std::size_t out_count = loaded.session->GetOutputCount();

        loaded.input_name_storage.reserve(in_count);
        loaded.input_names.reserve(in_count);
        loaded.input_types.reserve(in_count);
        loaded.input_shapes.reserve(in_count);

        loaded.output_name_storage.reserve(out_count);
        loaded.output_names.reserve(out_count);
        loaded.output_types.reserve(out_count);
        loaded.output_shapes.reserve(out_count);

        for (std::size_t i = 0; i < in_count; ++i) {
            Ort::AllocatedStringPtr name = loaded.session->GetInputNameAllocated(i, allocator);
            loaded.input_name_storage.push_back(name ? std::string(name.get()) : ("input_" + std::to_string(i)));
            loaded.input_names.push_back(loaded.input_name_storage.back().c_str());
            auto info = loaded.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            loaded.input_types.push_back(info.GetElementType());
            loaded.input_shapes.push_back(info.GetShape());
        }

        for (std::size_t i = 0; i < out_count; ++i) {
            Ort::AllocatedStringPtr name = loaded.session->GetOutputNameAllocated(i, allocator);
            loaded.output_name_storage.push_back(name ? std::string(name.get()) : ("output_" + std::to_string(i)));
            loaded.output_names.push_back(loaded.output_name_storage.back().c_str());
            auto info = loaded.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            loaded.output_types.push_back(info.GetElementType());
            loaded.output_shapes.push_back(info.GetShape());
        }

        meta = std::move(loaded);
        return true;
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    } catch (const std::exception &ex) {
        error = ex.what();
        return false;
    } catch (...) {
        error = "unknown ORT session load error";
        return false;
    }
}

bool load_nllb_config(const std::string &config_path, NllbConfig &config, std::string &error) {
    std::string json;
    if (!read_file_to_string(config_path, json, error)) {
        return false;
    }
    NllbConfig parsed;
    parsed.eos_id = first_json_int_or_default(json, {"eos_id", "eosTokenId"}, 2);
    parsed.pad_id = first_json_int_or_default(json, {"pad_id", "padTokenId"}, 1);
    parsed.unk_id = first_json_int_or_default(json, {"unk_id", "unkTokenId"}, 3);
    parsed.decoder_start_id = first_json_int_or_default(
        json,
        {"decoder_start_id", "decoderStartTokenId", "decoder_start_token_id"},
        -1
    );

    parsed.eng_lang_id = first_json_int_or_default(
        json,
        {"eng_Latn", "en", "en_XX", "en_Latn"},
        -1
    );
    parsed.hin_lang_id = first_json_int_or_default(
        json,
        {"hin_Deva", "hi", "hi_IN", "hi_Deva"},
        -1
    );

    if (parsed.eng_lang_id < 0 || parsed.hin_lang_id < 0) {
        if (parsed.decoder_start_id >= 0) {
            // Marian-style config (direction-specific model) does not use language IDs.
            parsed.uses_lang_ids = false;
            parsed.eng_lang_id = 0;
            parsed.hin_lang_id = 0;
        } else {
            error = "language IDs not found in config";
            return false;
        }
    } else {
        parsed.uses_lang_ids = true;
    }

    config = parsed;
    return true;
}

int nllb_language_id(const NllbConfig &config, Lang lang) {
    if (!config.uses_lang_ids) {
        return 0;
    }
    return lang == Lang::HIN ? config.hin_lang_id : config.eng_lang_id;
}

bool ensure_nllb_loaded_locked(ThermalMode mode, Lang src, Lang tgt) {
    if (mode == ThermalMode::CRITICAL) {
        unload_nllb_locked();
        return false;
    }

    const int needed_threads = mode_threads(mode);
    // Ensure correct direction is loaded for Marian models (direction-specific sessions).
    const bool wants_en_hi = (src == Lang::ENG && tgt == Lang::HIN);
    const bool wants_hi_en = (src == Lang::HIN && tgt == Lang::ENG);
    const std::string wanted_variant = wants_en_hi ? "en_hi" : (wants_hi_en ? "hi_en" : "");

    if (g_state.nllb.loaded && g_state.nllb.threads == needed_threads && g_state.nllb.variant == wanted_variant) {
        return true;
    }

    unload_nllb_locked();

    std::string error;
    NllbState loaded;
    loaded.threads = needed_threads;

    auto resolve_path = [](const std::vector<std::string> &candidates) -> std::string {
        for (const auto &candidate : candidates) {
            if (file_exists(candidate)) {
                return candidate;
            }
        }
        return candidates.empty() ? std::string() : candidates.front();
    };

    std::string nllb_dir = g_state.models_dir + "/nllb";
    {
        const std::string hint_path = g_state.models_dir + "/nllb_external_dir.txt";
        if (file_exists(hint_path)) {
            std::string hinted;
            std::string hint_error;
            if (read_file_to_string(hint_path, hinted, hint_error)) {
                // trim whitespace/newlines
                while (!hinted.empty() && std::isspace(static_cast<unsigned char>(hinted.front()))) {
                    hinted.erase(hinted.begin());
                }
                while (!hinted.empty() && std::isspace(static_cast<unsigned char>(hinted.back()))) {
                    hinted.pop_back();
                }
                if (!hinted.empty() && file_exists(hinted)) {
                    nllb_dir = hinted;
                }
            }
        }
    }
    // Prefer direction-specific subdir if present.
    if (!wanted_variant.empty()) {
        const std::string candidate_dir = nllb_dir + "/" + wanted_variant;
        if (file_exists(candidate_dir + "/encoder_model_int8.onnx")) {
            nllb_dir = candidate_dir;
        }
    }
    const std::string encoder_path = resolve_path({
        nllb_dir + "/encoder_model_int8.onnx",
        g_state.models_dir + "/encoder_model_int8.onnx",
    });
    const std::string decoder_path = resolve_path({
        nllb_dir + "/decoder_model_int8.onnx",
        g_state.models_dir + "/decoder_model_int8.onnx",
    });
    const std::string decoder_past_path = resolve_path({
        nllb_dir + "/decoder_with_past_model_int8.onnx",
        g_state.models_dir + "/decoder_with_past_model_int8.onnx",
    });
    const std::string tokenizer_path = resolve_path({
        nllb_dir + "/tokenizer.model",
        g_state.models_dir + "/tokenizer.model",
    });
    const std::string source_spm_path = resolve_path({
        nllb_dir + "/source.spm",
        g_state.models_dir + "/source.spm",
    });
    const std::string target_spm_path = resolve_path({
        nllb_dir + "/target.spm",
        g_state.models_dir + "/target.spm",
    });
    const std::string config_path = resolve_path({
        nllb_dir + "/nllb_config.json",
        g_state.models_dir + "/nllb_config.json",
    });
    const std::string vocab_path = resolve_path({
        nllb_dir + "/vocab.json",
        g_state.models_dir + "/vocab.json",
    });

    if (!load_nllb_config(config_path, loaded.config, error)) {
        logi("NLLB config load failed: " + error);
        return false;
    }

    if (file_exists(tokenizer_path)) {
        // NLLB-style single tokenizer.
        if (!load_sentencepiece_model(tokenizer_path, loaded.config.unk_id, loaded.source_tokenizer, error)) {
            logi("NLLB tokenizer load failed: " + error);
            return false;
        }
        loaded.target_tokenizer = loaded.source_tokenizer;
    } else {
        // Marian-style split tokenizers.
        if (!load_sentencepiece_model(source_spm_path, loaded.config.unk_id, loaded.source_tokenizer, error)) {
            logi("NLLB source tokenizer load failed: " + error);
            return false;
        }
        if (!load_sentencepiece_model(target_spm_path, loaded.config.unk_id, loaded.target_tokenizer, error)) {
            logi("NLLB target tokenizer load failed: " + error);
            return false;
        }
        if (file_exists(vocab_path)) {
            std::unordered_map<std::string, int> vocab_to_id;
            std::vector<std::string> id_to_token;
            if (!load_vocab_json_map(vocab_path, vocab_to_id, id_to_token, error)) {
                logi("NLLB vocab load failed: " + error);
                return false;
            }
            loaded.source_tokenizer.piece_to_id = vocab_to_id;
            loaded.target_tokenizer.piece_to_id = vocab_to_id;
            loaded.source_tokenizer.id_to_token = id_to_token;
            loaded.target_tokenizer.id_to_token = id_to_token;
        } else {
            logi("NLLB warning: Marian vocab.json missing, quality may be degraded");
        }
    }

    if (!load_ort_session_meta(encoder_path, needed_threads, loaded.encoder, error)) {
        logi("NLLB encoder load failed: " + error);
        return false;
    }
    if (!load_ort_session_meta(decoder_path, needed_threads, loaded.decoder, error)) {
        logi("NLLB decoder load failed: " + error);
        return false;
    }
    if (!load_ort_session_meta(decoder_past_path, needed_threads, loaded.decoder_with_past, error)) {
        logi("NLLB decoder-with-past load failed: " + error);
        return false;
    }

    loaded.loaded = true;
    loaded.last_used_ms = now_ms();
    loaded.variant = wanted_variant;
    g_state.nllb = std::move(loaded);
    logi("NLLB loaded (int8 ONNX)");
    return true;
}

bool ensure_nllb_loaded_locked(ThermalMode mode) {
    // Direction-agnostic warmup: do not load large models here (direction is unknown).
    // Translate() will load the correct direction lazily.
    if (mode == ThermalMode::CRITICAL) {
        unload_nllb_locked();
        return false;
    }
    return true;
}

std::string phrase_fallback_translate_locked(const std::string &text, Lang src, Lang tgt) {
    const bool en_hi = (src == Lang::ENG && tgt == Lang::HIN);
    const bool hi_en = (src == Lang::HIN && tgt == Lang::ENG);

    if (!en_hi && !hi_en) {
        return "[CRITICAL MODE] " + text;
    }

    if (!load_phrase_table_locked(en_hi)) {
        return "[CRITICAL MODE] " + text;
    }

    auto &table = en_hi ? g_state.fallback_en_hi : g_state.fallback_hi_en;
    std::string key = normalize_key(text);

    auto it = table.find(key);
    if (it != table.end()) {
        return it->second;
    }

    return "[CRITICAL MODE] " + text;
}

int find_logits_output_index(const OrtSessionMeta &meta, const std::vector<Ort::Value> &outputs) {
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        if (i < static_cast<int>(meta.output_name_storage.size())
            && contains_ci(meta.output_name_storage[static_cast<std::size_t>(i)], "logits")) {
            return i;
        }
    }
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        if (!outputs[static_cast<std::size_t>(i)].IsTensor()) {
            continue;
        }
        auto info = outputs[static_cast<std::size_t>(i)].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            return i;
        }
    }
    return -1;
}

int64_t argmax_last_logits(const Ort::Value &logits) {
    auto info = logits.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const std::size_t count = info.GetElementCount();
    if (count == 0) {
        return 0;
    }

    std::size_t vocab = count;
    if (!shape.empty() && shape.back() > 0) {
        vocab = static_cast<std::size_t>(shape.back());
    }
    if (vocab == 0 || vocab > count) {
        vocab = count;
    }
    const std::size_t start = count - vocab;

    int64_t arg = 0;
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *data = logits.GetTensorData<float>();
        float best = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < vocab; ++i) {
            const float value = data[start + i];
            if (value > best) {
                best = value;
                arg = static_cast<int64_t>(i);
            }
        }
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double *data = logits.GetTensorData<double>();
        double best = std::numeric_limits<double>::lowest();
        for (std::size_t i = 0; i < vocab; ++i) {
            const double value = data[start + i];
            if (value > best) {
                best = value;
                arg = static_cast<int64_t>(i);
            }
        }
    }
    return arg;
}

bool run_encoder_locked(
    const std::vector<int64_t> &encoder_input_ids,
    std::vector<float> &encoder_hidden_states,
    std::vector<int64_t> &encoder_hidden_shape,
    std::string &error) {
    auto &meta = g_state.nllb.encoder;
    if (!meta.session) {
        error = "encoder session unavailable";
        return false;
    }

    const int64_t src_len = static_cast<int64_t>(std::max<std::size_t>(1, encoder_input_ids.size()));
    std::vector<int64_t> attention_mask(static_cast<std::size_t>(src_len), 1);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<Ort::Value> inputs;
    inputs.reserve(meta.input_names.size());
    std::vector<std::vector<int64_t>> int64_buffers;
    std::vector<std::vector<float>> float_buffers;
    int64_buffers.reserve(meta.input_names.size());
    float_buffers.reserve(meta.input_names.size());

    for (std::size_t i = 0; i < meta.input_names.size(); ++i) {
        const auto type = meta.input_types[i];
        const std::string name = to_lower_ascii(meta.input_name_storage[i]);
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            std::vector<int64_t> shape = {1, src_len};
            std::vector<int64_t> values;
            if (contains_ci(name, "input_ids")) {
                values = encoder_input_ids;
            } else if (contains_ci(name, "attention_mask")) {
                values = attention_mask;
            } else {
                shape = normalize_shape(meta.input_shapes[i]);
                std::size_t n = safe_element_count(shape);
                if (n == 0) {
                    shape = {1};
                    n = 1;
                }
                values.assign(n, 0);
            }
            int64_buffers.push_back(std::move(values));
            auto &stored = int64_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else {
            std::vector<int64_t> shape = normalize_shape(meta.input_shapes[i]);
            std::size_t n = safe_element_count(shape);
            if (n == 0) {
                shape = {1};
                n = 1;
            }
            float_buffers.emplace_back(n, 0.0f);
            auto &stored = float_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        }
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = meta.session->Run(
            Ort::RunOptions{nullptr},
            meta.input_names.data(),
            inputs.data(),
            inputs.size(),
            meta.output_names.data(),
            meta.output_names.size()
        );
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    }

    int output_index = -1;
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        if (!outputs[static_cast<std::size_t>(i)].IsTensor()) {
            continue;
        }
        if (i < static_cast<int>(meta.output_name_storage.size())
            && contains_ci(meta.output_name_storage[static_cast<std::size_t>(i)], "last_hidden_state")) {
            output_index = i;
            break;
        }
    }
    if (output_index < 0) {
        for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
            if (!outputs[static_cast<std::size_t>(i)].IsTensor()) {
                continue;
            }
            auto info = outputs[static_cast<std::size_t>(i)].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                || info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
                output_index = i;
                break;
            }
        }
    }
    if (output_index < 0) {
        error = "encoder hidden state not found";
        return false;
    }

    auto &hidden_tensor = outputs[static_cast<std::size_t>(output_index)];
    auto hidden_info = hidden_tensor.GetTensorTypeAndShapeInfo();
    encoder_hidden_shape = hidden_info.GetShape();
    const std::size_t hidden_count = hidden_info.GetElementCount();
    if (hidden_count == 0) {
        error = "encoder hidden state empty";
        return false;
    }

    encoder_hidden_states.resize(hidden_count);
    if (hidden_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *src = hidden_tensor.GetTensorData<float>();
        std::memcpy(encoder_hidden_states.data(), src, hidden_count * sizeof(float));
    } else if (hidden_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double *src = hidden_tensor.GetTensorData<double>();
        for (std::size_t i = 0; i < hidden_count; ++i) {
            encoder_hidden_states[i] = static_cast<float>(src[i]);
        }
    } else {
        error = "unsupported encoder hidden dtype";
        return false;
    }

    return true;
}

bool run_decoder_step_locked(
    bool use_past,
    int64_t token,
    const std::vector<float> &encoder_hidden_states,
    const std::vector<int64_t> &encoder_hidden_shape,
    const std::vector<int64_t> &encoder_attention_mask,
    std::vector<std::vector<float>> &past_values,
    std::vector<std::vector<int64_t>> &past_shapes,
    int64_t &next_token,
    std::string &error) {
    auto &meta = use_past ? g_state.nllb.decoder_with_past : g_state.nllb.decoder;
    if (!meta.session) {
        error = use_past ? "decoder_with_past unavailable" : "decoder unavailable";
        return false;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<Ort::Value> inputs;
    inputs.reserve(meta.input_names.size());
    std::vector<std::vector<int64_t>> int64_buffers;
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<uint8_t>> bool_buffers;
    int64_buffers.reserve(meta.input_names.size());
    float_buffers.reserve(meta.input_names.size());
    bool_buffers.reserve(meta.input_names.size());

    std::vector<int64_t> token_ids = {token};
    std::vector<int64_t> token_shape = {1, 1};
    std::size_t past_in_index = 0;
    int64_t decoder_mask_len = static_cast<int64_t>(token_ids.size());
    if (use_past && !past_shapes.empty()) {
        const auto &shape0 = past_shapes.front();
        if (shape0.size() >= 3) {
            const int64_t past_seq = shape0[shape0.size() - 2];
            if (past_seq > 0) {
                decoder_mask_len = past_seq + static_cast<int64_t>(token_ids.size());
            }
        }
    }

    for (std::size_t i = 0; i < meta.input_names.size(); ++i) {
        const std::string name = to_lower_ascii(meta.input_name_storage[i]);
        const auto type = meta.input_types[i];
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            std::vector<int64_t> shape = {1, 1};
            std::vector<int64_t> values;
            if (contains_ci(name, "input_ids")) {
                values = token_ids;
                shape = token_shape;
            } else if (contains_ci(name, "decoder_attention_mask")
                || (contains_ci(name, "attention_mask") && contains_ci(name, "decoder"))) {
                values.assign(static_cast<std::size_t>(std::max<int64_t>(1, decoder_mask_len)), 1);
                shape = {1, static_cast<int64_t>(values.size())};
            } else if (contains_ci(name, "encoder_attention_mask")
                || (contains_ci(name, "attention_mask") && !contains_ci(name, "decoder"))) {
                values = encoder_attention_mask;
                shape = {1, static_cast<int64_t>(encoder_attention_mask.size())};
            } else {
                shape = normalize_shape(meta.input_shapes[i]);
                std::size_t n = safe_element_count(shape);
                if (n == 0) {
                    shape = {1};
                    n = 1;
                }
                values.assign(n, 0);
            }
            int64_buffers.push_back(std::move(values));
            auto &stored = int64_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            std::vector<int64_t> shape = normalize_shape(meta.input_shapes[i]);
            std::size_t n = safe_element_count(shape);
            if (n == 0) {
                shape = {1};
                n = 1;
            }
            bool_buffers.emplace_back(n, 0);
            auto &stored = bool_buffers.back();
            if (contains_ci(name, "use_cache")) {
                std::fill(stored.begin(), stored.end(), use_past ? 1 : 0);
            }
            inputs.push_back(Ort::Value::CreateTensor<bool>(
                mem_info,
                reinterpret_cast<bool *>(stored.data()),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else {
            std::vector<int64_t> shape;
            std::vector<float> values;
            if (contains_ci(name, "encoder_hidden_states")) {
                shape = encoder_hidden_shape;
                values = encoder_hidden_states;
            } else if (use_past && (contains_ci(name, "past_key_values") || contains_ci(name, "past"))) {
                if (past_in_index < past_values.size() && past_in_index < past_shapes.size()) {
                    shape = past_shapes[past_in_index];
                    values = past_values[past_in_index];
                    past_in_index++;
                } else {
                    shape = normalize_shape(meta.input_shapes[i]);
                    std::size_t n = safe_element_count(shape);
                    if (n == 0) {
                        shape = {1};
                        n = 1;
                    }
                    values.assign(n, 0.0f);
                }
            } else {
                shape = normalize_shape(meta.input_shapes[i]);
                std::size_t n = safe_element_count(shape);
                if (n == 0) {
                    shape = {1};
                    n = 1;
                }
                values.assign(n, 0.0f);
            }

            float_buffers.push_back(std::move(values));
            auto &stored = float_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        }
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = meta.session->Run(
            Ort::RunOptions{nullptr},
            meta.input_names.data(),
            inputs.data(),
            inputs.size(),
            meta.output_names.data(),
            meta.output_names.size()
        );
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    }

    const int logits_idx = find_logits_output_index(meta, outputs);
    if (logits_idx < 0) {
        error = "decoder logits not found";
        return false;
    }
    next_token = argmax_last_logits(outputs[static_cast<std::size_t>(logits_idx)]);

    std::vector<std::vector<float>> next_past_values;
    std::vector<std::vector<int64_t>> next_past_shapes;
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        if (i == logits_idx || !outputs[static_cast<std::size_t>(i)].IsTensor()) {
            continue;
        }
        auto info = outputs[static_cast<std::size_t>(i)].GetTensorTypeAndShapeInfo();
        const auto elem_type = info.GetElementType();
        const std::size_t n = info.GetElementCount();
        if (n == 0 || (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)) {
            continue;
        }

        std::vector<float> values(n);
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float *src = outputs[static_cast<std::size_t>(i)].GetTensorData<float>();
            std::memcpy(values.data(), src, n * sizeof(float));
        } else {
            const double *src = outputs[static_cast<std::size_t>(i)].GetTensorData<double>();
            for (std::size_t j = 0; j < n; ++j) {
                values[j] = static_cast<float>(src[j]);
            }
        }
        next_past_values.push_back(std::move(values));
        next_past_shapes.push_back(info.GetShape());
    }

    if (!next_past_values.empty()) {
        past_values = std::move(next_past_values);
        past_shapes = std::move(next_past_shapes);
    }
    return true;
}

bool run_decoder_full_step_locked(
    const std::vector<int64_t> &decoder_ids,
    const std::vector<float> &encoder_hidden_states,
    const std::vector<int64_t> &encoder_hidden_shape,
    const std::vector<int64_t> &encoder_attention_mask,
    int64_t &next_token,
    std::string &error) {
    auto &meta = g_state.nllb.decoder;
    if (!meta.session) {
        error = "decoder unavailable";
        return false;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<Ort::Value> inputs;
    inputs.reserve(meta.input_names.size());
    std::vector<std::vector<int64_t>> int64_buffers;
    std::vector<std::vector<float>> float_buffers;
    int64_buffers.reserve(meta.input_names.size());
    float_buffers.reserve(meta.input_names.size());

    for (std::size_t i = 0; i < meta.input_names.size(); ++i) {
        const std::string name = to_lower_ascii(meta.input_name_storage[i]);
        const auto type = meta.input_types[i];
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            std::vector<int64_t> shape = {1, 1};
            std::vector<int64_t> values;
            if (contains_ci(name, "input_ids")) {
                values = decoder_ids;
                shape = {1, static_cast<int64_t>(decoder_ids.size())};
            } else if (contains_ci(name, "decoder_attention_mask")
                || (contains_ci(name, "attention_mask") && contains_ci(name, "decoder"))) {
                values.assign(decoder_ids.size(), 1);
                shape = {1, static_cast<int64_t>(decoder_ids.size())};
            } else if (contains_ci(name, "encoder_attention_mask")
                || (contains_ci(name, "attention_mask") && !contains_ci(name, "decoder"))) {
                values = encoder_attention_mask;
                shape = {1, static_cast<int64_t>(encoder_attention_mask.size())};
            } else {
                shape = normalize_shape(meta.input_shapes[i]);
                std::size_t n = safe_element_count(shape);
                if (n == 0) {
                    shape = {1};
                    n = 1;
                }
                values.assign(n, 0);
            }
            int64_buffers.push_back(std::move(values));
            auto &stored = int64_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else {
            std::vector<int64_t> shape;
            std::vector<float> values;
            if (contains_ci(name, "encoder_hidden_states")) {
                shape = encoder_hidden_shape;
                values = encoder_hidden_states;
            } else {
                shape = normalize_shape(meta.input_shapes[i]);
                std::size_t n = safe_element_count(shape);
                if (n == 0) {
                    shape = {1};
                    n = 1;
                }
                values.assign(n, 0.0f);
            }
            float_buffers.push_back(std::move(values));
            auto &stored = float_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        }
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = meta.session->Run(
            Ort::RunOptions{nullptr},
            meta.input_names.data(),
            inputs.data(),
            inputs.size(),
            meta.output_names.data(),
            meta.output_names.size()
        );
    } catch (const Ort::Exception &ex) {
        error = ex.what();
        return false;
    }

    const int logits_idx = find_logits_output_index(meta, outputs);
    if (logits_idx < 0) {
        error = "decoder logits not found";
        return false;
    }
    next_token = argmax_last_logits(outputs[static_cast<std::size_t>(logits_idx)]);
    return true;
}

bool nllb_translate_locked(
    const std::string &text,
    Lang src,
    Lang tgt,
    ThermalMode mode,
    std::string &translation,
    int &tokens_in,
    int &tokens_out,
    int &max_new_tokens,
    std::string &error) {
    if (src == tgt) {
        translation = text;
        tokens_in = 1;
        tokens_out = 1;
        max_new_tokens = 1;
        return true;
    }
    if (!ensure_nllb_loaded_locked(mode, src, tgt)) {
        error = "NLLB unavailable";
        return false;
    }

    auto &nllb = g_state.nllb;

    std::vector<int64_t> pieces = sentencepiece_encode(nllb.source_tokenizer, text);
    tokens_in = std::max(1, static_cast<int>(pieces.size()));
    max_new_tokens = compute_max_new_tokens(tokens_in, mode);

    std::vector<int64_t> encoder_input_ids;
    if (nllb.config.uses_lang_ids) {
        const int src_lang_id = nllb_language_id(nllb.config, src);
        if (src_lang_id < 0) {
            error = "source language ID missing";
            return false;
        }
        encoder_input_ids.reserve(pieces.size() + 2);
        encoder_input_ids.push_back(src_lang_id);
        encoder_input_ids.insert(encoder_input_ids.end(), pieces.begin(), pieces.end());
    } else {
        // Marian models are direction-specific; no language ID prefix.
        encoder_input_ids.reserve(pieces.size() + 1);
        encoder_input_ids.insert(encoder_input_ids.end(), pieces.begin(), pieces.end());
    }
    encoder_input_ids.push_back(nllb.config.eos_id);

    std::vector<float> encoder_hidden_states;
    std::vector<int64_t> encoder_hidden_shape;
    if (!run_encoder_locked(encoder_input_ids, encoder_hidden_states, encoder_hidden_shape, error)) {
        return false;
    }

    std::vector<int64_t> encoder_attention_mask(encoder_input_ids.size(), 1);
    std::vector<std::vector<float>> past_values;
    std::vector<std::vector<int64_t>> past_shapes;
    std::vector<int64_t> output_tokens;
    output_tokens.reserve(static_cast<std::size_t>(max_new_tokens));

    int64_t decoder_start = 0;
    if (nllb.config.uses_lang_ids) {
        const int tgt_lang_id = nllb_language_id(nllb.config, tgt);
        if (tgt_lang_id < 0) {
            error = "target language ID missing";
            return false;
        }
        decoder_start = static_cast<int64_t>(tgt_lang_id);
    } else {
        if (nllb.config.decoder_start_id < 0) {
            error = "decoder_start_id missing";
            return false;
        }
        decoder_start = static_cast<int64_t>(nllb.config.decoder_start_id);
    }

    int64_t next = 0;
    if (!nllb.config.uses_lang_ids) {
        // Marian-compatible fallback: greedy full-decoder loop (no KV cache).
        std::vector<int64_t> decoder_ids = {decoder_start};
        while (static_cast<int>(output_tokens.size()) < max_new_tokens) {
            if (!run_decoder_full_step_locked(
                decoder_ids,
                encoder_hidden_states,
                encoder_hidden_shape,
                encoder_attention_mask,
                next,
                error)) {
                return false;
            }
            if (next == nllb.config.eos_id) {
                break;
            }
            output_tokens.push_back(next);
            decoder_ids.push_back(next);
        }
    } else {
        if (!run_decoder_step_locked(
            false,
            decoder_start,
            encoder_hidden_states,
            encoder_hidden_shape,
            encoder_attention_mask,
            past_values,
            past_shapes,
            next,
            error)) {
            return false;
        }

        if (next != nllb.config.eos_id) {
            output_tokens.push_back(next);
        }

        while (static_cast<int>(output_tokens.size()) < max_new_tokens && next != nllb.config.eos_id) {
            if (!run_decoder_step_locked(
                true,
                next,
                encoder_hidden_states,
                encoder_hidden_shape,
                encoder_attention_mask,
                past_values,
                past_shapes,
                next,
                error)) {
                return false;
            }
            if (next == nllb.config.eos_id) {
                break;
            }
            output_tokens.push_back(next);
        }
    }

    tokens_out = static_cast<int>(output_tokens.size());
    translation = sentencepiece_decode(nllb.target_tokenizer, output_tokens, nllb.config);
    if (translation.empty()) {
        translation = text;
    }
    return true;
}

std::vector<int64_t> resolve_input_shape(
    const std::vector<int64_t> &base_shape,
    bool is_audio,
    int frame_samples) {
    std::vector<int64_t> shape = normalize_shape(base_shape);
    if (is_audio) {
        if (shape.size() == 1) {
            if (base_shape.empty() || base_shape[0] <= 0) {
                shape[0] = frame_samples;
            }
        } else {
            const size_t last = shape.size() - 1;
            if (base_shape.size() <= last || base_shape[last] <= 0) {
                shape[last] = frame_samples;
            }
        }
    }
    return shape;
}

float run_silero_frame_locked(
    const float *frame_audio,
    int frame_len,
    int sample_rate,
    std::vector<std::vector<float>> &state_buffers,
    std::vector<float> &context_buffer) {
    auto &vad = g_state.vad;
    if (!vad.session) {
        return 0.0f;
    }

    std::vector<int> state_lookup(vad.input_types.size(), -1);
    for (size_t i = 0; i < vad.state_input_indices.size(); ++i) {
        int input_idx = vad.state_input_indices[i];
        if (input_idx >= 0 && static_cast<size_t>(input_idx) < state_lookup.size()) {
            state_lookup[input_idx] = static_cast<int>(i);
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<Ort::Value> inputs;
    inputs.reserve(vad.input_names.size());
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<int64_t>> int64_buffers;
    float_buffers.reserve(vad.input_names.size());
    int64_buffers.reserve(vad.input_names.size());

    const bool use_wrapper_mode = !vad.state_input_indices.empty();
    const int num_samples = (sample_rate == 8000) ? 256 : 512;
    const int context_size = (sample_rate == 8000) ? 32 : 64;
    std::vector<float> wrapper_input;
    if (use_wrapper_mode) {
        if (context_buffer.size() != static_cast<size_t>(context_size)) {
            context_buffer.assign(static_cast<size_t>(context_size), 0.0f);
        }

        std::vector<float> chunk(static_cast<size_t>(num_samples), 0.0f);
        const size_t copy_count = std::min(chunk.size(), static_cast<size_t>(std::max(0, frame_len)));
        if (copy_count > 0) {
            std::copy(frame_audio, frame_audio + copy_count, chunk.begin());
        }

        wrapper_input.reserve(static_cast<size_t>(context_size + num_samples));
        wrapper_input.insert(wrapper_input.end(), context_buffer.begin(), context_buffer.end());
        wrapper_input.insert(wrapper_input.end(), chunk.begin(), chunk.end());
        context_buffer.assign(chunk.end() - context_size, chunk.end());
    }

    for (size_t i = 0; i < vad.input_names.size(); ++i) {
        const auto elem_type = vad.input_types[i];
        const bool is_audio = static_cast<int>(i) == vad.audio_input_idx;
        const bool is_sr = static_cast<int>(i) == vad.sr_input_idx;
        std::vector<int64_t> shape = resolve_input_shape(vad.input_shapes[i], is_audio, frame_len);
        size_t tensor_size = 0;

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::vector<float> buffer;
            if (is_audio) {
                if (use_wrapper_mode) {
                    buffer = wrapper_input;
                    shape = {1, static_cast<int64_t>(buffer.size())};
                } else {
                    tensor_size = safe_element_count(shape);
                    if (tensor_size == 0) {
                        tensor_size = static_cast<size_t>(std::max(1, frame_len));
                        shape = {1, static_cast<int64_t>(tensor_size)};
                    }
                    buffer.assign(tensor_size, 0.0f);
                    const size_t copy_count = std::min(buffer.size(), static_cast<size_t>(std::max(0, frame_len)));
                    if (copy_count > 0) {
                        std::copy(frame_audio, frame_audio + copy_count, buffer.begin());
                    }
                }
            } else {
                tensor_size = safe_element_count(normalize_shape(shape));
                if (tensor_size == 0) {
                    tensor_size = 1;
                    shape = {1};
                }
                buffer.assign(tensor_size, 0.0f);
                const int state_idx = state_lookup[i];
                if (state_idx >= 0 && static_cast<size_t>(state_idx) < state_buffers.size()) {
                    auto &state = state_buffers[static_cast<size_t>(state_idx)];
                    const size_t copy_count = std::min(buffer.size(), state.size());
                    if (copy_count > 0) {
                        std::copy(state.begin(), state.begin() + copy_count, buffer.begin());
                    }
                }
            }
            float_buffers.push_back(std::move(buffer));
            auto &stored = float_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            tensor_size = safe_element_count(normalize_shape(shape));
            if (tensor_size == 0) {
                tensor_size = 1;
                shape = {1};
            }
            std::vector<int64_t> buffer(tensor_size, 0);
            if (is_sr) {
                std::fill(buffer.begin(), buffer.end(), static_cast<int64_t>(sample_rate));
            }
            int64_buffers.push_back(std::move(buffer));
            auto &stored = int64_buffers.back();
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem_info,
                stored.data(),
                stored.size(),
                shape.data(),
                shape.size()
            ));
        } else {
            std::vector<float> buffer(1, 0.0f);
            float_buffers.push_back(std::move(buffer));
            auto &stored = float_buffers.back();
            std::vector<int64_t> fallback_shape = {1};
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                stored.data(),
                stored.size(),
                fallback_shape.data(),
                fallback_shape.size()
            ));
        }
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = vad.session->Run(
            Ort::RunOptions{nullptr},
            vad.input_names.data(),
            inputs.data(),
            inputs.size(),
            vad.output_names.data(),
            vad.output_names.size()
        );
    } catch (const Ort::Exception &ex) {
        logi("Silero inference error: " + std::string(ex.what()));
        return 0.0f;
    } catch (const std::exception &ex) {
        logi("Silero inference error: " + std::string(ex.what()));
        return 0.0f;
    }

    if (!outputs.empty() && outputs[0].IsTensor()) {
        for (size_t i = 0; i < vad.state_output_indices.size() && i < state_buffers.size(); ++i) {
            int out_idx = vad.state_output_indices[i];
            if (out_idx < 0 || static_cast<size_t>(out_idx) >= outputs.size()) {
                continue;
            }
            auto &state_output = outputs[static_cast<size_t>(out_idx)];
            if (!state_output.IsTensor()) {
                continue;
            }
            auto info = state_output.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                continue;
            }
            size_t count = info.GetElementCount();
            const float *data = state_output.GetTensorData<float>();
            if (data && count > 0) {
                state_buffers[i].assign(data, data + count);
            }
        }

        auto prob_info = outputs[0].GetTensorTypeAndShapeInfo();
        const size_t prob_count = prob_info.GetElementCount();
        if (prob_count > 0 && prob_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float *prob_data = outputs[0].GetTensorData<float>();
            return prob_data ? prob_data[0] : 0.0f;
        }
        if (prob_count > 0 && prob_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            const double *prob_data = outputs[0].GetTensorData<double>();
            return prob_data ? static_cast<float>(prob_data[0]) : 0.0f;
        }
    }
    return 0.0f;
}

std::vector<int> vad_trim_and_split(
    const float *audio,
    int len,
    int sample_rate,
    int split_gap_ms) {
    std::vector<int> segments;
    if (!audio || len <= 0 || sample_rate <= 0) {
        return segments;
    }

    if (!g_state.vad.loaded || !g_state.vad.session) {
        segments.push_back(0);
        segments.push_back(len);
        return segments;
    }

    const int frame_samples = std::max(1, sample_rate * 30 / 1000);
    const float speech_threshold = 0.5f;

    std::vector<std::vector<float>> state_buffers;
    state_buffers.reserve(g_state.vad.state_input_shapes.size());
    for (const auto &shape : g_state.vad.state_input_shapes) {
        const size_t count = std::max(static_cast<size_t>(1), safe_element_count(normalize_shape(shape)));
        state_buffers.emplace_back(count, 0.0f);
    }
    std::vector<float> context_buffer;

    std::vector<std::pair<int, int>> raw_segments;
    bool in_speech = false;
    int speech_start = 0;
    int last_speech_end = 0;

    for (int frame_start = 0; frame_start < len; frame_start += frame_samples) {
        const int frame_end = std::min(frame_start + frame_samples, len);
        const float prob = run_silero_frame_locked(
            audio + frame_start,
            frame_end - frame_start,
            sample_rate,
            state_buffers,
            context_buffer
        );
        const bool is_speech = prob >= speech_threshold;

        if (is_speech) {
            if (!in_speech) {
                in_speech = true;
                speech_start = frame_start;
                logi("VAD start sample=" + std::to_string(speech_start));
            }
            last_speech_end = frame_end;
        } else if (in_speech) {
            raw_segments.emplace_back(speech_start, last_speech_end);
            logi("VAD end sample=" + std::to_string(last_speech_end));
            in_speech = false;
        }
    }

    if (in_speech) {
        raw_segments.emplace_back(speech_start, last_speech_end > 0 ? last_speech_end : len);
        logi("VAD end sample=" + std::to_string(last_speech_end > 0 ? last_speech_end : len));
    }

    if (raw_segments.empty()) {
        segments.push_back(0);
        segments.push_back(len);
        logi("VAD merged segments=[(0," + std::to_string(len) + ")]");
        logi("VAD final segment count=1");
        return segments;
    }

    const int split_gap_samples = sample_rate * split_gap_ms / 1000;
    std::vector<std::pair<int, int>> merged;
    merged.push_back(raw_segments.front());
    for (size_t i = 1; i < raw_segments.size(); ++i) {
        auto &last = merged.back();
        const auto &curr = raw_segments[i];
        int gap = curr.first - last.second;
        if (gap <= split_gap_samples) {
            last.second = curr.second;
        } else {
            merged.push_back(curr);
        }
    }

    for (const auto &seg : merged) {
        segments.push_back(seg.first);
        segments.push_back(seg.second);
    }
    logi("VAD merged segments=" + segments_to_string(merged));
    logi("VAD final segment count=" + std::to_string(merged.size()));
    return segments;
}

void log_ggml_cpu_features_once_locked() {
    if (g_state.ggml_features_logged) {
        return;
    }
    g_state.ggml_features_logged = true;

    const int neon = ggml_cpu_has_neon();
    const int fp16 = ggml_cpu_has_fp16_va();
    const int dotprod = ggml_cpu_has_dotprod();

    __android_log_print(
        ANDROID_LOG_INFO,
        kTag,
        "GGML CPU FEATURES: NEON=%d FP16=%d DOTPROD=%d",
        neon,
        fp16,
        dotprod
    );
    __android_log_print(ANDROID_LOG_INFO, kTag, "GGML CPU FEATURES:");
    __android_log_print(ANDROID_LOG_INFO, kTag, "NEON=%d", neon);
    __android_log_print(ANDROID_LOG_INFO, kTag, "FP16=%d", fp16);
    __android_log_print(ANDROID_LOG_INFO, kTag, "DOTPROD=%d", dotprod);
}

std::string whisper_transcribe_once(
    const float *audio,
    int len,
    int sample_rate,
    Lang lang,
    const std::string &model_path,
    ThermalMode thermal_mode) {
    if (!audio || len <= 0 || sample_rate <= 0) {
        logi("Whisper load start");
        logi("Whisper infer start");
        logi("Whisper infer done text=");
        logi("Whisper context freed");
        return "";
    }

    logi("Whisper load start");

    whisper_context_params ctx_params = whisper_context_default_params();
    ctx_params.use_gpu = false;
    ctx_params.flash_attn = false;

    whisper_context *ctx = whisper_init_from_file_with_params(model_path.c_str(), ctx_params);
    if (ctx == nullptr) {
        logi("Whisper infer start");
        logi("Whisper infer done text=");
        logi("Whisper context freed");
        return "";
    }

    std::unique_ptr<whisper_context, void(*)(whisper_context *)> ctx_guard(ctx, whisper_free);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = whisper_threads_for_mode(thermal_mode);
    params.translate = false;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.token_timestamps = false;
    params.language = whisper_lang_code(lang);
    params.detect_language = false;
    params.greedy.best_of = 1;
    params.beam_search.beam_size = 1;
    params.offset_ms = 0;
    params.duration_ms = 0;

    logi("Whisper infer start");
    std::string text;
    const int ret = whisper_full(ctx, params, audio, len);
    if (ret == 0) {
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char *seg = whisper_full_get_segment_text(ctx, i);
            if (seg != nullptr) {
                text += seg;
            }
        }

        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.erase(text.begin());
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }
    }

    logi("Whisper infer done text=" + text);
    ctx_guard.reset();
    logi("Whisper context freed");
    return text;
}

std::vector<int16_t> synth_sine_pcm(const std::string &text, int sample_rate) {
    const int words = std::max(1, static_cast<int>(std::count(text.begin(), text.end(), ' ') + 1));
    const int duration_ms = std::min(2200, std::max(200, words * 180));
    const int sample_count = sample_rate * duration_ms / 1000;

    std::vector<int16_t> out(sample_count);
    const float freq = 220.0f;
    const float two_pi = 6.283185307f;
    for (int i = 0; i < sample_count; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        float env = std::min(1.0f, i / 2000.0f);
        float value = std::sin(two_pi * freq * t) * 0.20f * env;
        out[i] = static_cast<int16_t>(value * 32767.0f);
    }
    return out;
}

jstring make_jstring(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

} // namespace

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_init(
    JNIEnv *env,
    jobject,
    jstring modelsDir) {
    (void) env;
    if (modelsDir == nullptr) {
        return JNI_FALSE;
    }

    const char *dir = env->GetStringUTFChars(modelsDir, nullptr);
    std::string sanity_model_path;
    bool env_ready = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_state = GlobalState{};
        g_state.models_dir = dir;
        env_ready = ensure_ort_env_locked();
        sanity_model_path = g_state.models_dir + "/sanity/mul_1.onnx";
    }
    env->ReleaseStringUTFChars(modelsDir, dir);

    if (!env_ready) {
        logi("ONNX Runtime env init failed");
        return JNI_FALSE;
    }

    std::string sanity_error;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!ortSanityCheck(*g_state.ort_env, sanity_model_path, sanity_error)) {
            logi("ONNX Runtime sanity check failed: " + sanity_error);
            return JNI_FALSE;
        }
    }
    logi("ONNX Runtime sanity check passed");
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_setThermalMode(
    JNIEnv *,
    jobject,
    jint modeValue) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_state.thermal_mode = static_cast<ThermalMode>(modeValue);
    if (g_state.thermal_mode == ThermalMode::CRITICAL) {
        unload_nllb_locked();
        return;
    }

    if (g_state.nllb.loaded) {
        const int needed_threads = mode_threads(g_state.thermal_mode);
        if (g_state.nllb.threads != needed_threads) {
            // Recreate lazily on next translation request.
            unload_nllb_locked();
        }
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_onTrimMemory(
    JNIEnv *,
    jobject,
    jint level) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (level >= kTrimMemoryRunningCritical) {
        unload_nllb_locked();
        unload_piper_locked();
    }
    if (level >= kTrimMemoryComplete) {
        unload_silero_locked();
    }
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_vadTrimAndSplit(
    JNIEnv *env,
    jobject,
    jfloatArray pcmF32,
    jint sampleRate,
    jint splitGapMs) {
    if (pcmF32 == nullptr) {
        return env->NewIntArray(0);
    }

    jsize len = env->GetArrayLength(pcmF32);
    std::vector<float> buffer(static_cast<size_t>(len));
    env->GetFloatArrayRegion(pcmF32, 0, len, buffer.data());

    std::vector<int> segments;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        maybe_idle_unload_locked();
        std::string vad_error;
        if (!ensure_silero_loaded_locked(vad_error)) {
            logi("Silero VAD load failed: " + vad_error);
            segments = {0, static_cast<int>(buffer.size())};
        } else {
            segments = vad_trim_and_split(buffer.data(), static_cast<int>(buffer.size()), sampleRate, splitGapMs);
        }
    }

    jintArray out = env->NewIntArray(static_cast<jsize>(segments.size()));
    if (!segments.empty()) {
        env->SetIntArrayRegion(out, 0, static_cast<jsize>(segments.size()), segments.data());
    }
    return out;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_whisperTranscribeOnce(
    JNIEnv *env,
    jobject,
    jfloatArray pcmF32,
    jint sampleRate,
    jint langValue) {
    if (pcmF32 == nullptr) {
        return make_jstring(env, "");
    }

    jsize len = env->GetArrayLength(pcmF32);
    std::vector<float> buffer(static_cast<size_t>(len));
    env->GetFloatArrayRegion(pcmF32, 0, len, buffer.data());

    std::string model_path;
    ThermalMode thermal_mode = ThermalMode::NORMAL;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        maybe_idle_unload_locked();
        log_ggml_cpu_features_once_locked();
        model_path = g_state.models_dir + "/whisper/ggml-tiny.bin";
        thermal_mode = g_state.thermal_mode;
    }
    std::string text = whisper_transcribe_once(
        buffer.data(),
        static_cast<int>(buffer.size()),
        sampleRate,
        static_cast<Lang>(langValue),
        model_path,
        thermal_mode
    );

    return make_jstring(env, text);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_nllbEnsureLoaded(
    JNIEnv *,
    jobject,
    jint modeValue) {
    std::lock_guard<std::mutex> lock(g_mu);
    maybe_idle_unload_locked();
    bool ok = ensure_nllb_loaded_locked(static_cast<ThermalMode>(modeValue));
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_nllbTranslate(
    JNIEnv *env,
    jobject,
    jstring text,
    jint srcLang,
    jint tgtLang,
    jint modeValue) {
    if (text == nullptr) {
        return make_jstring(env, "");
    }

    const char *raw = env->GetStringUTFChars(text, nullptr);
    std::string input = raw == nullptr ? "" : std::string(raw);
    env->ReleaseStringUTFChars(text, raw);

    std::string output;
    int tokens_in = 0;
    int tokens_out = 0;
    int max_new_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        maybe_idle_unload_locked();

        ThermalMode mode = static_cast<ThermalMode>(modeValue);
        Lang src = static_cast<Lang>(srcLang);
        Lang tgt = static_cast<Lang>(tgtLang);

        if (mode == ThermalMode::CRITICAL) {
            unload_nllb_locked();
            output = phrase_fallback_translate_locked(input, src, tgt);
            tokens_in = rough_word_token_count(input);
            tokens_out = 0;
            max_new_tokens = 0;
        } else {
            tokens_in = rough_word_token_count(input);
            max_new_tokens = compute_max_new_tokens(tokens_in, mode);
            tokens_out = 0;
            std::string error;
            if (!nllb_translate_locked(
                input,
                src,
                tgt,
                mode,
                output,
                tokens_in,
                tokens_out,
                max_new_tokens,
                error)) {
                logi("NLLB inference failed: " + error);
                output = phrase_fallback_translate_locked(input, src, tgt);
            }

            g_state.nllb.last_input_tokens = tokens_in;
            g_state.nllb.last_max_new_tokens = max_new_tokens;
            g_state.nllb.last_used_ms = now_ms();
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    std::ostringstream metric;
    metric.setf(std::ios::fixed);
    metric.precision(2);
    metric << "NLLB tokens_in=" << tokens_in
           << " tokens_out=" << tokens_out
           << " max_new_tokens=" << max_new_tokens
           << " time=" << elapsed_sec << "s";
    logi(metric.str());

    return make_jstring(env, output);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_piperUnloadVoice(
    JNIEnv *,
    jobject) {
    std::lock_guard<std::mutex> lock(g_mu);
    unload_piper_locked();
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_piperSynthesize(
    JNIEnv *env,
    jobject,
    jint voice,
    jstring text) {
    const char *raw = (text == nullptr) ? nullptr : env->GetStringUTFChars(text, nullptr);
    std::string input = raw == nullptr ? "" : std::string(raw);
    if (raw != nullptr) {
        env->ReleaseStringUTFChars(text, raw);
    }

    std::vector<int16_t> pcm;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        maybe_idle_unload_locked();

        if (!g_state.piper.loaded || g_state.piper.loaded_voice != voice) {
            unload_piper_locked();
            g_state.piper.loaded = true;
            g_state.piper.loaded_voice = voice;
        }
        g_state.piper.last_used_ms = now_ms();

        pcm = synth_sine_pcm(input, 22050);
    }

    const jsize bytes = static_cast<jsize>(pcm.size() * sizeof(int16_t));
    jbyteArray out = env->NewByteArray(bytes);
    if (bytes > 0) {
        env->SetByteArrayRegion(
            out,
            0,
            bytes,
            reinterpret_cast<const jbyte *>(pcm.data())
        );
    }
    return out;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_armfinal_translator_nativebridge_NativeBridge_runtimeStats(
    JNIEnv *env,
    jobject) {
    std::ostringstream os;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        os << "mode=" << static_cast<int>(g_state.thermal_mode)
           << ", vad_loaded=" << (g_state.vad.loaded ? "1" : "0")
           << ", nllb_loaded=" << (g_state.nllb.loaded ? "1" : "0")
           << ", nllb_threads=" << g_state.nllb.threads
           << ", nllb_input_tokens=" << g_state.nllb.last_input_tokens
           << ", nllb_max_new_tokens=" << g_state.nllb.last_max_new_tokens
           << ", piper_loaded=" << (g_state.piper.loaded ? "1" : "0")
           << ", piper_voice=" << g_state.piper.loaded_voice;
    }

    return make_jstring(env, os.str());
}
