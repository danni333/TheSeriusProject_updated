#include "search_engine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "../io/gguf_loader.h"
#include "../io/safetensors.h"
#include "metadata_manager.h"

// windows.h (via io/mmap.h) defines min/max macros that break std::min/std::max.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace serius::hub {

namespace {

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool contains_case_insensitive(const std::string &haystack, const std::string &needle)
{
    return to_lower(haystack).find(to_lower(needle)) != std::string::npos;
}

std::set<std::string> word_set(const std::string &text)
{
    std::set<std::string> out;
    std::istringstream iss(to_lower(text));
    std::string w;
    while (iss >> w) {
        // strip simple punctuation
        w.erase(std::remove_if(w.begin(), w.end(), [](unsigned char c) { return std::ispunct(c) != 0; }), w.end());
        if (!w.empty()) out.insert(w);
    }
    return out;
}

/// Jaccard similarity over lowercase word sets — cheap lexical signal that
/// complements the hashed embedding for ranking.
float token_jaccard(const std::string &a, const std::string &b)
{
    const auto sa = word_set(a), sb = word_set(b);
    if (sa.empty() || sb.empty()) return 0.0f;
    size_t inter = 0;
    for (const auto &w : sa)
        if (sb.count(w)) ++inter;
    const size_t uni = sa.size() + sb.size() - inter;
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

/// Path of the analytics sidecar file (search query counters).
std::string analytics_path(const std::string &db_path)
{
    return db_path + ".analytics.json";
}

}  // namespace

SearchEngine::SearchEngine(const std::string &db_path)
    : db_path_(db_path), embedding_manager_(std::make_unique<EmbeddingManager>())
{
}

SearchEngine::~SearchEngine() {}

bool SearchEngine::initialize()
{
    // The embedding manager is best-effort: hash-based embeddings work without
    // a model file, so initialization never fails hard.
    embedding_manager_->initialize("models/embedding_model.bin");
    return true;
}

bool SearchEngine::create_tables()
{
    return true;  // file-based storage needs no schema
}

std::vector<SearchResult> SearchEngine::search_models(const SearchFilter &filter, int limit, int offset)
{
    // Pull a generous candidate set from the metadata store, then apply the
    // SearchFilter fields the MetadataManager API doesn't cover.
    constexpr int kCandidateCap = 10000;
    auto candidates =
        MetadataManager(db_path_).search_models(filter.query, filter.tags, filter.author, kCandidateCap, 0);

    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    for (const auto &m : candidates) {
        if (!filter.model_type.empty() && m.model_type != filter.model_type) continue;
        if (!filter.license.empty() && m.license != filter.license) continue;
        if (m.rating < filter.min_rating) continue;
        if (m.download_count < filter.min_downloads) continue;
        if (!filter.date_after.empty() && m.created_at < filter.date_after) continue;
        if (!filter.date_before.empty() && m.created_at > filter.date_before) continue;

        SearchResult r;
        r.model_id = m.id;
        r.name = m.name;
        r.description = m.description;
        r.author = m.author;
        r.matched_tags.clear();
        for (const auto &t : m.tags) {
            if (std::find(filter.tags.begin(), filter.tags.end(), t) != filter.tags.end()) {
                r.matched_tags.push_back(t);
            }
        }
        r.highlighted_text = highlight_text(m.name + " — " + m.description, filter.query);
        r.score = static_cast<float>(m.download_count) + m.rating * 10.0f;
        results.push_back(std::move(r));
    }

    // v1 ranking: download_count + rating blended into score at insertion.
    std::sort(results.begin(), results.end(),
              [](const SearchResult &a, const SearchResult &b) { return a.score > b.score; });

    if (offset >= static_cast<int>(results.size())) return {};
    const size_t begin = static_cast<size_t>(offset);
    const size_t end = std::min(results.size(), begin + static_cast<size_t>(limit));
    return {results.begin() + begin, results.begin() + end};
}

std::vector<SearchResult> SearchEngine::semantic_search(const std::string &query, const std::vector<std::string> &tags,
                                                        int limit, int offset)
{
    const auto query_emb = compute_query_embedding(query);
    if (query_emb.empty()) return {};

    const auto all = MetadataManager(db_path_).list_all_models();
    if (all.empty()) return {};

    std::vector<std::pair<float, SearchResult>> scored;
    for (const auto &m : all) {
        if (!m.is_public) continue;
        if (!tags.empty()) {
            bool any = false;
            for (const auto &t : tags) {
                if (std::find(m.tags.begin(), m.tags.end(), t) != m.tags.end()) {
                    any = true;
                    break;
                }
            }
            if (!any) continue;
        }
        const auto doc_text = m.name + " " + m.description;
        const auto doc_emb = embedding_manager_->compute_document_embedding(doc_text);
        const float sim = cosine_similarity(query_emb, doc_emb);
        const float score = 0.7f * sim + 0.3f * token_jaccard(query, doc_text);

        SearchResult r;
        r.model_id = m.id;
        r.name = m.name;
        r.description = m.description;
        r.author = m.author;
        r.score = score;
        r.highlighted_text = m.description;
        scored.emplace_back(sim, std::move(r));
    }

    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<SearchResult> out;
    const size_t begin = std::min(scored.size(), static_cast<size_t>(offset));
    const size_t end = std::min(scored.size(), begin + static_cast<size_t>(limit));
    for (size_t i = begin; i < end; ++i) out.push_back(std::move(scored[i].second));
    return out;
}

std::vector<SearchResult> SearchEngine::find_similar_models(const std::string &model_id, int limit)
{
    const auto all = MetadataManager(db_path_).list_all_models();
    const ModelMetadata *target = nullptr;
    for (const auto &m : all) {
        if (m.id == model_id) {
            target = &m;
            break;
        }
    }
    if (target == nullptr) return {};

    const auto target_emb = embedding_manager_->compute_document_embedding(target->name + " " + target->description);
    if (target_emb.empty()) return {};

    std::vector<std::pair<float, SearchResult>> scored;
    for (const auto &m : all) {
        if (m.id == model_id) continue;
        if (!m.is_public) continue;
        const auto doc_text = m.name + " " + m.description;
        const auto emb = embedding_manager_->compute_document_embedding(doc_text);
        const float sim = cosine_similarity(target_emb, emb);
        const float score = 0.7f * sim + 0.3f * token_jaccard(target->name + " " + target->description, doc_text);
        SearchResult r;
        r.model_id = m.id;
        r.name = m.name;
        r.description = m.description;
        r.author = m.author;
        r.score = score;
        r.highlighted_text = m.description;
        scored.emplace_back(sim, std::move(r));
    }
    std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<SearchResult> out;
    const size_t n = std::min(scored.size(), static_cast<size_t>(limit));
    for (size_t i = 0; i < n; ++i) out.push_back(std::move(scored[i].second));
    return out;
}

std::vector<std::string> SearchEngine::get_suggestions(const std::string &partial_query, int limit)
{
    const std::string prefix = to_lower(partial_query);
    const auto all = MetadataManager(db_path_).list_all_models();
    std::vector<std::pair<std::string, std::string>> matches;  // (name_lower, display)
    for (const auto &m : all) {
        if (!m.is_public) continue;
        const std::string name_l = to_lower(m.name);
        const std::string id_l = to_lower(m.id);
        if (name_l.rfind(prefix, 0) == 0 || id_l.rfind(prefix, 0) == 0 || name_l.find(prefix) != std::string::npos) {
            matches.emplace_back(name_l, m.name.empty() ? m.id : m.name);
        }
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(
        std::unique(matches.begin(), matches.end(), [](const auto &a, const auto &b) { return a.second == b.second; }),
        matches.end());
    std::vector<std::string> out;
    const size_t n = std::min(matches.size(), static_cast<size_t>(limit));
    for (size_t i = 0; i < n; ++i) out.push_back(matches[i].second);
    return out;
}

std::vector<std::string> SearchEngine::get_trending_searches(int limit)
{
    return get_popular_queries(limit);
}

void SearchEngine::log_search(const std::string &query, const std::string &user_id, int results_count)
{
    const std::string path = analytics_path(db_path_);
    nlohmann::json db;
    {
        std::ifstream in(path);
        if (in.is_open()) {
            try {
                in >> db;
            }
            catch (...) {
                db = nlohmann::json::object();
            }
        }
    }
    if (!db.is_object()) db = nlohmann::json::object();
    auto &entry = db[query];
    if (!entry.is_object()) entry = nlohmann::json::object();
    entry["count"] = entry.value("count", 0) + 1;
    entry["last_user"] = user_id;
    entry["last_results"] = results_count;
    std::ofstream out(path);
    out << db.dump(2);
}

std::vector<std::string> SearchEngine::get_popular_queries(int limit)
{
    const std::string path = analytics_path(db_path_);
    std::ifstream in(path);
    if (!in.is_open()) return {};
    nlohmann::json db;
    try {
        in >> db;
    }
    catch (...) {
        return {};
    }
    if (!db.is_object()) return {};

    std::vector<std::pair<int, std::string>> counted;
    for (auto &[q, j] : db.items()) {
        counted.emplace_back(j.value("count", 0), q);
    }
    std::sort(counted.begin(), counted.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<std::string> out;
    const size_t n = std::min(counted.size(), static_cast<size_t>(limit));
    for (size_t i = 0; i < n; ++i) out.push_back(counted[i].second);
    return out;
}

std::vector<float> SearchEngine::compute_query_embedding(const std::string &query)
{
    return embedding_manager_->compute_query_embedding(query);
}

float SearchEngine::cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.empty() || a.size() != b.size()) return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return static_cast<float>(dot / std::sqrt(na * nb));
}

std::string SearchEngine::highlight_text(const std::string &text, const std::string &query)
{
    if (query.empty()) return text;
    std::string lower_text = to_lower(text);
    const std::string lower_query = to_lower(query);
    std::string out;
    size_t pos = 0;
    while (true) {
        const size_t hit = lower_text.find(lower_query, pos);
        if (hit == std::string::npos) {
            out += text.substr(pos);
            break;
        }
        out += text.substr(pos, hit - pos);
        out += "<em>" + text.substr(hit, query.size()) + "</em>";
        pos = hit + query.size();
    }
    return out;
}

bool SearchEngine::is_cache_valid(const std::string &cache_key)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    // Cache entries store results only; validity is approximated by presence
    // (TTL enforcement would need timestamps — kept simple for v1).
    return search_cache_.find(cache_key) != search_cache_.end();
}

void SearchEngine::cache_results(const std::string &cache_key, const std::vector<SearchResult> &results)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (search_cache_.size() > 256) search_cache_.clear();
    search_cache_[cache_key] = results;
}

std::vector<SearchResult> SearchEngine::get_cached_results(const std::string &cache_key)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = search_cache_.find(cache_key);
    return it == search_cache_.end() ? std::vector<SearchResult>{} : it->second;
}

std::vector<SearchResult> SearchEngine::fulltext_search(const SearchFilter &filter, int limit, int offset)
{
    return search_models(filter, limit, offset);
}

std::vector<SearchResult> SearchEngine::tag_search(const std::vector<std::string> &tags, int limit, int offset)
{
    SearchFilter f;
    f.tags = tags;
    return search_models(f, limit, offset);
}

std::vector<SearchResult> SearchEngine::author_search(const std::string &author, int limit, int offset)
{
    SearchFilter f;
    f.author = author;
    return search_models(f, limit, offset);
}

std::vector<SearchResult> SearchEngine::merge_and_rank_results(
    const std::vector<std::vector<SearchResult>> &result_sets)
{
    std::vector<SearchResult> merged;
    for (const auto &set : result_sets) {
        merged.insert(merged.end(), set.begin(), set.end());
    }
    std::sort(merged.begin(), merged.end(),
              [](const SearchResult &a, const SearchResult &b) { return a.score > b.score; });
    merged.erase(std::unique(merged.begin(), merged.end(),
                             [](const SearchResult &a, const SearchResult &b) { return a.model_id == b.model_id; }),
                 merged.end());
    return merged;
}

// EmbeddingManager implementation
EmbeddingManager::EmbeddingManager() : model_handle_(nullptr), model_loaded_(false) {}

EmbeddingManager::~EmbeddingManager()
{
    unload_model();
}

bool EmbeddingManager::initialize(const std::string &model_path)
{
    return load_model(model_path);
}

std::vector<float> EmbeddingManager::compute_embedding(const std::string &text)
{
    if (bert_ != nullptr && bert_->is_loaded()) {
        return bert_->embed(text);
    }
    if (embed_matrix_.size() > 0 && tokenizer_ != nullptr) {
        const size_t vocab = embed_matrix_.shape()[0];
        const size_t dim = embed_matrix_.shape()[1];
        std::vector<float> vec(dim, 0.0f);
        auto ids = tokenizer_->encode(text);
        if (ids.empty()) {
            return compute_with_simple_tokenizer(text);
        }
        for (auto id : ids) {
            const size_t tid = static_cast<size_t>(id) % vocab;
            for (size_t d = 0; d < dim; ++d) {
                vec[d] += embed_matrix_.at(tid, d);
            }
        }
        const float inv = 1.0f / static_cast<float>(ids.size());
        double norm = 0.0;
        for (auto &v : vec) {
            v *= inv;
            norm += static_cast<double>(v) * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (auto &v : vec) v = static_cast<float>(v / norm);
        }
        return vec;
    }
    return compute_with_simple_tokenizer(text);
}

std::vector<float> EmbeddingManager::compute_query_embedding(const std::string &text)
{
    return compute_embedding(use_task_prefixes_ ? "search_query: " + text : text);
}

std::vector<float> EmbeddingManager::compute_document_embedding(const std::string &text)
{
    return compute_embedding(use_task_prefixes_ ? "search_document: " + text : text);
}

std::vector<std::vector<float>> EmbeddingManager::compute_embeddings(const std::vector<std::string> &texts)
{
    std::vector<std::vector<float>> out;
    out.reserve(texts.size());
    for (const auto &t : texts) out.push_back(compute_embedding(t));
    return out;
}

common::TextEmbedFn EmbeddingManager::query_embed_fn()
{
    return [this](const std::string &text) { return compute_query_embedding(text); };
}

common::TextEmbedFn EmbeddingManager::document_embed_fn()
{
    return [this](const std::string &text) { return compute_document_embedding(text); };
}

bool EmbeddingManager::load_model(const std::string &model_path)
{
    unload_model();
    model_path_ = model_path;
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        if (!model_path.empty()) {
            spdlog::warn("Embedding model '{}' not found; using hash embeddings", model_path);
        }
        return true;  // hash embeddings are always available as fallback
    }

    const std::filesystem::path p(model_path);
    std::error_code ec;
    if (std::filesystem::is_regular_file(p, ec) && p.extension() == ".gguf") {
        if (try_load_gguf(model_path)) {
            model_loaded_ = true;
            spdlog::info("EmbeddingManager: loaded embedding model from '{}'", model_path);
            return true;
        }
    } else if (std::filesystem::is_directory(p, ec)) {
        if (try_load_safetensors_dir(model_path)) {
            model_loaded_ = true;
            spdlog::info("EmbeddingManager: loaded safetensors embedding matrix from '{}'", model_path);
            return true;
        }
        // Directory may itself contain a .gguf (e.g. a hub model dir).
        std::error_code ec;
        for (const auto &entry : std::filesystem::directory_iterator(p, ec)) {
            if (entry.path().extension() == ".gguf" && try_load_gguf(entry.path().string())) {
                model_loaded_ = true;
                spdlog::info("EmbeddingManager: loaded embedding model from '{}'", entry.path().string());
                return true;
            }
        }
    }

    model_loaded_ = true;  // path exists but no usable embedding matrix: hash fallback
    spdlog::warn("EmbeddingManager: no token_embd/embed_tokens tensor in '{}'; using hash embeddings", model_path);
    return true;
}

void EmbeddingManager::unload_model()
{
    model_loaded_ = false;
    model_handle_ = nullptr;
    embed_matrix_ = core::Tensor();
    tokenizer_.reset();
    bert_.reset();
    use_task_prefixes_ = false;
}

namespace {

/// Copy a [vocab, dim] (or transposed) float matrix into embed_matrix_.
/// `dim_major` selects the source layout: true means src[d*vocab + t].
bool store_embedding_matrix(core::Tensor &dst, const float *src, size_t numel, size_t vocab, size_t dim, bool dim_major)
{
    if (src == nullptr || vocab == 0 || dim == 0 || numel != vocab * dim) {
        return false;
    }
    dst = core::Tensor({vocab, dim});
    for (size_t t = 0; t < vocab; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            dst.at(t, d) = dim_major ? src[d * vocab + t] : src[t * dim + d];
        }
    }
    return true;
}

}  // namespace

bool EmbeddingManager::try_load_gguf(const std::string &path)
{
    // Embedding-specialized architectures (nomic-bert) need the full
    // encoder forward, not just the token_embd lookup.
    const std::string arch = io::gguf_architecture(path);
    if (arch == "nomic-bert" || arch == "nomic-bert-moe" || arch == "jina-bert-v2" || arch == "jina-bert-v3") {
        return try_load_nomic_gguf(path);
    }

    std::unordered_map<std::string, core::Tensor> tensors;
    try {
        tensors = io::load_all_tensors(path);
    }
    catch (const std::exception &e) {
        spdlog::warn("EmbeddingManager: GGUF '{}' rejected: {}", path, e.what());
        return false;
    }
    auto it = tensors.find("token_embd.weight");
    if (it == tensors.end()) {
        return false;
    }
    const core::Tensor &t = it->second;
    if (t.size() == 0) {
        return false;
    }

    auto tok = io::load_gguf_tokenizer(path);
    if (!tok.valid) {
        return false;
    }
    tokenizer_ = std::make_unique<serius::BPETokenizer>();
    tokenizer_->load_from_gguf(tok);
    const size_t vocab = tokenizer_->vocab_size();
    if (vocab == 0 || t.size() % vocab != 0) {
        return false;
    }
    const size_t dim = t.size() / vocab;

    // GGUF/ggml dim order: shape[0] is the contiguous dim. If the contiguous
    // dim equals vocab the flat data is dim-major and needs a transpose.
    const bool dim_major = (t.ndim() == 2 && t.shape()[0] == vocab && t.shape()[1] != vocab);
    return store_embedding_matrix(embed_matrix_, t.data(), t.size(), vocab, dim, dim_major);
}

bool EmbeddingManager::try_load_nomic_gguf(const std::string &path)
{
    std::unordered_map<std::string, core::Tensor> tensors;
    try {
        tensors = io::load_all_tensors(path);
    }
    catch (const std::exception &e) {
        spdlog::warn("EmbeddingManager: nomic-bert GGUF '{}' rejected: {}", path, e.what());
        return false;
    }
    if (tensors.empty()) {
        return false;
    }

    auto tok = io::load_gguf_tokenizer(path);
    if (!tok.valid || tok.tokens.empty()) {
        spdlog::warn("EmbeddingManager: nomic-bert GGUF '{}' has no usable tokenizer", path);
        return false;
    }

    auto emb = std::make_unique<NomicBertEmbedder>();
    emb->set_vocab(tok.tokens);
    if (!emb->load(tensors)) {
        spdlog::warn("EmbeddingManager: nomic-bert weights incomplete in '{}'", path);
        return false;
    }
    bert_ = std::move(emb);
    use_task_prefixes_ = true;  // nomic-embed models expect task prefixes
    return true;
}

bool EmbeddingManager::try_load_safetensors_dir(const std::string &dir)
{
    const std::string st_path = (std::filesystem::path(dir) / "model.safetensors").string();
    if (!std::filesystem::exists(st_path)) {
        return false;
    }

    io::SafetensorsReader reader(st_path);
    static const char *kNames[] = {
        "model.embed_tokens.weight",
        "token_embd.weight",
        "embed_tokens.weight",
        "embedding.weight",
    };
    std::string tensor_name;
    for (const char *name : kNames) {
        if (reader.has_tensor(name)) {
            tensor_name = name;
            break;
        }
    }
    if (tensor_name.empty()) {
        return false;
    }

    const auto &meta = reader.tensor_info(tensor_name);
    if (meta.shape.size() != 2) {
        return false;
    }
    const size_t rows = static_cast<size_t>(meta.shape[0]);
    const size_t cols = static_cast<size_t>(meta.shape[1]);

    tokenizer_ = std::make_unique<serius::BPETokenizer>();
    const std::string tok_path = (std::filesystem::path(dir) / "tokenizer.json").string();
    if (std::filesystem::exists(tok_path)) {
        tokenizer_->load(tok_path);
    }
    size_t vocab = tokenizer_->vocab_size();
    size_t dim = 0;
    bool dim_major = false;
    if (vocab > 0 && rows == vocab) {
        dim = cols;  // standard HF [vocab, dim] row-major
    } else if (vocab > 0 && cols == vocab) {
        dim = rows;  // transposed [dim, vocab]
        dim_major = true;
    } else {
        // No usable tokenizer: infer vocab/dim from shape heuristics is
        // unsafe — require the smaller dim to be plausible.
        vocab = rows <= cols ? rows : cols;
        dim = rows <= cols ? cols : rows;
        dim_major = rows > cols;  // guess: wider axis is vocab? fall back flat
        spdlog::warn("EmbeddingManager: no tokenizer.json in '{}'; using shape heuristic", dir);
    }

    core::Tensor t = reader.get_tensor(tensor_name);
    if (t.size() != vocab * dim) {
        return false;
    }
    return store_embedding_matrix(embed_matrix_, t.data(), t.size(), vocab, dim, dim_major);
}

bool EmbeddingManager::is_model_loaded() const
{
    return model_loaded_;
}

std::vector<float> EmbeddingManager::compute_with_simple_tokenizer(const std::string &text)
{
    constexpr size_t kDim = 1024;  // larger space → fewer hash collisions
    std::vector<float> vec(kDim, 0.0f);

    // Signed FNV-1a hashing: bucket index from one hash, sign from a second
    // so unrelated tokens don't all pile onto positive buckets.
    auto add_feature = [&](const std::string &feat, float weight) {
        uint32_t h = 2166136261u;
        for (unsigned char c : feat) {
            h ^= c;
            h *= 16777619u;
        }
        uint32_t h2 = h * 2654435761u;
        const size_t idx = h % kDim;
        vec[idx] += ((h2 >> 16) & 1) ? weight : -weight;
    };

    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        const std::string w = to_lower(word);
        // Word unigram — strong exact-match signal.
        add_feature("w:" + w, 1.0f);
        // Character n-grams capture morphology and partial matches
        // ("quantization" ~ "quantized", typos, subwords). Pad the word so
        // boundary n-grams exist too.
        const std::string padded = "^" + w + "$";
        for (size_t n = 3; n <= 5 && n <= padded.size(); ++n) {
            for (size_t i = 0; i + n <= padded.size(); ++i) {
                add_feature("c:" + padded.substr(i, n), 0.5f);
            }
        }
    }
    // L2 normalize so cosine_similarity is well-behaved.
    double norm = 0.0;
    for (float v : vec) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (auto &v : vec) v = static_cast<float>(v / norm);
    }
    return vec;
}

}  // namespace serius::hub
