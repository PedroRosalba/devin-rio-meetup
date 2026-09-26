#include "gpt2_cuda/weights.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string manifest_dir(const std::string& manifest_path) {
  auto slash = manifest_path.find_last_of("/\\");
  if (slash == std::string::npos) return ".";
  return manifest_path.substr(0, slash);
}

int find_int_after(const std::string& s, std::size_t from, const char* key) {
  auto p = s.find(key, from);
  if (p == std::string::npos) return -1;
  p = s.find(':', p);
  if (p == std::string::npos) return -1;
  ++p;
  while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
  return std::stoi(s.substr(p));
}

float find_float_after(const std::string& s, std::size_t from, const char* key) {
  auto p = s.find(key, from);
  if (p == std::string::npos) return 0.f;
  p = s.find(':', p);
  if (p == std::string::npos) return 0.f;
  ++p;
  while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
  return std::stof(s.substr(p));
}

std::vector<int> parse_shape(const std::string& block) {
  std::vector<int> shape;
  auto lb = block.find('[');
  auto rb = block.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos) return shape;
  std::stringstream ss(block.substr(lb + 1, rb - lb - 1));
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (!part.empty()) shape.push_back(std::stoi(part));
  }
  return shape;
}

}  // namespace

namespace gpt2 {

const TensorView* WeightStore::find(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) return nullptr;
  return &it->second;
}

std::unique_ptr<WeightStore> WeightStore::load(const std::string& manifest_path) {
  auto store = std::make_unique<WeightStore>();
  const std::string json = read_file(manifest_path);
  const std::string dir = manifest_dir(manifest_path);

  auto model_pos = json.find("\"model\":");
  if (model_pos == std::string::npos) throw std::runtime_error("manifest missing model");
  const std::string model_block = json.substr(model_pos, 512);
  store->config_.vocab_size = find_int_after(model_block, 0, "\"vocab_size\"");
  store->config_.n_embd = find_int_after(model_block, 0, "\"n_embd\"");
  store->config_.n_head = find_int_after(model_block, 0, "\"n_head\"");
  store->config_.n_layer = find_int_after(model_block, 0, "\"n_layer\"");
  store->config_.n_ctx = find_int_after(model_block, 0, "\"n_ctx\"");
  store->config_.n_inner = find_int_after(model_block, 0, "\"n_inner\"");
  store->config_.layer_norm_epsilon = find_float_after(model_block, 0, "\"layer_norm_epsilon\"");

  const std::string bin_path = dir + "/gpt2_weights_fp32.bin";
  std::ifstream bin(bin_path, std::ios::binary);
  if (!bin) throw std::runtime_error("cannot open " + bin_path);
  bin.seekg(0, std::ios::end);
  const std::size_t file_bytes = static_cast<std::size_t>(bin.tellg());
  bin.seekg(0, std::ios::beg);
  std::vector<char> raw(file_bytes);
  bin.read(raw.data(), static_cast<std::streamsize>(file_bytes));
  if (raw.size() < 8 || std::string(raw.data(), 8) != "GPT2WGT1") {
    throw std::runtime_error("bad weight magic");
  }
  const float* base = reinterpret_cast<const float*>(raw.data() + 8);
  store->blob_.assign(base, base + (raw.size() - 8) / sizeof(float));

  auto list_pos = json.find("\"tensor_list\":");
  if (list_pos == std::string::npos) throw std::runtime_error("manifest missing tensor_list");
  list_pos = json.find('[', list_pos);
  if (list_pos == std::string::npos) throw std::runtime_error("bad tensor_list");

  std::size_t i = list_pos + 1;
  while (i < json.size()) {
    auto obj = json.find('{', i);
    if (obj == std::string::npos || obj > json.find(']', i)) break;
    auto end = json.find('}', obj);
    if (end == std::string::npos) break;
    std::string block = json.substr(obj, end - obj + 1);
    auto name_key = block.find("\"name\":");
    if (name_key == std::string::npos) {
      i = end + 1;
      continue;
    }
    auto q0 = block.find('"', name_key + 7);
    auto q1 = block.find('"', q0 + 1);
    if (q0 == std::string::npos || q1 == std::string::npos) {
      i = end + 1;
      continue;
    }
    std::string name = block.substr(q0 + 1, q1 - q0 - 1);

    int offset = find_int_after(block, 0, "\"offset\"");
    int length_bytes = find_int_after(block, 0, "\"length_bytes\"");
    if (offset >= 0 && length_bytes > 0) {
      TensorView tv;
      tv.name = name;
      tv.shape = parse_shape(block);
      tv.data = store->blob_.data() + (offset / static_cast<int>(sizeof(float)));
      tv.numel = static_cast<std::size_t>(length_bytes / sizeof(float));
      store->tensors_[name] = tv;
    }
    i = end + 1;
  }

  if (store->tensors_.find("wte") == store->tensors_.end()) {
    throw std::runtime_error("manifest parse failed (missing wte) — re-run export_gpt2_weights.py");
  }
  return store;
}

}  // namespace gpt2
