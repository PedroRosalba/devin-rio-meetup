#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpt2 {

struct ModelConfig {
  int vocab_size = 50257;
  int n_embd = 768;
  int n_head = 12;
  int n_layer = 12;
  int n_ctx = 1024;
  int n_inner = 3072;
  float layer_norm_epsilon = 1e-5f;
};

struct TensorView {
  std::string name;
  std::vector<int> shape;
  const float* data = nullptr;
  std::size_t numel = 0;
};

class WeightStore {
 public:
  static std::unique_ptr<WeightStore> load(const std::string& manifest_path);

  const ModelConfig& config() const { return config_; }
  const TensorView* find(const std::string& name) const;

  std::size_t total_bytes() const { return blob_.size(); }

 private:
  ModelConfig config_;
  std::vector<float> blob_;
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace gpt2
