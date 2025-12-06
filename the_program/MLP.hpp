#ifndef MLP_HPP
#define MLP_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Vector = std::vector<float>;

///----ACTIVATION FUNCTIONS/LOSS FUNCTIONS/DERIVATIONS------
enum class Activation { ReLU, Sigmoid, Tanh, Linear };

inline float apply_activation(float s, Activation a) {
  switch (a) {
  case Activation::ReLU:
    return s > 0.f ? s : 0.f;
  case Activation::Sigmoid:
    return 1.f / (1.f + std::exp(-s));
  case Activation::Tanh:
    return std::tanh(s);
  case Activation::Linear:
    return s;
  }
  return s;
}

inline float apply_activation_derivative(float s, Activation a) {
  switch (a) {
  case Activation::ReLU:
    return s > 0.f ? 1.f : 0.f;

  case Activation::Sigmoid: {
    float y = 1.f / (1.f + std::exp(-s));
    return y * (1.f - y);
  }

  case Activation::Tanh: {
    float t = std::tanh(s);
    return 1.f - t * t;
  }

  case Activation::Linear:
    return 1.f;
  }
  return 1.f;
}
///--------------------UTILITY------------------------------

inline void readMNISTImage(Vector &layer_in, int imageIndex,
                           const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filePath);
  }
  // Skip the first 16 bytes of metadata
  // (magic number + number of images + rows + cols)
  file.seekg(16 + imageIndex * 28 * 28, std::ios::beg);

  // Temporary buffer to read raw uint8 pixels
  std::vector<uint8_t> temp(28 * 28);
  file.read(reinterpret_cast<char *>(temp.data()), temp.size());

  // Convert to float in range [0, 1]
  for (int i = 0; i < 28 * 28; i++)
    layer_in[i] = temp[i] / 255.0f;
}

inline int readMNISTLabel(int labelIndex, const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open file: " + filePath);

  // Skip the first 8 bytes (metadata)
  file.seekg(8 + labelIndex, std::ios::beg);

  uint8_t label;
  file.read(reinterpret_cast<char *>(&label), 1);

  return static_cast<int>(label);
}

///------------- MATRIX STRUCT WITH OPTIMIZATION -------------
struct Matrix {
  Vector data;
  int rows;
  int cols;

  Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

  float &operator()(int r, int c) { return data[r * cols + c]; }

  const float &operator()(int r, int c) const { return data[r * cols + c]; }

  // Optimized forward pass: y = activation(W * x)
  // Loops are swapped to access W in row-major order (sequentially)
  void forward_pass(const Vector &in, Vector &potential_out, Vector &out,
                    Activation act) const {
    // Initialize potentials with bias (first row of weights)
    for (int k = 0; k < cols; ++k) {
      potential_out[k] = (*this)(0, k);
    }

    // Matrix multiplication: potential += W * in
    // Outer loop: inputs (rows of W), Inner loop: outputs (cols of W)
    // W access is sequential
    for (int j = 1; j < rows; ++j) {
      float input_val = in[j - 1];
      if (input_val == 0.0f)
        continue;

      for (int k = 0; k < cols; ++k) {
        potential_out[k] += (*this)(j, k) * input_val;
      }
    }

    // Apply activation
    for (int k = 0; k < cols; ++k) {
      out[k] = apply_activation(potential_out[k], act);
    }
  }

  // Optimized backward update: W += eta * delta * input
  void backward_update(float eta, const Vector &delta, const Vector &in) {
    // Update biases (row 0)
    for (int k = 0; k < cols; ++k) {
      (*this)(0, k) += eta * delta[k];
    }

    // Update weights
    // Outer loop: inputs (rows), Inner loop: outputs (cols)
    // -> Sequential access
    for (int j = 1; j < rows; ++j) {
      float input_val = in[j - 1];
      if (input_val == 0.0f)
        continue;

      float scaled_input = eta * input_val;
      for (int k = 0; k < cols; ++k) {
        (*this)(j, k) += scaled_input * delta[k];
      }
    }
  }

  // Backward pass: delta_in = (W^T * delta_out) * activation_derivative
  void backward_pass(const Vector &delta_out, Vector &delta_in,
                     const Vector &potential_in, Activation act) const {
    int kmax = delta_in.size();  // neurons in current layer (rows-1 of W)
    int lmax = delta_out.size(); // neurons in next layer (cols of W)

    for (int k = 0; k < kmax; ++k) {
      float sum_result = 0;
      for (int l = 0; l < lmax; ++l) {
        sum_result += delta_out[l] * (*this)(k + 1, l);
      }
      delta_in[k] =
          sum_result * apply_activation_derivative(potential_in[k], act);
    }
  }
};

inline void fill_random_matrix(Matrix &mat) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto &v : mat.data)
    v = dist(gen);
}

inline int get_digit_from_outputs(const Vector &outputs) {
  return std::distance(outputs.begin(),
                       std::max_element(outputs.begin(), outputs.end()));
}
inline void get_desired_outputs_from_digit(Vector &desired_outputs, int digit) {
  std::fill(desired_outputs.begin(), desired_outputs.end(), 0.0f);
  desired_outputs[digit] = 1;
}

///-------------------FUNCTIONS FOR ARCHITECTURE------------------------

/*
here are 4 structures for network architecture,
Layer holds every parameter for neuron layer
Forward_propagation holds function and logic for forward cycle
Backward_propagation holds function and logic for backpropagation
MLP put together upper structures
*/

/////////////////////////////
struct Layer {
  Vector y;       // outputs (activations)
  Vector s;       // potentials
  Vector delta;   // error signals
  Activation act; // activation function
};

//////////////////////////////

struct Forward_propagation {
  void forward_step(const Vector &neuron_vec_in, Vector &neuron_vec_out,
                    Vector &potential_out, const Matrix &weight_matrix,
                    Activation act);
};

inline void Forward_propagation::forward_step(const Vector &neuron_vec_in,
                                              Vector &neuron_vec_out,
                                              Vector &potential_out,
                                              const Matrix &weight_matrix,
                                              Activation act) {
  weight_matrix.forward_pass(neuron_vec_in, potential_out, neuron_vec_out, act);
}
/////////////////////////////////////
struct Backward_propagation {
  // TODO: predelat na static?
  void backward_step(Matrix &weight_matrix,     // latex delta w_jk^m
                     float eta,                 // learning rate
                     const Vector &delta_km,    // latex delta_k^m
                     const Vector &y_in) const; // latex y_j^{m-1}

  void fill_delta_hidden(Vector &delta_km,            // latex delta_k^m
                         const Vector &delta_lm_out,  // latex delta_l^m+1
                         const Matrix &weight_matrix, // latex w_kl^m+1
                         const Vector &potential_km,  // latex s_k^m
                         Activation act) const;       // derivative of ac fc

  void fill_delta_output(Vector &delta_ko,           // latex delta_k^o
                         const Vector &desired_ko,   // latex d_k^o
                         const Vector &y_out,        // latex y_k^o
                         const Vector &potential_ko, // latex s_k^o
                         Activation act) const;      // derivative of ac fc
};

// Backpropagation function implements equation 11 (latex)
inline void
Backward_propagation::backward_step(Matrix &weight_matrix, // latex delta w_jk^m
                                    float eta,             // learning rate
                                    const Vector &delta_km,   // latex delta_k^m
                                    const Vector &y_in) const // latex y_j^{m-1}
{
#ifndef NDEBUG
  if (weight_matrix.rows != (int)y_in.size() + 1 ||
      weight_matrix.cols != (int)delta_km.size()) {
    throw std::runtime_error("backward_step: dimension mismatch");
  }
#endif

  weight_matrix.backward_update(eta, delta_km, y_in);
}

// Function fills delta vector in a general hidden layer - equation 12 (latex)
inline void Backward_propagation::fill_delta_hidden(
    Vector &delta_km,            // latex delta_k^m
    const Vector &delta_lm_out,  // latex delta_l^m+1
    const Matrix &weight_matrix, // latex w_kl^m+1
    const Vector &potential_km,  // latex s_k^m
    Activation act) const        // derivative of ac fc
{
#ifndef NDEBUG
  if (weight_matrix.rows != (int)potential_km.size() + 1 ||
      weight_matrix.cols != (int)delta_lm_out.size() ||
      (int)delta_km.size() != (int)potential_km.size()) {
    throw std::runtime_error("fill_delta_hidden: dimension mismatch");
  }
#endif

  weight_matrix.backward_pass(delta_lm_out, delta_km, potential_km, act);
}

// Function fills delta vector in the output layer - equation 13 (latex)
inline void Backward_propagation::fill_delta_output(
    Vector &delta_ko,           // latex delta_k^o
    const Vector &desired_ko,   // latex d_k^o
    const Vector &y_out,        // latex y_k^o
    const Vector &potential_ko, // latex s_k^o
    Activation act) const       // derivative of ac fc
{
#ifndef NDEBUG
  if ((int)delta_ko.size() != (int)potential_ko.size() ||
      (int)desired_ko.size() != (int)potential_ko.size() ||
      (int)y_out.size() != (int)potential_ko.size()) {
    throw std::runtime_error("fill_delta_output: dimension mismatch");
  }
#endif

  const int kmax = (int)potential_ko.size();
  for (int k = 0; k < kmax; k++) {
    float derivative = apply_activation_derivative(potential_ko[k], act);
    delta_ko[k] = (desired_ko[k] - y_out[k]) * derivative;
  }
}

/////////////////////////////////////
struct MLP {

  // Topology
  std::vector<Layer> L;  // L[0]..L[n-1]
  std::vector<Matrix> W; // W[i] is (|L[i]|+1)x|L[i+1]|

  // Engines (only function holders)
  Forward_propagation fp;
  Backward_propagation bp;

  // Topology and weights initialization
  void init(const std::vector<int> &layer_sizes,
            const std::vector<Activation> &acts);

  // One pass: forward through all layers
  void forward_one_sample();

  // Backward: calculate deltas for all layers
  void backward_build_deltas(const Vector &desired);

  // Update: rewrite W[i] based on deltas and inputs
  void apply_updates(float eta);

  // Training loop
  void train(int epochs, float eta, int n_samples,
             const std::string &images_file, const std::string &labels_file);

  // Testing loop
  void test(const std::string &image_file, const std::string &label_file,
            int n_samples);

  // Save model to file
  void save(const std::string &filename) const;

  // Load model from file
  void load(const std::string &filename);

private:
  // Helper: Trains one sample, returns true if prediction was correct
  bool train_one_sample(int idx, float eta, Vector &desired,
                        const std::string &images_file,
                        const std::string &labels_file);

  // Helper: Trains one epoch
  void train_one_epoch(int epoch, float eta, int n_samples,
                       std::vector<int> &order, std::mt19937 &rng,
                       Vector &desired, const std::string &images_file,
                       const std::string &labels_file);
};

inline void MLP::init(const std::vector<int> &layer_sizes,
                      const std::vector<Activation> &acts) {

  // --- Layer count == activation count (excluding input layer) ---
  if (layer_sizes.size() != acts.size() + 1) {
    throw std::runtime_error("Activation count must be 1 less than layer "
                             "count (input layer has no activation).");
  }

  const int layer_count = (int)layer_sizes.size();
  L.resize(layer_count);

  // --- Layer initialization ---
  for (int i = 0; i < layer_count; ++i) {
    L[i].y.resize(layer_sizes[i]);     // neuron outputs
    L[i].s.resize(layer_sizes[i]);     // potentials
    L[i].delta.resize(layer_sizes[i]); // error delta

    // Input layer (i=0) has no activation, others take from acts[i-1]
    if (i > 0) {
      L[i].act = acts[i - 1];
    }
  }

  // --- Weight matrix initialization ---
  W.clear();
  W.reserve(layer_count - 1); // one matrix between every two layers

  for (int i = 0; i < layer_count - 1; ++i) {
    const int in_size = layer_sizes[i];
    const int out_size = layer_sizes[i + 1];

    // +1 for bias
    W.emplace_back(in_size + 1, out_size);

    fill_random_matrix(W[i]);
  }
}

inline void MLP::forward_one_sample() {
  for (size_t i = 0; i + 1 < L.size(); ++i) {
    fp.forward_step(L[i].y,      // neuron_vec_in
                    L[i + 1].y,  // neuron_vec_out
                    L[i + 1].s,  // potential_out
                    W[i],        // weight_matrix (const&)
                    L[i + 1].act // Activation
    );
  }
}

inline void MLP::backward_build_deltas(const Vector &desired) {
  // o = output layer index
  const int o = static_cast<int>(L.size()) - 1;

  // 1) output deltas: delta^out = (d - y) * phi'(s)
  bp.fill_delta_output(L[o].delta, // delta_ko
                       desired,    // desired_ko
                       L[o].y,     // y_out
                       L[o].s,     // potential_ko
                       L[o].act);

  // 2) hidden deltas top-down: i = o-1, ..., 1
  for (int i = o - 1; i >= 1; --i) {
    bp.fill_delta_hidden(L[i].delta,     // delta_km
                         L[i + 1].delta, // delta_lm_out
                         W[i],   // weight_matrix (K+1 x L), row 0 = bias
                         L[i].s, // potential_km
                         L[i].act);
  }
}
inline void MLP::apply_updates(float eta) {
  const int o = static_cast<int>(L.size()) - 1; // last layer = L[o]
  for (int i = 0; i < o; ++i) {
    // W[i]: (|L[i]|+1) x |L[i+1]|
    // target layer delta = L[i+1].delta, input = L[i].y
    bp.backward_step(W[i], eta, L[i + 1].delta, L[i].y);
  }
}

inline bool MLP::train_one_sample(int idx, float eta, Vector &desired,
                                  const std::string &images_file,
                                  const std::string &labels_file) {
  // 1) load input and label directly
  readMNISTImage(L[0].y, idx, images_file);
  int label = readMNISTLabel(idx, labels_file);
  get_desired_outputs_from_digit(desired, label);

  // 2) forward
  forward_one_sample();

  // 3) metric (no other dependencies)
  int pred = get_digit_from_outputs(L.back().y);
  bool correct = (pred == label);

  // 4) backprop (deltas) + update
  backward_build_deltas(desired);
  apply_updates(eta);

  return correct;
}

inline void MLP::train_one_epoch(int epoch, float eta, int n_samples,
                                 std::vector<int> &order, std::mt19937 &rng,
                                 Vector &desired,
                                 const std::string &images_file,
                                 const std::string &labels_file) {
  std::shuffle(order.begin(), order.end(), rng);
  int correct_count = 0;

  for (int t = 0; t < n_samples; ++t) {
    const int idx = order[t];
    if (train_one_sample(idx, eta, desired, images_file, labels_file)) {
      correct_count++;
    }
  }

  std::cout << "epoch " << epoch << " acc=" << (float)correct_count / n_samples
            << "\n";
}

inline void MLP::train(int epochs, float eta, int n_samples,
                       const std::string &images_file,
                       const std::string &labels_file) {
  Vector desired(L.back().y.size(), 0.f);

  std::vector<int> order(n_samples);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng{std::random_device{}()};

  for (int e = 0; e < epochs; ++e) {
    train_one_epoch(e, eta, n_samples, order, rng, desired, images_file,
                    labels_file);
  }
}

inline void MLP::test(const std::string &image_file,
                      const std::string &label_file, int n_samples) {
  int correct_count = 0;
  for (int i = 0; i < n_samples; ++i) {
    readMNISTImage(L[0].y, i, image_file);
    int label = readMNISTLabel(i, label_file);

    forward_one_sample();

    int pred = get_digit_from_outputs(L.back().y);
    if (pred == label) {
      correct_count++;
    }
  }
  std::cout << "Testing accuracy: " << (float)correct_count / n_samples << "\n";
}

inline void MLP::save(const std::string &filename) const {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for saving: " + filename);
  }

  // 1. Magic number / Version
  uint32_t magic = 0x4D4C5031; // "MLP1"
  file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

  // 2. Topology
  uint32_t layer_count = static_cast<uint32_t>(L.size());
  file.write(reinterpret_cast<const char *>(&layer_count), sizeof(layer_count));

  for (const auto &layer : L) {
    uint32_t size = static_cast<uint32_t>(layer.y.size());
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }

  // 3. Activations (skip input layer)
  for (size_t i = 1; i < L.size(); ++i) {
    int act_val = static_cast<int>(L[i].act);
    file.write(reinterpret_cast<const char *>(&act_val), sizeof(act_val));
  }

  // 4. Weights
  for (const auto &mat : W) {
    file.write(reinterpret_cast<const char *>(mat.data.data()),
               mat.data.size() * sizeof(float));
  }
}

inline void MLP::load(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for loading: " + filename);
  }

  // 1. Magic number
  uint32_t magic;
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (magic != 0x4D4C5031) {
    throw std::runtime_error("Invalid model file format");
  }

  // 2. Topology
  uint32_t layer_count;
  file.read(reinterpret_cast<char *>(&layer_count), sizeof(layer_count));

  std::vector<int> layer_sizes(layer_count);
  for (uint32_t i = 0; i < layer_count; ++i) {
    uint32_t size;
    file.read(reinterpret_cast<char *>(&size), sizeof(size));
    layer_sizes[i] = static_cast<int>(size);
  }

  // 3. Activations
  std::vector<Activation> acts;
  for (uint32_t i = 1; i < layer_count; ++i) {
    int act_val;
    file.read(reinterpret_cast<char *>(&act_val), sizeof(act_val));
    acts.push_back(static_cast<Activation>(act_val));
  }

  // Re-initialize network
  init(layer_sizes, acts);

  // 4. Weights
  for (auto &mat : W) {
    file.read(reinterpret_cast<char *>(mat.data.data()),
              mat.data.size() * sizeof(float));
  }
}

#endif // MLP_HPP
