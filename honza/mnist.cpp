// ================================================================
// MLP DATAFLOW DIAGRAM
// ---------------------------------------------------------------
// Input image (28x28 pixels flattened into 784 floats)
//         │
//         ▼
//    Input vector x (784)
//         │
//         ▼  Weights W1 (hidden_size x 784)
//         │
//    Linear layer 1: z1 = W1·x + b1
//         │
//         ▼
//    Activation (ReLU)
//         │
//         ▼  Hidden vector h (hidden_size)
//         │
//         ▼  Weights W2 (10 x hidden_size)
//         │
//    Linear layer 2: z2 = W2·h + b2
//         │
//         ▼
//    Softmax: probabilities for 10 classes (digits 0–9)
//         │
//         ▼
//    Loss function: Cross-entropy
// ================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

using namespace std;

// ================================================================
// Function: read_be_uint32
// ---------------------------------------------------------------
// Reads a 32-bit integer stored in *big-endian* byte order from
// a file stream. IDX format (used by MNIST) encodes integers in
// big-endian. Most CPUs are little-endian, so we must reorder.
// ================================================================
uint32_t read_be_uint32(ifstream &f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char *>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

// ================================================================
// Struct: MNISTData
// ---------------------------------------------------------------
// Holds an MNIST dataset (train or test set) in memory.
// - images: flattened pixel intensities [0.0, 1.0]
//           with shape [num_samples * 784]
// - labels: one-hot encoded vectors of size 10 for each sample
// - num_samples: number of examples in dataset
// - rows, cols: image dimensions (28x28 for MNIST)
// ================================================================
struct MNISTData {
    vector<float> images; // [num_samples * 784]
    vector<float> labels; // [num_samples * 10]
    int num_samples{};
    int rows{}, cols{};
};

// ================================================================
// Function: load_mnist
// ---------------------------------------------------------------
// Loads MNIST dataset from IDX image + label files.
// Steps:
//   1. Reads image header (magic, count, rows, cols)
//   2. Reads label header (magic, count)
//   3. Normalizes pixels to [0,1]
//   4. Converts labels into one-hot encoded vectors
// Returns: MNISTData struct
// ================================================================
MNISTData load_mnist(const string &image_file, const string &label_file) {
    ifstream img(image_file, ios::binary);
    ifstream lbl(label_file, ios::binary);

    int magic_img = read_be_uint32(img);
    int num_imgs  = read_be_uint32(img);
    int rows      = read_be_uint32(img);
    int cols      = read_be_uint32(img);

    int magic_lbl = read_be_uint32(lbl);
    int num_lbls  = read_be_uint32(lbl);

    MNISTData data;
    data.num_samples = num_imgs;
    data.rows = rows;
    data.cols = cols;
    data.images.resize(num_imgs * rows * cols);
    data.labels.resize(num_lbls * 10);

    // Read samples one by one
    for (int i = 0; i < num_imgs; i++) {
        // Load image (784 pixels)
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            img.read(reinterpret_cast<char *>(&pixel), 1);
            data.images[i * 784 + j] = pixel / 255.0f;
        }
        // Load label and convert to one-hot encoding
        unsigned char lab;
        lbl.read(reinterpret_cast<char *>(&lab), 1);
        for (int k = 0; k < 10; k++)
            data.labels[i * 10 + k] = (k == lab) ? 1.0f : 0.0f;
    }
    return data;
}

// ================================================================
// Activation functions
// ---------------------------------------------------------------
// ReLU: f(x) = max(0, x)
// dReLU: derivative wrt output (0 if x<=0 else 1)
// softmax: converts logits into probabilities
// ================================================================
inline float relu(float x)   { return (x > 0) ? x : 0.0f; }
inline float d_relu(float y) { return (y > 0) ? 1.0f : 0.0f; }

vector<float> softmax(span<const float> z) {
    vector<float> out(z.size());
    float m = *ranges::max_element(z); // subtract max for stability
    float sum = 0.0f;
    for (float v : z) sum += exp(v - m);
    for (size_t i = 0; i < z.size(); i++)
        out[i] = exp(z[i] - m) / sum;
    return out;
}

// ================================================================
// Struct: MLP
// ---------------------------------------------------------------
// Defines a fully-connected neural network with:
//   Input layer: 784 (28x28 pixels)
//   Hidden layer: configurable size
//   Output layer: 10 (digit classes)
//
// Members:
//   W1, b1 - weights/biases from input -> hidden
//   W2, b2 - weights/biases from hidden -> output
//   hidden_buf - reusable buffer for hidden activations
//   output_buf - reusable buffer for output logits
//
// Functions:
//   forward(x) - run single sample forward pass
//   train_batch(data, indices) - update weights with mini-batch
//   evaluate(data) - compute classification accuracy
// ================================================================
struct MLP {
    int in_size, hidden_size, out_size;
    vector<float> W1, b1, W2, b2;

    // Reuse buffers to avoid repeated allocations
    vector<float> hidden_buf, output_buf;

    // ------------------------------------------------------------
    // Constructor
    // Initializes weights randomly in [-0.1, 0.1]
    // Biases start at 0
    // ------------------------------------------------------------
    MLP(int in_s, int h_s, int o_s)
        : in_size(in_s), hidden_size(h_s), out_size(o_s) {
        W1.resize(in_size * hidden_size);
        b1.assign(hidden_size, 0.0f);
        W2.resize(hidden_size * out_size);
        b2.assign(out_size, 0.0f);

        hidden_buf.resize(hidden_size);
        output_buf.resize(out_size);

        mt19937 gen{random_device{}()};
        uniform_real_distribution<float> dis(-0.1f, 0.1f);
        for (auto &w : W1) w = dis(gen);
        for (auto &w : W2) w = dis(gen);
    }

    // ------------------------------------------------------------
    // Function: forward
    // ------------------------------------------------------------
    // Runs one forward pass on a sample:
    //   1. Compute hidden activations (ReLU)
    //   2. Compute output logits
    //   3. Apply softmax -> probability distribution
    //
    // Parameters:
    //   x : span of input pixels (784 floats)
    //
    // Returns:
    //   span<float> with output probabilities
    // ------------------------------------------------------------
    span<const float> forward(span<const float> x) {
        // Hidden layer z1 = W1*x + b1
        for (int j = 0; j < hidden_size; j++) {
            float z = b1[j];
            for (int i = 0; i < in_size; i++)
                z += W1[j * in_size + i] * x[i];
            hidden_buf[j] = relu(z);
        }

        // Output layer z2 = W2*h + b2
        for (int k = 0; k < out_size; k++) {
            float z = b2[k];
            for (int j = 0; j < hidden_size; j++)
                z += W2[k * hidden_size + j] * hidden_buf[j];
            output_buf[k] = z;
        }

        // Apply softmax inplace
        auto sm = softmax(output_buf);
        ranges::copy(sm, output_buf.begin());
        return span<const float>(output_buf);
    }

    // ------------------------------------------------------------
    // Function: train_batch
    // ------------------------------------------------------------
    // Performs one mini-batch gradient descent update.
    // For each sample:
    //   1. Forward pass
    //   2. Compute cross-entropy loss
    //   3. Backpropagate error
    //   4. Accumulate gradients
    // After all samples:
    //   5. Average gradients
    //   6. Update weights and biases
    //
    // Parameters:
    //   data          : dataset (images + labels)
    //   batch_indices : indices of samples in this batch
    //   lr            : learning rate
    //   batch_loss    : reference, accumulates loss for logging
    // ------------------------------------------------------------
    void train_batch(const MNISTData &data,
                     span<const int> batch_indices,
                     float lr, float &batch_loss) {
        // Gradient accumulators (zero init)
        vector<float> dW1(W1.size()), db1(hidden_size);
        vector<float> dW2(W2.size()), db2(out_size);

        for (int idx : batch_indices) {
            auto x = span(data.images).subspan(idx * in_size, in_size);
            auto y = span(data.labels).subspan(idx * out_size, out_size);

            auto out = forward(x);

            // Loss contribution: -log(p[label])
            int truth = int(ranges::max_element(y) - y.begin());
            batch_loss -= log(out[truth] + 1e-8f);

            // Output error (softmax + cross-entropy gradient)
            vector<float> delta2(out_size);
            for (int k = 0; k < out_size; k++)
                delta2[k] = out[k] - y[k];

            // Gradients for W2, b2
            for (int k = 0; k < out_size; k++) {
                for (int j = 0; j < hidden_size; j++)
                    dW2[k * hidden_size + j] += delta2[k] * hidden_buf[j];
                db2[k] += delta2[k];
            }

            // Hidden error backprop
            vector<float> delta1(hidden_size);
            for (int j = 0; j < hidden_size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < out_size; k++)
                    sum += W2[k * hidden_size + j] * delta2[k];
                delta1[j] = d_relu(hidden_buf[j]) * sum;
            }

            // Gradients for W1, b1
            for (int j = 0; j < hidden_size; j++) {
                for (int i = 0; i < in_size; i++)
                    dW1[j * in_size + i] += delta1[j] * x[i];
                db1[j] += delta1[j];
            }
        }

        // Apply gradient descent updates (average over batch)
        float scale = 1.0f / batch_indices.size();
        for (size_t i = 0; i < W1.size(); i++) W1[i] -= lr * dW1[i] * scale;
        for (size_t i = 0; i < b1.size(); i++) b1[i] -= lr * db1[i] * scale;
        for (size_t i = 0; i < W2.size(); i++) W2[i] -= lr * dW2[i] * scale;
        for (size_t i = 0; i < b2.size(); i++) b2[i] -= lr * db2[i] * scale;
    }

    // ------------------------------------------------------------
    // Function: evaluate
    // ------------------------------------------------------------
    // Evaluates network accuracy on a dataset.
    // For each sample:
    //   1. Forward pass
    //   2. Choose argmax prediction
    //   3. Compare to label
    // Computes % correctly classified.
    // ------------------------------------------------------------
    float evaluate(const MNISTData &data) {
        int correct = 0;
        for (auto i : views::iota(0, data.num_samples)) {
            auto x = span(data.images).subspan(i * in_size, in_size);
            auto y = span(data.labels).subspan(i * out_size, out_size);
            auto out = forward(x);
            int pred  = int(ranges::max_element(out) - out.begin());
            int truth = int(ranges::max_element(y)   - y.begin());
            if (pred == truth) correct++;
        }
        return 100.0f * correct / data.num_samples;
    }
};

// ================================================================
// Main training loop
// ---------------------------------------------------------------
// Workflow:
//   1. Load training + test datasets from IDX files
//   2. Initialize MLP with 784 -> hidden -> 10
//   3. For each epoch:
//      - Shuffle training order
//      - Create mini-batches via indices
//      - Run training step
//      - Compute training/test accuracy
//   4. Print metrics each epoch
// ================================================================
int main() {
    auto train = load_mnist("train-images.idx3-ubyte",
                            "train-labels.idx1-ubyte");
    auto test  = load_mnist("t10k-images.idx3-ubyte",
                            "t10k-labels.idx1-ubyte");

    // Model configuration
    MLP net(784, 128, 10); // input=784, hidden=128, output=10 classes

    // Training hyperparameters
    int epochs = 5;        // number of training passes over dataset
    int batch_size = 64;   // size of mini-batch for SGD
    float lr = 0.01f;      // learning rate

    // Shuffle container
    vector<int> indices(train.num_samples);
    iota(indices.begin(), indices.end(), 0);

    mt19937 gen{random_device{}()};

    for (int e = 0; e < epochs; e++) {
        // Shuffle dataset order each epoch
        ranges::shuffle(indices, gen);

        float total_loss = 0.0f;

        // Iterate mini-batches
        for (int b = 0; b < train.num_samples; b += batch_size) {
            int curr_bs = min(batch_size, train.num_samples - b);
            auto batch_idx = span(indices).subspan(b, curr_bs);

            // Train step
            net.train_batch(train, batch_idx, lr, total_loss);
        }

        // After epoch: evaluate performance
        float train_acc = net.evaluate(train);
        float test_acc  = net.evaluate(test);

        cout << "Epoch " << e+1
             << " Loss=" << total_loss / train.num_samples
             << " TrainAcc=" << train_acc
             << "% TestAcc=" << test_acc << "%\n";
    }
}
