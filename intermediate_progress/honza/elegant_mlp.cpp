// ================================================================ 
// MLP for MNIST classification 
// --------------------------------------------------------------- 
// This file contains a C++ implementation of a Multi-Layer 
// Perceptron (MLP) for classifying handwritten digits from the 
// MNIST dataset. 
// 
// The implementation is inspired by the code structure from 
// honza/mnist.cpp and the MLP logic from 
// spoluprace/MLP_structures.cpp. 
// 
// The network architecture is flexible, allowing for multiple 
// hidden layers with different activation functions. 
// 
// The MLP can be trained, and the trained model can be saved to 
// a file and loaded back for inference. 
// ================================================================ 

#include <iostream> 
#include <fstream> 
#include <vector> 
#include <iomanip> 
#include <cstdint> 
#include <cmath> 
#include <random> 
#include <algorithm> 
#include <numeric> 
#include <stdexcept> 
#include <string> 
#include <sstream> 

// ================================================================ 
// MNIST Data Loading 
// 
// Functions to load the MNIST dataset from the binary IDX format. 
// ================================================================ 

uint32_t swap_endian(uint32_t val) { 
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF); 
    return (val << 16) | (val >> 16); 
} 

void read_mnist_images(const std::string& path, std::vector<std::vector<float>>& images, int max_images = -1) { 
    std::ifstream file(path, std::ios::binary); 
    if (!file.is_open()) { 
        throw std::runtime_error("Cannot open file: " + path); 
    } 

    uint32_t magic, num_images, rows, cols; 
    file.read(reinterpret_cast<char*>(&magic), 4); 
    file.read(reinterpret_cast<char*>(&num_images), 4); 
    file.read(reinterpret_cast<char*>(&rows), 4); 
    file.read(reinterpret_cast<char*>(&cols), 4); 

    magic = swap_endian(magic); 
    num_images = swap_endian(num_images); 
    rows = swap_endian(rows); 
    cols = swap_endian(cols); 

    if (max_images > 0) { 
        num_images = std::min(num_images, (uint32_t)max_images); 
    } 

    images.resize(num_images, std::vector<float>(rows * cols)); 
    std::vector<uint8_t> buffer(rows * cols); 

    for (uint32_t i = 0; i < num_images; ++i) { 
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()); 
        for (size_t j = 0; j < buffer.size(); ++j) { 
            images[i][j] = buffer[j] / 255.0f; 
        } 
    } 
} 

void read_mnist_labels(const std::string& path, std::vector<int>& labels, int max_labels = -1) { 
    std::ifstream file(path, std::ios::binary); 
    if (!file.is_open()) { 
        throw std::runtime_error("Cannot open file: " + path); 
    } 

    uint32_t magic, num_labels; 
    file.read(reinterpret_cast<char*>(&magic), 4); 
    file.read(reinterpret_cast<char*>(&num_labels), 4); 

    magic = swap_endian(magic); 
    num_labels = swap_endian(num_labels); 
    
    if (max_labels > 0) { 
        num_labels = std::min(num_labels, (uint32_t)max_labels); 
    } 

    labels.resize(num_labels); 
    std::vector<uint8_t> buffer(num_labels); 
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()); 

    for (uint32_t i = 0; i < num_labels; ++i) { 
        labels[i] = buffer[i]; 
    } 
} 

// ================================================================ 
// Activation Functions 
// ================================================================ 

enum class Activation { 
    ReLU, 
    Sigmoid, 
    Linear 
}; 

float apply_activation(float x, Activation activation) { 
    switch (activation) { 
        case Activation::ReLU:    return x > 0.0f ? x : 0.0f; 
        case Activation::Sigmoid: return 1.0f / (1.0f + std::exp(-x)); 
        case Activation::Linear:  return x; 
    } 
    return x; 
} 

float apply_activation_derivative(float x, Activation activation) { 
    switch (activation) { 
        case Activation::ReLU: 
            return x > 0.0f ? 1.0f : 0.0f; 
        case Activation::Sigmoid: { 
            float sigmoid = 1.0f / (1.0f + std::exp(-x)); 
            return sigmoid * (1.0f - sigmoid); 
        } 
        case Activation::Linear: 
            return 1.0f; 
    } 
    return 1.0f; 
} 

// ================================================================ 
// Layer and MLP structures 
// ================================================================ 

// ---------------------------------------------------------------- 
// Struct: Layer 
// ---------------------------------------------------------------- 
// Represents a single layer in the MLP. 
// 
// Members: 
//   neurons    - The activation values of the neurons. 
//   potentials - The pre-activation values of the neurons. 
//   deltas     - The error signals for each neuron. 
//   activation - The activation function for this layer. 
// ---------------------------------------------------------------- 
struct Layer { 
    std::vector<float> neurons; 
    std::vector<float> potentials; 
    std::vector<float> deltas; 
    Activation activation; 

    Layer(int size, Activation act) 
        : neurons(size), potentials(size), deltas(size), activation(act) {} 
}; 

// ---------------------------------------------------------------- 
// Class: MLP 
// ---------------------------------------------------------------- 
// Defines a multi-layer perceptron. 
// 
// The network is defined by a vector of layer sizes and a 
// vector of activation functions. The class provides methods for 
// training, prediction, saving, and loading the model. 
// ---------------------------------------------------------------- 
class MLP { 
public: 
    MLP(const std::vector<int>& layer_sizes, const std::vector<Activation>& activations); 

    void train(const std::vector<std::vector<float>>& train_images, 
               const std::vector<int>& train_labels, 
               int epochs, float learning_rate); 

    int predict(const std::vector<float>& image); 
    int predict_from_vector(const std::vector<float>& drawing_vector); 

    void save_model(const std::string& path); 
    void load_model(const std::string& path); 


private: 
    void initialize_weights(); 
    void forward_pass(const std::vector<float>& input); 
    void backward_pass(const std::vector<float>& desired_output); 
    void update_weights(float learning_rate); 

    std::vector<Layer> layers_; 
    std::vector<std::vector<std::vector<float>>> weights_; 
}; 

// ---------------------------------------------------------------- 
// MLP: Constructor 
// ---------------------------------------------------------------- 
// Initializes the MLP with the given layer sizes and activation 
// functions. 
// ---------------------------------------------------------------- 
MLP::MLP(const std::vector<int>& layer_sizes, const std::vector<Activation>& activations) { 
    if (layer_sizes.size() != activations.size() + 1) { 
        throw std::invalid_argument("Layer sizes and activations mismatch."); 
    } 

    for (size_t i = 0; i < layer_sizes.size(); ++i) { 
        layers_.emplace_back(layer_sizes[i], i == 0 ? Activation::Linear : activations[i-1]); 
    } 

    initialize_weights(); 
} 

// ---------------------------------------------------------------- 
// MLP: initialize_weights 
// ---------------------------------------------------------------- 
// Initializes the weights of the network with random values. 
// ---------------------------------------------------------------- 
void MLP::initialize_weights() { 
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f); 

    weights_.resize(layers_.size() - 1); 
    for (size_t i = 0; i < weights_.size(); ++i) { 
        int rows = layers_[i].neurons.size() + 1; // +1 for bias 
        int cols = layers_[i+1].neurons.size(); 
        weights_[i].resize(rows, std::vector<float>(cols)); 
        for (auto& row : weights_[i]) { 
            for (auto& w : row) { 
                w = dist(gen); 
            } 
        } 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: forward_pass 
// ---------------------------------------------------------------- 
// Performs a forward pass through the network. 
// ---------------------------------------------------------------- 
void MLP::forward_pass(const std::vector<float>& input) { 
    if (input.size() != layers_[0].neurons.size()) { 
        throw std::invalid_argument("Input size mismatch."); 
    } 
    layers_[0].neurons = input; 

    for (size_t i = 0; i < weights_.size(); ++i) { 
        const auto& prev_layer = layers_[i]; 
        auto& current_layer = layers_[i+1]; 

        for (size_t j = 0; j < current_layer.neurons.size(); ++j) { 
            float potential = weights_[i][0][j]; // Bias 
            for (size_t k = 0; k < prev_layer.neurons.size(); ++k) { 
                potential += prev_layer.neurons[k] * weights_[i][k+1][j]; 
            } 
            current_layer.potentials[j] = potential; 
            current_layer.neurons[j] = apply_activation(potential, current_layer.activation); 
        } 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: backward_pass 
// ---------------------------------------------------------------- 
// Performs a backward pass to calculate the error deltas. 
// ---------------------------------------------------------------- 
void MLP::backward_pass(const std::vector<float>& desired_output) { 
    // Output layer 
    Layer& output_layer = layers_.back(); 
    for (size_t i = 0; i < output_layer.neurons.size(); ++i) { 
        float error = desired_output[i] - output_layer.neurons[i]; 
        output_layer.deltas[i] = error * apply_activation_derivative(output_layer.potentials[i], output_layer.activation); 
    } 

    // Hidden layers 
    for (int i = layers_.size() - 2; i > 0; --i) { 
        const auto& next_layer = layers_[i+1]; 
        auto& current_layer = layers_[i]; 
        const auto& current_weights = weights_[i]; 

        for (size_t j = 0; j < current_layer.neurons.size(); ++j) { 
            float sum = 0.0f; 
            for (size_t k = 0; k < next_layer.neurons.size(); ++k) { 
                sum += next_layer.deltas[k] * current_weights[j+1][k]; 
            } 
            current_layer.deltas[j] = sum * apply_activation_derivative(current_layer.potentials[j], current_layer.activation); 
        } 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: update_weights 
// ---------------------------------------------------------------- 
// Updates the weights of the network using the calculated deltas. 
// ---------------------------------------------------------------- 
void MLP::update_weights(float learning_rate) { 
    for (size_t i = 0; i < weights_.size(); ++i) { 
        const auto& prev_layer = layers_[i]; 
        const auto& current_layer = layers_[i+1]; 

        for (size_t j = 0; j < current_layer.neurons.size(); ++j) { 
            // Update bias weight 
            weights_[i][0][j] += learning_rate * current_layer.deltas[j]; 
            // Update other weights 
            for (size_t k = 0; k < prev_layer.neurons.size(); ++k) { 
                weights_[i][k+1][j] += learning_rate * current_layer.deltas[j] * prev_layer.neurons[k]; 
            } 
        } 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: train 
// ---------------------------------------------------------------- 
// Trains the MLP on the given dataset for a number of epochs. 
// ---------------------------------------------------------------- 
void MLP::train(const std::vector<std::vector<float>>& train_images, 
               const std::vector<int>& train_labels, 
               int epochs, float learning_rate) { 

    for (int epoch = 0; epoch < epochs; ++epoch) { 
        int correct_predictions = 0; 
        for (size_t i = 0; i < train_images.size(); ++i) { 
            forward_pass(train_images[i]); 

            int predicted_digit = predict(train_images[i]); 
            if (predicted_digit == train_labels[i]) { 
                correct_predictions++; 
            } 

            std::vector<float> desired_output(layers_.back().neurons.size(), 0.0f); 
            if (train_labels[i] < desired_output.size()) { 
                desired_output[train_labels[i]] = 1.0f; 
            } 

            backward_pass(desired_output); 
            update_weights(learning_rate); 
            
            if(i > 0 && i % 1000 == 0){ 
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Image " << i << "/" << train_images.size() 
                      << ", Accuracy: " << static_cast<float>(correct_predictions) / (i+1) * 100 << "%" << std::endl; 
            } 
        } 
        std::cout << "Epoch " << epoch + 1 << " Final Accuracy: " 
                  << static_cast<float>(correct_predictions) / train_images.size() * 100 << "%" << std::endl; 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: predict 
// ---------------------------------------------------------------- 
// Predicts the digit for a given image. 
// ---------------------------------------------------------------- 
int MLP::predict(const std::vector<float>& image) { 
    forward_pass(image); 
    const auto& output_neurons = layers_.back().neurons; 
    return std::distance(output_neurons.begin(), std::max_element(output_neurons.begin(), output_neurons.end())); 
} 

// ---------------------------------------------------------------- 
// MLP: predict_from_vector 
// ---------------------------------------------------------------- 
// Predicts the digit from a hand-drawn number vector. 
// ---------------------------------------------------------------- 
int MLP::predict_from_vector(const std::vector<float>& drawing_vector) { 
    if (drawing_vector.size() != layers_[0].neurons.size()) { 
        throw std::invalid_argument("Drawing vector size is not compatible with the input layer."); 
    } 
    return predict(drawing_vector); 
} 

// ---------------------------------------------------------------- 
// MLP: save_model 
// ---------------------------------------------------------------- 
// Saves the model's weights and biases to a text file. 
// ---------------------------------------------------------------- 
void MLP::save_model(const std::string& path) { 
    std::ofstream file(path); 
    if (!file.is_open()) { 
        throw std::runtime_error("Cannot open file for writing: " + path); 
    } 
    
    file << std::fixed << std::setprecision(8); 

    for (size_t i = 0; i < weights_.size(); ++i) { 
        file << "layer " << i << std::endl; 
        // Biases are the first row of the weights matrix 
        file << "biases" << std::endl; 
        for(size_t j = 0; j < weights_[i][0].size(); ++j) { 
            file << weights_[i][0][j] << " "; 
        } 
        file << std::endl; 

        // Weights 
        file << "weights" << std::endl; 
        for (size_t j = 1; j < weights_[i].size(); ++j) { 
            for (size_t k = 0; k < weights_[i][j].size(); ++k) { 
                file << weights_[i][j][k] << " "; 
            } 
            file << std::endl; 
        } 
    } 
} 

// ---------------------------------------------------------------- 
// MLP: load_model 
// ---------------------------------------------------------------- 
// Loads the model's weights and biases from a text file. 
// ---------------------------------------------------------------- 
void MLP::load_model(const std::string& path) { 
    std::ifstream file(path); 
    if (!file.is_open()) { 
        throw std::runtime_error("Cannot open file for reading: " + path); 
    } 

    std::string line; 
    size_t current_layer = -1; 
    bool reading_biases = false; 
    bool reading_weights = false; 
    int weight_row = 1; 

    while (std::getline(file, line)) { 
        std::stringstream ss(line); 
        std::string keyword; 
        ss >> keyword; 

        if (keyword == "layer") { 
            ss >> current_layer; 
            weight_row = 1; 
        } else if (keyword == "biases") { 
            reading_biases = true; 
            reading_weights = false; 
            std::getline(file, line); 
            std::stringstream bias_ss(line); 
            for (size_t i = 0; i < weights_[current_layer][0].size(); ++i) { 
                bias_ss >> weights_[current_layer][0][i]; 
            } 
        } else if (keyword == "weights") { 
            reading_weights = true; 
            reading_biases = false; 
            weight_row = 1; 
            while(weight_row < weights_[current_layer].size() && std::getline(file, line)) { 
                std::stringstream weight_ss(line); 
                for(size_t i = 0; i < weights_[current_layer][weight_row].size(); ++i) { 
                    weight_ss >> weights_[current_layer][weight_row][i]; 
                } 
                weight_row++; 
            } 
        } 
    } 
} 


// ================================================================ 
// Main Function 
// ================================================================ 
int main() { 
    try { 
        std::vector<std::vector<float>> train_images; 
        std::vector<int> train_labels; 
        read_mnist_images("train-images.idx3-ubyte", train_images, 10000); 
        read_mnist_labels("train-labels.idx1-ubyte", train_labels, 10000); 
        
        std::vector<std::vector<float>> test_images; 
        std::vector<int> test_labels; 
        read_mnist_images("t10k-images.idx3-ubyte", test_images, 1000); 
        read_mnist_labels("t10k-labels.idx1-ubyte", test_labels, 1000); 

        MLP mlp({784, 128, 64, 10}, {Activation::ReLU, Activation::ReLU, Activation::Sigmoid}); 

        std::cout << "Training model..." << std::endl; 
        mlp.train(train_images, train_labels, 5, 0.05f); 

        std::cout << "Saving model to my_model.txt..." << std::endl; 
        mlp.save_model("my_model.txt"); 

        std::cout << "Evaluating model on test data..." << std::endl; 
        int correct_predictions = 0; 
        for(size_t i = 0; i < test_images.size(); ++i) { 
            if (mlp.predict(test_images[i]) == test_labels[i]) { 
                correct_predictions++; 
            } 
        } 
        std::cout << "Test accuracy: " << (float)correct_predictions / test_images.size() * 100 << "%" << std::endl; 

        std::cout << "\nLoading model from my_model.txt..." << std::endl; 
        MLP loaded_mlp({784, 128, 64, 10}, {Activation::ReLU, Activation::ReLU, Activation::Sigmoid}); 
        loaded_mlp.load_model("my_model.txt"); 
        
        std::cout << "Evaluating loaded model on test data..." << std::endl; 
        correct_predictions = 0; 
        for(size_t i = 0; i < test_images.size(); ++i) { 
            if (loaded_mlp.predict(test_images[i]) == test_labels[i]) { 
                correct_predictions++; 
            } 
        } 
        std::cout << "Loaded model test accuracy: " << (float)correct_predictions / test_images.size() * 100 << "%" << std::endl; 

        std::cout << "\nPredicting a single image..." << std::endl; 
        int prediction = loaded_mlp.predict_from_vector(test_images[0]); 
        std::cout << "Predicted digit: " << prediction << ", Actual digit: " << test_labels[0] << std::endl; 


    } catch (const std::exception& e) { 
        std::cerr << "Error: " << e.what() << std::endl; 
        return 1; 
    } 

    return 0; 
} 
