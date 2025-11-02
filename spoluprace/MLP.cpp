#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
///--------------------UTILITY------------------------------
// --- Funkce co obdrží vektor vstupů a index obrázku; naplní vektor vstupy ---
void readMNISTImage(std::vector<float> &layer_in, int imageIndex) {
	const std::string filePath = "train-images.idx3-ubyte";
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }
    // Skip the first 16 bytes of metadata (magic number + number of images + rows + cols)
    file.seekg(16 + imageIndex * 28 * 28, std::ios::beg);

    // Temporary buffer to read raw uint8 pixels
    std::vector<uint8_t> temp(28 * 28);
    file.read(reinterpret_cast<char*>(temp.data()), temp.size());

    // Convert to float in range [0, 1]
    for (int i = 0; i < 28 * 28; i++)
        layer_in[i] = temp[i] / 255.0f;
}

// --- Funkce co obdrží index labelu a vrátí int label
int readMNISTLabel(int labelIndex) {
	const std::string filePath = "t10k-labels.idx1-ubyte";
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    // Skip the first 8 bytes (metadata)
    file.seekg(8 + labelIndex, std::ios::beg);

    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);

    return static_cast<int>(label);
}


// --- Funkce pro naplnění matice náhodnými čísly z intervalu [0, 1] ---
void fill_random_matrix(std::vector<std::vector<float>> &mat) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &row : mat)
        for (auto &v : row)
            v = dist(gen);
}


///-----------------------ACTIVATION FUNCTIONS/LOSS FUNCTIONS/DERIVATIONS-----------------------
enum class Activation { ReLU, Sigmoid, Tanh, Linear };
//inline add fc into code while compilation-should be fast
inline float apply_activation(float s, Activation a) {
    switch (a) {
        case Activation::ReLU:    return s > 0.f ? s : 0.f;
        case Activation::Sigmoid: return 1.f / (1.f + std::exp(-s));
        case Activation::Tanh:    return std::tanh(s);
        case Activation::Linear:  return s;
    }
    return s; // fallback
}
//derivation of activation functions
enum class div_Activation { ReLU, Sigmoid, Tanh, Linear };

inline float apply_activation_derivative(float s, div_Activation a) {
    switch (a) {
        case div_Activation::ReLU:
            return s > 0.f ? 1.f : 0.f;

        case div_Activation::Sigmoid: {
            float y = 1.f / (1.f + std::exp(-s)); // sigmoid(s)
            return y * (1.f - y);
        }

        case div_Activation::Tanh: {
            float t = std::tanh(s);
            return 1.f - t * t;
        }

        case div_Activation::Linear:
            return 1.f;
    }
    return 1.f; // fallback
}




///-----------------------FUNCTIONS FOR ARCHITECTURE-----------------------------------

void forward_step(const std::vector<float> &neuron_vec_in,
                                 std::vector<float> &neuron_vec_out,
                                 std::vector<float> &potencial_out,
                                 const std::vector<std::vector<float>> &weight_matrix,
                                 Activation act)
{
    const int count_in = (int)weight_matrix.size();         // = 1 (bias) + count of neurons from previous layer
    const int count_out = (int)weight_matrix[0].size();     // = count of neurons of the upper layer being calculated

    for (int k = 0; k < count_out; ++k) {          // column = weights coming into neuron k
        float potencial = weight_matrix[0][k];           // bias is in first row weight_matrix[0][k]
        for (int j = 1; j < count_in; ++j) {         // i = 1..in_count
            potencial += weight_matrix[j][k] * neuron_vec_in[j - 1];
        }
        potencial_out[k] = potencial;
        neuron_vec_out[k] = apply_activation(potencial, act);
    }
}


//TODO: repair. this will not work
//vec names are given from left to right architecture point of view
void backward_step_output(const std::vector<float>& right_neuron_vec,   // output
                          const std::vector<float>& left_neuron_vec,    // hiden layer
                          std::vector<std::vector<float>>& weight_matrix, // weights
                          const std::vector<float>& label,              // labels
                          const std::vector<float>& potencial,          // argument values 
                          float n,                                      // learning rate
                          div_Activation act,                               // derivative of ac fc
                          std::vector<float>& delta_out)                // for other hidden layers
{
    const int n_out = (int)right_neuron_vec.size();      // počet výstupních neuronů = řádky W
    const int n_inp = (int)left_neuron_vec.size();       // počet vstupů = (sloupce W - 1)

    delta_out.resize(n_out);

    // 1) spočti δ_k^L = (d_k - y_k) * f'(s_k)
    for (int k = 0; k < n_out; ++k) {
        float d_k = label[k];
        float y_k = right_neuron_vec[k];
        float s_k = potencial[k];
        float dact = apply_activation_derivative(s_k, act);
        delta_out[k] = (d_k - y_k) * dact;
    }

    // 2) update vah: w_{k,0} (bias) a w_{k,i}, i=1..n_inp
    for (int k = 0; k < n_out; ++k) {
        // bias (i = 0, vstup y_0 = 1)
        weight_matrix[k][0] += n * delta_out[k] * 1.0f;

        // běžné vstupy (i = 1..n_inp), vstup je y_{i-1} z levé vrstvy
        for (int i = 1; i <= n_inp; ++i) {
            float y_in = left_neuron_vec[i - 1];
            weight_matrix[k][i] += n * delta_out[k] * y_in;
        }
    }
}

// ---- dostane vektor outputů, vrátí jedinou číslici
int get_digit_from_outputs(const std::vector<float>& outputs)
{
    return std::distance(outputs.begin(), 
                         std::max_element(outputs.begin(), outputs.end()));
}

// ---- dostane jedinou číslici, naplní vektor správných výstupů
void get_desired_outputs_from_digit(std::vector<float>& desired_outputs, int digit)
{
	std::fill(desired_outputs.begin(), desired_outputs.end(), 0.0f);
	desired_outputs[digit] = 1;
}



int main() {
    // --- Inicializace vrstev ---
    int in_count = 784;
    int out_count = 10;
    int count_neuron_h1 = 64;
    int count_neuron_h2 = 32;

    //inicialization of vectors and weight matrixes (alokate on heap)
    std::vector<float> layer_in(in_count);
    std::vector<float> layer_h1(count_neuron_h1);
    std::vector<float> layer_h2(count_neuron_h2);
    std::vector<float> layer_out(out_count);
    
    std::vector<float> layer_h1_potencial(count_neuron_h1);
    std::vector<float> layer_h2_potencial(count_neuron_h2);
    std::vector<float> layer_out_potencial(out_count);
    
    std::vector<float> layer_out_desired(out_count);
    
    std::vector<std::vector<float>> weight_in_first(in_count+1, std::vector<float>(count_neuron_h1));
    std::vector<std::vector<float>> weight_first_second(count_neuron_h1+1, std::vector<float>(count_neuron_h2));
    std::vector<std::vector<float>> weight_second_out(count_neuron_h2+1, std::vector<float>(out_count));

    // filling weight matrix with random 
    fill_random_matrix(weight_in_first);
    fill_random_matrix(weight_first_second);
    fill_random_matrix(weight_second_out);

    //----------------------CHATGPT(vypis abych videl ze to neco dela)
    // ---- JEDEN FORWARD PRŮCHOD + VÝPIS ----
    // připravíme vstup - načtení MNIST obrázku s indexem 0

    readMNISTImage(layer_in, 0);
    int desired_digit = readMNISTLabel(0);
    std::cout << "Desired digit: " << desired_digit << std::endl;
    std::cout << "Printing how outputs should actually look like" << std::endl;
    get_desired_outputs_from_digit(layer_out_desired, desired_digit);
	for (int i = 0; i < out_count; ++i) {
        std::cout << "y_skut[" << i << "] = " << layer_out_desired[i] << "\n";
    }

    // forward: input -> H1 (ReLU)
    forward_step(layer_in,  layer_h1, layer_h1_potencial, weight_in_first, Activation::ReLU);
    // forward: H1 -> H2 (ReLU)
    forward_step(layer_h1,  layer_h2, layer_h2_potencial, weight_first_second, Activation::ReLU);
    // forward: H2 -> OUT (Sigmoid např. pro (0,1))
    forward_step(layer_h2,  layer_out, layer_out_potencial, weight_second_out, Activation::Sigmoid);

    // výpis výstupní vrstvy
    std::cout << "Output layer (size " << out_count << "):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < out_count; ++i) {
        std::cout << "y[" << i << "] = " << layer_out[i] << "\n";
    }
	
	int digit = get_digit_from_outputs(layer_out);
	std::cout << "chosen digit " << digit << std::endl;
	

    return 0;
}
