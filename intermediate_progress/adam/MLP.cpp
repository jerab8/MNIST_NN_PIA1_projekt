#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
///--------------------UTILITY------------------------------
// --- Funkce pro čtení 32-bit celého čísla z big-endian ---
int readInt(std::ifstream &f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3];
}

// --- Funkce pro načtení jednoho obrázku MNIST ---
std::vector<std::vector<uint8_t>> readMNISTImage(std::ifstream &f, int rows, int cols) {
    std::vector<std::vector<uint8_t>> image(rows, std::vector<uint8_t>(cols));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            unsigned char pixel;
            f.read((char*)&pixel, 1);
            image[r][c] = pixel;
        }
    return image;
}

// --- Převod matice na 1D normalizovaný vektor ---
std::vector<float> transform_matrix_to_1D_normalized(const std::vector<std::vector<uint8_t>> &matrix) {
    std::vector<float> vector1D;
    for (auto &row : matrix)
        for (auto &pixel : row)
            vector1D.push_back(pixel / 255.0f);
    return vector1D;
}
// --- Funkce pro naplnění vektoru náhodnými čísly z intervalu [0, 1] ---
void fill_random_vector(std::vector<float> &vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto &v : vec)
        v = dist(gen);
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

std::vector<float>& forward_step(const std::vector<float> &neuron_vec_in,
                                 std::vector<float> &neuron_vec_out,
                                 const std::vector<std::vector<float>> &weight_matrix,
                                 std::vector<float> &potencial_out,
                                Activation act)
{
    const int neurons = (int)weight_matrix.size();        // = count of neurons
    const int cols    = (int)weight_matrix[0].size();     // = 1 (bias) + count of input neurons 

    neuron_vec_out.resize(neurons);
    potencial_out.resize(neurons);

    for (int k = 0; k < neurons; ++k) {          // row = neuron k
        float potencial = weight_matrix[k][0];           // bias is in first column weight_matrix[k][0]
        for (int i = 1; i < cols; ++i) {         // i = 1..in_count
            potencial += weight_matrix[k][i] * neuron_vec_in[i - 1];
        }
        potencial_out[k] = potencial;
        neuron_vec_out[k] = apply_activation(potencial, act);
    }
    return neuron_vec_out;
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







int main() {
    // --- Inicializace vrstev ---
    int in_count = 576;
    int out_count = 10;
    int count_neuron_h1 = 64;
    int count_neuron_h2 = 32;

    //inicialization of vectors and weight matrixes (alokate on heap)
    std::vector<float> layer_h1(count_neuron_h1);
    std::vector<float> layer_h2(count_neuron_h2);
    std::vector<float> layer_out(out_count);

    std::vector<std::vector<float>> weight_in_first(count_neuron_h1, std::vector<float>(in_count+1));
    std::vector<std::vector<float>> weight_first_second(count_neuron_h2, std::vector<float>(count_neuron_h1+1));
    std::vector<std::vector<float>> weight_second_out(out_count, std::vector<float>(count_neuron_h2+1));

    // filling vectors and matrix with random 
    fill_random_vector(layer_h1);
    fill_random_vector(layer_h2);
    fill_random_vector(layer_out);

    fill_random_matrix(weight_in_first);
    fill_random_matrix(weight_first_second);
    fill_random_matrix(weight_second_out);





    //----------------------CHATGPT(vypis abych videl ze to neco dela)
    // ---- JEDEN FORWARD PRŮCHOD + VÝPIS ----
    // připravíme vstup (zatím náhodný), velikosti in_count
    std::vector<float> layer_in(in_count);
    fill_random_vector(layer_in);

    // forward: input -> H1 (ReLU)
    forward_step(layer_in,  layer_h1,  weight_in_first,    Activation::ReLU);
    // forward: H1 -> H2 (ReLU)
    forward_step(layer_h1,  layer_h2,  weight_first_second, Activation::ReLU);
    // forward: H2 -> OUT (Sigmoid např. pro (0,1))
    forward_step(layer_h2,  layer_out, weight_second_out,  Activation::Sigmoid);

    // výpis výstupní vrstvy
    std::cout << "Output layer (size " << out_count << "):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < out_count; ++i) {
        std::cout << "y[" << i << "] = " << layer_out[i] << "\n";
    }



    return 0;
}