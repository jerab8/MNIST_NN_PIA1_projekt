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
	const std::string filePath = "train-labels.idx1-ubyte";
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

inline float apply_activation_derivative(float s, Activation a) {
    switch (a) {
        case Activation::ReLU:
            return s > 0.f ? 1.f : 0.f;

        case Activation::Sigmoid: {
            float y = 1.f / (1.f + std::exp(-s)); // sigmoid(s)
            return y * (1.f - y);
        }

        case Activation::Tanh: {
            float t = std::tanh(s);
            return 1.f - t * t;
        }

        case Activation::Linear:
            return 1.f;
    }
    return 1.f; // fallback
}




///-----------------------FUNCTIONS FOR ARCHITECTURE-----------------------------------

void forward_step(const std::vector<float> &neuron_vec_in,
                                 std::vector<float> &neuron_vec_out,
                                 std::vector<float> &potential_out,
                                 const std::vector<std::vector<float>> &weight_matrix,
                                 Activation act)
{
    const int count_in = (int)weight_matrix.size();         // = 1 (bias) + count of neurons from previous layer
    const int count_out = (int)weight_matrix[0].size();     // = count of neurons of the upper layer being calculated

    for (int k = 0; k < count_out; ++k) {          // column = weights coming into neuron k
        float potential = weight_matrix[0][k];           // bias is in first row weight_matrix[0][k]
        for (int j = 1; j < count_in; ++j) {         // i = 1..in_count
            potential += weight_matrix[j][k] * neuron_vec_in[j - 1];
        }
        potential_out[k] = potential;
        neuron_vec_out[k] = apply_activation(potential, act);
    }
}


// Funkce backpropagace provádí rovnici 11 (latex)
void backward_step(	std::vector<std::vector<float>>& delta_weight_matrix, 	// latex delta w_jk^m
					float eta,                                  			// learning rate   
					const std::vector<float>& delta_km,    					// latex delta_k^m
					const std::vector<float>& y_in)     					// latex y_j^{m-1}                      			                    
{
    const int jmax = (int)y_in.size();
    const int kmax = (int)delta_km.size();
    for (int k = 0; k < kmax; k++) 
    {
		delta_weight_matrix[0][k] = eta * delta_km[k]; // biasy ... j = 0, y = 1
		for (int j = 0; j < jmax; j++)
		{
			delta_weight_matrix[j+1][k] = eta * delta_km[k] * y_in[j];
		}

	}
}

// Funkce vyplní vektor delta v obecné skryté vrstvě - rovnice 12 (latex)
void fill_delta_hidden(	std::vector<float>& delta_km,    						// latex delta_k^m
						const std::vector<float>& delta_lm_out,    				// latex delta_l^m+1
						const std::vector<std::vector<float>>& weight_matrix,   // latex w_kl^m+1
						const std::vector<float>& potential_km,     			// latex s_k^m
						Activation act  )                          			// derivative of ac fc                      
{
    const int lmax = (int)delta_lm_out.size();
    const int kmax = (int)potential_km.size();
    for (int k = 0; k < kmax; k++) 
    {
		float sum_result = 0;
		for (int l = 0; l < lmax; l++)
		{
			sum_result += delta_lm_out[l] * weight_matrix[k+1][l]; // k+1 protoze preskakujeme sloupec biasu
		}
		float derivative = apply_activation_derivative(potential_km[k], act);
		delta_km[k] = sum_result * derivative;

	}
}

// Funkce vyplní vektor delta ve výstupní vrstvě - rovnice 13 (latex)
void fill_delta_output(	std::vector<float>& delta_ko,    						// latex delta_k^o
						const std::vector<float>& desired_ko,    				// latex d_k^o
						const std::vector<float>& y_out,   						// latex y_k^o
						const std::vector<float>& potential_ko,     			// latex s_k^o
						Activation act  )                          			// derivative of ac fc                      
{
    const int kmax = (int)potential_ko.size();
    for (int k = 0; k < kmax; k++) 
    {
		float derivative = apply_activation_derivative(potential_ko[k], act);
		delta_ko[k] = (desired_ko[k] - y_out[k]) * derivative;
	}
}

void update_weights(
    std::vector<std::vector<float>>& weight_in_first,
    std::vector<std::vector<float>>& weight_first_second,
    std::vector<std::vector<float>>& weight_second_out,
    const std::vector<std::vector<float>>& delta_weight_in_first,
    const std::vector<std::vector<float>>& delta_weight_first_second,
    const std::vector<std::vector<float>>& delta_weight_second_out)
{
    // Update input→first hidden layer
    for (size_t i = 0; i < weight_in_first.size(); ++i)
        for (size_t j = 0; j < weight_in_first[i].size(); ++j)
            weight_in_first[i][j] += delta_weight_in_first[i][j];

    // Update first hidden→second hidden layer
    for (size_t i = 0; i < weight_first_second.size(); ++i)
        for (size_t j = 0; j < weight_first_second[i].size(); ++j)
            weight_first_second[i][j] += delta_weight_first_second[i][j];

    // Update second hidden→output layer
    for (size_t i = 0; i < weight_second_out.size(); ++i)
        for (size_t j = 0; j < weight_second_out[i].size(); ++j)
            weight_second_out[i][j] += delta_weight_second_out[i][j];
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
    const float eta = 0.1;  //learning rate

    //inicialization of vectors and weight matrixes (alokate on heap)
    std::vector<float> layer_in(in_count);
    std::vector<float> layer_h1(count_neuron_h1);
    std::vector<float> layer_h2(count_neuron_h2);
    std::vector<float> layer_out(out_count);
    
    std::vector<float> layer_h1_potential(count_neuron_h1);
    std::vector<float> layer_h2_potential(count_neuron_h2);
    std::vector<float> layer_out_potential(out_count);
    
    std::vector<float> layer_h1_delta(count_neuron_h1);
    std::vector<float> layer_h2_delta(count_neuron_h2);
    std::vector<float> layer_out_delta(out_count);
    
    std::vector<float> layer_out_desired(out_count);
    
    std::vector<std::vector<float>> weight_in_first(in_count+1, std::vector<float>(count_neuron_h1));
    std::vector<std::vector<float>> weight_first_second(count_neuron_h1+1, std::vector<float>(count_neuron_h2));
    std::vector<std::vector<float>> weight_second_out(count_neuron_h2+1, std::vector<float>(out_count));
    
    std::vector<std::vector<float>> delta_weight_in_first(in_count+1, std::vector<float>(count_neuron_h1));
    std::vector<std::vector<float>> delta_weight_first_second(count_neuron_h1+1, std::vector<float>(count_neuron_h2));
    std::vector<std::vector<float>> delta_weight_second_out(count_neuron_h2+1, std::vector<float>(out_count));

    // filling weight matrix with random 
    fill_random_matrix(weight_in_first);
    fill_random_matrix(weight_first_second);
    fill_random_matrix(weight_second_out);

    // připravíme vstup - načtení MNIST obrázku s indexem 0
	
	int total_guesses = 0;
	int correct_guesses = 0;
	
	for (int imageIndex = 0; imageIndex < 50000; imageIndex++)
	{
		readMNISTImage(layer_in, imageIndex);
		int desired_digit = readMNISTLabel(imageIndex);
		std::cout << "Desired digit: " << desired_digit << std::endl;
		//std::cout << "Printing how outputs should actually look like" << std::endl;
		get_desired_outputs_from_digit(layer_out_desired, desired_digit);
		//for (int i = 0; i < out_count; ++i) {
		//    std::cout << "y_skut[" << i << "] = " << layer_out_desired[i] << "\n";
		//}

		// forward: input -> H1 (ReLU)
		forward_step(layer_in,  layer_h1, layer_h1_potential, weight_in_first, Activation::ReLU);
		// forward: H1 -> H2 (ReLU)
		forward_step(layer_h1,  layer_h2, layer_h2_potential, weight_first_second, Activation::ReLU);
		// forward: H2 -> OUT (Sigmoid např. pro (0,1))
		forward_step(layer_h2,  layer_out, layer_out_potential, weight_second_out, Activation::Sigmoid);

		// výpis výstupní vrstvy
		//std::cout << "Output layer (size " << out_count << "):\n";
		//std::cout << std::fixed << std::setprecision(6);
		//for (int i = 0; i < out_count; ++i) {
		//    std::cout << "y[" << i << "] = " << layer_out[i] << "\n";
		//}
		
		int digit = get_digit_from_outputs(layer_out);
		std::cout << "chosen digit " << digit << std::endl;
		
		total_guesses++;
		if (desired_digit == digit)
		{
			correct_guesses++;
		}
		
		float success_rate = static_cast<float>(correct_guesses) / static_cast<float>(total_guesses);
		std::cout << "total success rate " << success_rate << std::endl;
		
		// backward: vypocet delta_ko
		fill_delta_output(layer_out_delta, layer_out_desired, layer_out, layer_out_potential, Activation::Sigmoid);
		// backward: OUT -> H2 (Sigmoid)
		backward_step(delta_weight_second_out, eta, layer_out_delta, layer_h2);
		
		// backward: vypocet delta ve druhe skryte vrstve
		fill_delta_hidden(layer_h2_delta, layer_out_delta, weight_second_out, layer_h2_potential, Activation::ReLU);
		// backward: H2 -> H1
		backward_step(delta_weight_first_second, eta, layer_h2_delta, layer_h1);
		
		// backward: vypocet delta v prvni skryte vrstve
		fill_delta_hidden(layer_h1_delta, layer_h2_delta, weight_first_second, layer_h1_potential, Activation::ReLU);
		// backward: H1 -> IN
		backward_step(delta_weight_in_first, eta, layer_h1_delta, layer_in);
		
		update_weights(weight_in_first, weight_first_second, weight_second_out, delta_weight_in_first, delta_weight_first_second, delta_weight_second_out);
	}
    return 0;
}
