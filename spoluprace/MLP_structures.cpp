#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <numeric>
///--------------------UTILITY------------------------------
// --- Funkce co obdrží vektor vstupů a index obrázku; naplní vektor vstupy ---
void readMNISTImage(std::vector<float> &layer_in, int imageIndex) {
	const std::string filePath = "train-images-idx3-ubyte";
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
	const std::string filePath = "train-labels-idx1-ubyte";
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

/*
here are 4 structer for network architacture,
Layer holds everz parametrs for neuron layer
forward propagation holds function an logic for forward cycle
backward propagation holds -//-
MLP put together upper structures
*/

/////////////////////////////
struct Layer
{
    std::vector<float> y;        // výstupy (aktivace)
    std::vector<float> s;        // potenciály
    std::vector<float> delta;    // chybové signály
    Activation act;              // typ aktivační funkce
};

//////////////////////////////////////////

struct Forward_propagation
{
    void forward_step(const std::vector<float> &neuron_vec_in,
                                 std::vector<float> &neuron_vec_out,
                                 std::vector<float> &potential_out,
                                 const std::vector<std::vector<float>> &weight_matrix,
                                 Activation act);   
};

void Forward_propagation::forward_step(const std::vector<float> &neuron_vec_in,
                                 std::vector<float> &neuron_vec_out,
                                 std::vector<float> &potential_out,
                                 const std::vector<std::vector<float>> &weight_matrix,
                                 Activation act)
{
    const int count_in = (int)weight_matrix.size();         // = 1 (bias) + count of neurons from previous layer
    const int count_out = (int)weight_matrix[0].size();     // = count of neurons of the upper layer being calculated

    for (int k = 0; k < count_out; ++k) {          // column = weights coming into neuron k
        float potential = weight_matrix[0][k];           // bias is in first row weight_matrix[0][k]
        for (int j = 1; j < count_in; ++j) {         // j = 1..in_count
            potential += weight_matrix[j][k] * neuron_vec_in[j - 1];
        }
        potential_out[k] = potential;
        neuron_vec_out[k] = apply_activation(potential, act);
    }
}
//////////////////////////////////////////////////
struct Backward_propagation
{
    float eta = 0.1f;

    void backward_step(	std::vector<std::vector<float>>& weight_matrix, 	// latex delta w_jk^m
					float eta,                                  			// learning rate   
					const std::vector<float>& delta_km,    					// latex delta_k^m
					const std::vector<float>& y_in) const;					// latex y_j^{m-1}                      			                    

    void fill_delta_hidden(	std::vector<float>& delta_km,    						// latex delta_k^m
						const std::vector<float>& delta_lm_out,    				// latex delta_l^m+1
						const std::vector<std::vector<float>>& weight_matrix,   // latex w_kl^m+1
						const std::vector<float>& potential_km,     			// latex s_k^m
						Activation act  ) const;                          			// derivative of ac fc                      
    
    void fill_delta_output(	std::vector<float>& delta_ko,    						// latex delta_k^o
						const std::vector<float>& desired_ko,    				// latex d_k^o
						const std::vector<float>& y_out,   						// latex y_k^o
						const std::vector<float>& potential_ko,     			// latex s_k^o
						Activation act  ) const;                          			// derivative of ac fc                      
};


// Funkce backpropagace provádí rovnici 11 (latex)
void Backward_propagation::backward_step(	std::vector<std::vector<float>>& weight_matrix, 	// latex delta w_jk^m
					float eta,                                  			// learning rate   
					const std::vector<float>& delta_km,    					// latex delta_k^m
					const std::vector<float>& y_in) const     					// latex y_j^{m-1}                      			                    
{
    #ifndef NDEBUG
        if ((int)weight_matrix.size() != (int)y_in.size() + 1 ||
            (int)weight_matrix[0].size() != (int)delta_km.size()) {
            throw std::runtime_error("backward_step: dimension mismatch");
        }
    #endif

        const int jmax = (int)y_in.size();
        const int kmax = (int)delta_km.size();

        for (int k = 0; k < kmax; ++k) {
            float e = eta * delta_km[k];
            weight_matrix[0][k] += e;                  // biasy ... j = 0, y = 1
            for (int j = 0; j < jmax; ++j)
                weight_matrix[j+1][k] += e * y_in[j];  // Δw_{jk} = η * δ_k * y_j
        }
    }


// Funkce vyplní vektor delta v obecné skryté vrstvě - rovnice 12 (latex)
void Backward_propagation::fill_delta_hidden(	std::vector<float>& delta_km,    						// latex delta_k^m
						const std::vector<float>& delta_lm_out,    				// latex delta_l^m+1
						const std::vector<std::vector<float>>& weight_matrix,   // latex w_kl^m+1
						const std::vector<float>& potential_km,     			// latex s_k^m
						Activation act  ) const                          			// derivative of ac fc                      
{
    #ifndef NDEBUG
        if ((int)weight_matrix.size() != (int)potential_km.size() + 1 ||
            (int)weight_matrix[0].size() != (int)delta_lm_out.size() ||
            (int)delta_km.size() != (int)potential_km.size()) {
            throw std::runtime_error("fill_delta_hidden: dimension mismatch");
        }
    #endif

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
void Backward_propagation::fill_delta_output(	std::vector<float>& delta_ko,    						// latex delta_k^o
						const std::vector<float>& desired_ko,    				// latex d_k^o
						const std::vector<float>& y_out,   						// latex y_k^o
						const std::vector<float>& potential_ko,     			// latex s_k^o
						Activation act  ) const                          			// derivative of ac fc                      
    {
    #ifndef NDEBUG
        if ((int)delta_ko.size() != (int)potential_ko.size() ||
            (int)desired_ko.size() != (int)potential_ko.size() ||
            (int)y_out.size() != (int)potential_ko.size()) {
            throw std::runtime_error("fill_delta_output: dimension mismatch");
        }
    #endif

        const int kmax = (int)potential_ko.size();
        for (int k = 0; k < kmax; k++) 
        {
            float derivative = apply_activation_derivative(potential_ko[k], act);
            delta_ko[k] = (desired_ko[k] - y_out[k]) * derivative;
        }
    }

/////////////////////////////////////
struct MLP
{   
    int epochs = 5;

    // Topologie
    std::vector<Layer> L;                              // L[0]..L[n-1]
    std::vector<std::vector<std::vector<float>>> W;    // W[i] je (|L[i]|+1)×|L[i+1]|

    // Enginy (jen držáky funkcí)
    Forward_propagation fp;
    Backward_propagation bp;

    // Inicializace topologie a vah
    void init(const std::vector<int>& layer_sizes,
              const std::vector<Activation>& acts);

    // Jeden průchod: dopředně přes všechny vrstvy
    void forward_one_sample();

    // Zpětně: spočti delty všech vrstev
    void backward_build_deltas(const std::vector<float>& desired);

    // Update: přepiš W[i] podle delt a vstupů
    void apply_updates(float eta);

    // Tréninková smyčka
    void train(int epochs, float eta, int n_samples);
};



void MLP::init(const std::vector<int>& layer_sizes,
               const std::vector<Activation>& acts)
{
    /*
    args:   count of neurons in layer,
            act fc for layer 
            (count of arguments = count of layers)

    net.init({784, 64, 32, 10}, 
            {Activation::ReLU, Activation::ReLU, Activation::Sigmoid});
    */
    // --- Kontrola, že počet vrstev odpovídá počtu aktivací ---
    if (layer_sizes.size() != acts.size()) {
        throw std::runtime_error("Počet vrstev a počet aktivací se musí shodovat.");
    }

    const int layer_count = (int)layer_sizes.size();
    L.resize(layer_count);

    // --- Inicializace vrstev ---
    for (int i = 0; i < layer_count; ++i) {
        L[i].y.resize(layer_sizes[i]);      // výstupy neuronů
        L[i].s.resize(layer_sizes[i]);      // potenciály
        L[i].delta.resize(layer_sizes[i]);  // delta chyby
        L[i].act = acts[i];                 // aktivace vrstvy
    }

    // --- Inicializace váhových matic ---
    W.resize(layer_count - 1);  // mezi každými dvěma vrstvami je jedna matice

    for (int i = 0; i < layer_count - 1; ++i) {
        const int in_size  = layer_sizes[i];
        const int out_size = layer_sizes[i + 1];

        // +1 pro bias
        W[i].assign(in_size + 1, std::vector<float>(out_size));

        fill_random_matrix(W[i]);
    }
}

void MLP::forward_one_sample()
{
    for (size_t i = 0; i + 1 < L.size(); ++i)
    {
        fp.forward_step(
            L[i].y,         // neuron_vec_in
            L[i+1].y,       // neuron_vec_out
            L[i+1].s,       // potential_out
            W[i],           // weight_matrix (const&)
            L[i+1].act      // Activation
        );
    }
}


void MLP::backward_build_deltas(const std::vector<float>& desired)
{
    // o = index výstupní vrstvy
    const int o = static_cast<int>(L.size()) - 1;

    // 1) výstupní delty: δ^out = (d - y) ⊙ φ'(s)
    bp.fill_delta_output(
        L[o].delta,     // delta_ko
        desired,        // desired_ko
        L[o].y,         // y_out
        L[o].s,         // potential_ko
        L[o].act
    );

    // 2) skryté delty shora dolů: i = o-1, ..., 1
    for (int i = o - 1; i >= 1; --i) {
        bp.fill_delta_hidden(
            L[i].delta,     // delta_km
            L[i+1].delta,   // delta_lm_out
            W[i],           // weight_matrix (K+1 x L), řádek 0 = bias
            L[i].s,         // potential_km
            L[i].act
        );
    }
}
void MLP::apply_updates(float eta)
{
    const int o = static_cast<int>(L.size()) - 1; // poslední vrstva = L[o]
    for (int i = 0; i < o; ++i) {
        // W[i]: (|L[i]|+1) x |L[i+1]|
        // delta cílové vrstvy = L[i+1].delta, vstup = L[i].y
        bp.backward_step(W[i], eta, L[i+1].delta, L[i].y);
    }
}



void MLP::train(int epochs, float eta, int n_samples)
{
    std::vector<float> desired(L.back().y.size(), 0.f);

    std::vector<int> order(n_samples);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng{std::random_device{}()};

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(order.begin(), order.end(), rng);
        int correct = 0;

        for (int t = 0; t < n_samples; ++t) {
            const int idx = order[t];

            // 1) načti vstup a label přímo
            readMNISTImage(L[0].y, idx);
            int label = readMNISTLabel(idx);
            get_desired_outputs_from_digit(desired, label);

            // 2) forward
            forward_one_sample();

            // 3) metrika (bez dalších závislostí)
            int pred = get_digit_from_outputs(L.back().y);
            if (pred == label) ++correct;

            // 4) backprop (delty) + update
            backward_build_deltas(desired);
            apply_updates(eta);
        }

        std::cout << "epoch " << e << " acc=" << (float)correct / n_samples << "\n";
    }
}
int main() {
    try {
        MLP net;
        //Activation::Linear is not aplied in reality. musel jsem jenom splnit podminku abz to jelo. musi se ylepsit
        net.init({784, 64, 32, 10},
                 {Activation::Linear, Activation::ReLU, Activation::ReLU, Activation::Sigmoid});

        net.train(/*epochs=*/8, /*eta=*/0.05f, /*n_samples=*/6000);

        std::cout << "Trénink dokončen úspěšně.\n";
    } catch (const std::exception& e) {
        std::cerr << "Chyba: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
