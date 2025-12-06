#include "MLP.hpp"
#include <fstream>
#include <iostream>

// Helper class to redirect output to both cout and a file
///////////////////////////////////////////////////////////
class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *sb1, std::streambuf *sb2) : sb1(sb1), sb2(sb2) {}

protected:
  virtual int overflow(int c) {
    if (c == EOF) {
      return !EOF;
    } else {
      int const r1 = sb1->sputc(c);
      int const r2 = sb2->sputc(c);
      return r1 == EOF || r2 == EOF ? EOF : c;
    }
  }

  virtual int sync() {
    int const r1 = sb1->pubsync();
    int const r2 = sb2->pubsync();
    return r1 == 0 && r2 == 0 ? 0 : -1;
  }

private:
  std::streambuf *sb1;
  std::streambuf *sb2;
};
///////////////////////////////////////////////////////////

int main() {
  // Open log file
  std::ofstream log_file("last_trained_model.log");
  // Save original cout buffer
  std::streambuf *original_cout_buffer = std::cout.rdbuf();
  // Create TeeBuf
  TeeBuf tee_buf(original_cout_buffer, log_file.rdbuf());
  // Redirect cout
  std::cout.rdbuf(&tee_buf);

  try {
    // Define training parameters here
    const int epochs = 15;
    const float learning_rate = 0.05f;
    const int n_samples = 60000;

    MLP net;
    net.init({784, 64, 32, 10},
             {Activation::ReLU, Activation::ReLU, Activation::Sigmoid});

    std::cout << "Starting training with:\n"
              << "Epochs: " << epochs << "\n"
              << "Learning Rate: " << learning_rate << "\n"
              << "Samples: " << n_samples << "\n";

    const std::string dataset_dir = "../datasets/";
    const std::string train_images = "train-images-idx3-ubyte";
    const std::string train_labels = "train-labels-idx1-ubyte";
    const std::string test_images = "t10k-images-idx3-ubyte";
    const std::string test_labels = "t10k-labels-idx1-ubyte";

    net.train(epochs, learning_rate, n_samples, dataset_dir + train_images,
              dataset_dir + train_labels);

    std::cout << "Training finished successfully.\n";

    // Save the model
    net.save("model.bin");
    std::cout << "Model saved to model.bin\n";

    // Testing
    std::cout
        << "Running testing on the full testing dataset (10000 images)...\n";
    net.test(dataset_dir + test_images, dataset_dir + test_labels, 10000);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    // Restore buffer before exiting
    std::cout.rdbuf(original_cout_buffer);
    return 1;
  }

  // Restore buffer
  std::cout.rdbuf(original_cout_buffer);
  return 0;
}
