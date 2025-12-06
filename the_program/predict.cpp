#include "MLP.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Function to print image as ASCII art
void print_ascii_art(const Vector &image) {
  const std::string chars = " .:-=+*#%@";
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      float val = image[i * 28 + j];
      int char_idx = static_cast<int>(val * (chars.size() - 1));
      std::cout << chars[char_idx]
                << chars[char_idx]; // Double char for aspect ratio
    }
    std::cout << "\n";
  }
}

// Helper to check if file exists
bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

// Helper to load and resize image from text file (ASCII art or floats)
// Detects if file contains floats or ASCII art.
Vector load_and_resize_image(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  // Read entire file into memory to inspect content
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

  if (content.empty()) {
    throw std::runtime_error("File is empty: " + path);
  }

  // Check if content looks like floats (digits, dots, whitespace)
  // or ASCII art (other characters).

  std::vector<std::string> lines;
  std::stringstream ss(content);
  std::string line;
  int max_width = 0;
  while (std::getline(ss, line)) {
    // Remove trailing carriage return if present
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(line);
    if (line.length() > max_width)
      max_width = line.length();
  }

  int height = lines.size();
  int width = max_width;

  // Try to parse as floats first (backward compatibility)
  try {
    std::stringstream ss_float(content);
    Vector float_vals;
    float v;
    while (ss_float >> v) {
      float_vals.push_back(v);
    }
    if (float_vals.size() == 784) {
      // It's a valid float vector
      return float_vals; // Already 28x28
    }
    int dim = static_cast<int>(std::sqrt(float_vals.size()));
    if (dim * dim == float_vals.size() && float_vals.size() > 0) {
      // Reuse the resizing logic for floats...
    }
  } catch (...) {
    // Ignore
  }

  // ASCII Art Parsing
  // Map: ' ' -> 0, anything else -> 1
  if (width == 0 || height == 0) {
    return Vector(784, 0.0f);
  }

  Vector raw_values(width * height, 0.0f);
  int min_x = width, max_x = -1;
  int min_y = height, max_y = -1;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      char c = (x < lines[y].length()) ? lines[y][x] : ' ';
      float val = (c != ' ') ? 1.0f : 0.0f;
      raw_values[y * width + x] = val;

      if (val > 0.5f) {
        if (x < min_x)
          min_x = x;
        if (x > max_x)
          max_x = x;
        if (y < min_y)
          min_y = y;
        if (y > max_y)
          max_y = y;
      }
    }
  }

  // If image is empty (all whitespace), return empty
  if (max_x == -1) {
    return Vector(784, 0.0f);
  }

  // Define bounding box dimensions
  int bb_width = max_x - min_x + 1;
  int bb_height = max_y - min_y + 1;

  // Make the bounding box square to preserve aspect ratio
  int max_dim = std::max(bb_width, bb_height);

  // Add 25% padding to each side (total 50% increase in dimension)
  // New dimension = max_dim + 0.25*max_dim + 0.25*max_dim = 1.5 * max_dim
  float padded_dim = max_dim * 1.5f;

  // Center the square on the original bounding box
  float center_x = min_x + bb_width / 2.0f;
  float center_y = min_y + bb_height / 2.0f;

  float start_x = center_x - padded_dim / 2.0f;
  float start_y = center_y - padded_dim / 2.0f;

  // Resize square bounding box to 28x28
  Vector resized(28 * 28, 0.0f);

  float scale = padded_dim / 28.0f;

  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      // Map target (x,y) to source square coordinates
      float start_src_x = start_x + x * scale;
      float start_src_y = start_y + y * scale;
      float end_src_x = start_x + (x + 1) * scale;
      float end_src_y = start_y + (y + 1) * scale;

      float sum = 0.0f;
      int count = 0;

      for (int sy = static_cast<int>(start_src_y); sy < std::ceil(end_src_y);
           ++sy) {
        for (int sx = static_cast<int>(start_src_x); sx < std::ceil(end_src_x);
             ++sx) {
          // Check bounds (padding with 0 if outside original image)
          if (sy >= 0 && sy < height && sx >= 0 && sx < width) {
            sum += raw_values[sy * width + sx];
          }
          // We count all pixels in the source block, even if out of bounds
          // (effectively 0 padding)
          count++;
        }
      }

      if (count > 0) {
        resized[y * 28 + x] = sum / count;
      }
    }
  }

  return resized;
}

inline bool command_exists(const std::string &cmd) {
  std::string check_cmd = "which " + cmd + " > /dev/null 2>&1";
  return std::system(check_cmd.c_str()) == 0;
}

int main() {
  try {
    MLP net;
    std::cout << "Loading model from model.bin...\n";
    net.load("model.bin");
    std::cout << "Model loaded.\n";

    Vector input_image(784);
    int true_label = -1;
    // Check for human_testing.txt in datasets folder
    std::string human_testing_path = "../datasets/human_testing.txt";
    bool has_human_file = file_exists(human_testing_path);
    bool has_draw_cmd = command_exists("draw");

    std::cout << "Select an option:\n";
    if (has_human_file) {
      std::cout << "1. Use existing 'human_testing.txt' file\n";
    } else {
      std::cout << "1. [Disabled] 'human_testing.txt' file not found in "
                   "../datasets/\n";
    }

    if (has_draw_cmd) {
      std::cout << "2. Run 'draw' command to create new input\n";
    } else {
      std::cout << "2. [Disabled] 'draw' command not found (try 'sudo apt "
                   "install draw')\n";
    }

    std::cout << "3. Use random image from validation set\n";
    std::cout << "Choice: ";

    int choice;
    if (!(std::cin >> choice)) {
      std::cerr << "Invalid input.\n";
      return 1;
    }

    if (choice == 1) {
      if (!has_human_file) {
        std::cerr << "Option 1 is not available.\n";
        return 1;
      }
      std::cout << "Reading from 'human_testing.txt'...\n";
      input_image = load_and_resize_image(human_testing_path);

    } else if (choice == 2) {
      if (!has_draw_cmd) {
        std::cerr << "Option 2 is not available. 'draw' command not found.\n";
        return 1;
      }
      std::cout << "Running 'draw' command...\n";
      int ret = std::system("draw");
      if (ret != 0) {
        std::cerr << "Error running 'draw' command.\n";
      }

      if (file_exists("/tmp/draw.txt")) {
        std::cout << "Found /tmp/draw.txt. Moving and processing...\n";
        // Move/Copy to datasets directory
        std::ifstream src("/tmp/draw.txt", std::ios::binary);
        std::ofstream dst(human_testing_path, std::ios::binary);
        dst << src.rdbuf();
        src.close();
        dst.close();

        input_image = load_and_resize_image(human_testing_path);

      } else {
        std::cerr << "Error: /tmp/draw.txt not found after running 'draw'.\n";
        return 1;
      }
    } else if (choice == 3) {
      std::cout << "Selecting random image from test set...\n";
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, 9999);
      int idx = dis(gen);

      // Use test set files (assuming they are in parent directory)
      const std::string test_images = "../datasets/t10k-images-idx3-ubyte";
      const std::string test_labels = "../datasets/t10k-labels-idx1-ubyte";

      readMNISTImage(input_image, idx, test_images);
      true_label = readMNISTLabel(idx, test_labels);
      std::cout << "Selected image index: " << idx << "\n";
    } else {
      std::cerr << "Invalid choice.\n";
      return 1;
    }

    // Visualize
    std::cout << "\nInput Image:\n";
    print_ascii_art(input_image);

    if (true_label != -1) {
      std::cout << "\nTrue Label: " << true_label << "\n";
    }

    // Predict
    net.L[0].y = input_image;
    net.forward_one_sample();

    int predicted_label = get_digit_from_outputs(net.L.back().y);

    std::cout << "Model Prediction: " << predicted_label << "\n";

    // Print probabilities
    std::cout << "Probabilities:\n";
    for (int i = 0; i < 10; ++i) {
      std::cout << i << ": " << net.L.back().y[i] << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
