#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>

std::vector<uint8_t> loadImage(const std::string& filePath, int imageIndex) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    // Skip the first 16 bytes of metadata (magic number + number of images + rows + cols)
    file.seekg(16 + imageIndex * 28 * 28, std::ios::beg); // MNIST images are 28x28

    std::vector<uint8_t> image(28 * 28);
    file.read(reinterpret_cast<char*>(image.data()), image.size());

    return image; // Flattened row-major vector
}

uint8_t loadLabel(const std::string& filePath, int index) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");

    // Skip the first 8 bytes (metadata)
    file.seekg(8 + index, std::ios::beg);

    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);

    return label;
}

int main() {
	auto image = loadImage("train-images.idx3-ubyte", 0);
	std::cout << "Loaded image with " << image.size() << " pixels." << std::endl;
	for (int i = 0; i < 784; i++) {
		std::cout << (int)image[i] << " ";
	}
	std::cout << std::endl;

	auto label = loadLabel("t10k-labels.idx1-ubyte", 0);
	std::cout << "Label = " << (int)label << std::endl;

	return 0;
    }

