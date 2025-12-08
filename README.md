# MNIST Neural Network Project (PIA1)

**Authors:** Adam Bednář, Jan Jeřábek, Štěpán Müller

This project contains a C++ implementation of a Multi-Layer Perceptron (MLP) built from scratch to classify handwritten digits from the MNIST dataset. It features a custom neural network library (`MLP.hpp`), a training pipeline with logging (`train.cpp`), and an interactive prediction tool (`predict.cpp`).

## File Structure

The repository is organized as follows (excluding intermediate/build files):

```
.
├── README.md                  # Project documentation
├── datasets/                  # Directory for MNIST data and input files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   └── human_testing.txt      # Input file for manual testing
└── the_program/               # Source code directory
    ├── CMakeLists.txt         # CMake build configuration
    ├── MLP.hpp                # Core Neural Network implementation (header-only)
    ├── train.cpp              # Training executable source
    └── predict.cpp            # Prediction executable source
```

## Core Components

### 1. `MLP.hpp` (Neural Network Engine)
This header file contains the complete implementation of the Multi-Layer Perceptron.
- **Matrix Struct**: Optimized matrix operations with cache-friendly memory access loop swapping.
- **Propagations**: Separate structures for Forward and Backward propagation logic.
- **Architecture**: Flexible layer sizes and activation functions (ReLU, Sigmoid, Tanh, Linear).
- **Serialization**: Custom binary format to save and load trained models (`.bin`).

### 2. `train.cpp` (Training Pipeline)
This program handles the training process of the neural network.
- **Configuration**: Uses a TeeBuf class to split output to both the console and a log file (`last_trained_model.log`).
- **Process**:
    1. Initializes the network topology.
    2. Loads MNIST training data from `../datasets/`.
    3. Trains for a specified number of epochs (default: 15).
    4. Evaluates accuracy on the test set.
    5. Saves the final model to `model.bin`.

### 3. `predict.cpp` (Interactive Prediction)
This tool allows users to test the trained model on various inputs. It features an intelligent image processing pipeline:
- **ASCII/Draw Input**: Can read "drawn" inputs (ASCII art style) from `human_testing.txt` or system `draw` commands.
- **Smart Padding**: Automatically detects the bounding box of the drawn content and adds ~25% padding to each side before resizing to the 28x28 network input. This ensures drawn digits are centered and sized similarly to the MNIST training data (zooming out effect).
- **Interactive Menu**:
    1. **Existing File**: Reads from `datasets/human_testing.txt`.
    2. **Draw Command**: Launches system `draw` tool (if available) to create new input.
    3. **Random Test**: Picks a random image from the MNIST validation set to verify performance.

## Compilation and Usage

Prerequisites: CMake and a C++ compiler supporting C++11 or later.

### 1. Build the Project
Navigate to the source directory and build using CMake:

```bash
cd the_program
cmake .
make
```

### 2. Train the Model
Run the training executable. This will generate `model.bin` and `last_trained_model.log`.

```bash
./train
```

*Note: Ensure the MNIST dataset files are located in `../datasets/` relative to the executable.*

### 3. Run Predictions
Run the prediction tool to interactively test the model.

```bash
./predict
```

Follow the on-screen menu to select your input method.

### 4. Customizing Architecture
To change the network structure (e.g., number of hidden layers or neurons), edit the `net.init` call in `train.cpp` before recompiling:

```cpp
// Example: 784 input, 2 hidden layers (64, 32 neurons), 10 output
net.init({784, 64, 32, 10}, 
         {Activation::ReLU, Activation::ReLU, Activation::Sigmoid});
```
