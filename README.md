# 🚗 Self-Driving Car Simulation

A comprehensive Python project for training and testing autonomous vehicle simulations using deep learning and computer vision.


![Photo](https://github.com/Lyth40/Self-Driving-Simulation-using-NVIDIA-s-Model/Model.png)

## 📋 Overview

This project implements a self-driving car simulator with capabilities for:
- **Training simulations** to train neural network models
- **Testing simulations** to evaluate model performance
- **Real-time image processing** using computer vision techniques
- **Deep learning models** built with TensorFlow

## ✨ Features

- 🤖 **Neural Network Training** - Train deep learning models to drive autonomously
- 🎮 **Interactive Simulation** - Real-time driving simulation with visual feedback
- 📊 **Data Processing** - Image augmentation and data preparation
- 📈 **Performance Analysis** - Monitor and evaluate model accuracy
- 🔄 **Model Persistence** - Save and load trained models

## 📁 Project Structure

```
self-driving-car/
├── TrainingSim.py          # Training simulation environment
├── TestSimulation.py       # Test and evaluation environment
├── utlis.py               # Utility functions
├── model.h5               # Pre-trained model weights
├── my_data/               # Training data directory
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/self-driving-car.git
cd self-driving-car
```

2. **Create a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training the Model

Run the training simulation to train the neural network:

```bash
python TrainingSim.py
```

This will:
- Load training data from `my_data/`
- Train the deep learning model
- Save the trained model to `model.h5`
- Display real-time training progress

### Testing the Model

Run the test simulation to evaluate the trained model:

```bash
python TestSimulation.py
```

This will:
- Load the pre-trained model from `model.h5`
- Run the simulation in test mode
- Display model predictions and performance metrics

## 📦 Dependencies

Key libraries used in this project:

- **TensorFlow** - Deep learning framework
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning utilities
- **Matplotlib** - Data visualization
- **Flask** - Web framework for simulation interface
- **imgaug** - Image augmentation library

For a complete list, see `requirements.txt`.

## 🧠 Model Architecture

The neural network model processes:
- Real-time camera input from the simulation
- Road conditions and environmental data
- Outputs steering, acceleration, and braking commands

Pre-trained weights are available in `model.h5`.

## 📊 Data Format

Training data should be organized in the `my_data/` directory with:
- **Images** - Camera frames from the simulation
- **Labels** - Corresponding driving actions (steering angle, speed, etc.)

## 🔧 Configuration

Key configuration options can be modified in the source files:
- Model architecture in training code
- Simulation parameters
- Data augmentation settings

## 📈 Performance

The model achieves:
- Real-time prediction on standard hardware
- Continuous driving control in simulation
- Adaptive learning from simulation feedback


