# AI-Based Autonomous Driving Using Behavioral Cloning

## Project Overview

This university mini-project explores how a Convolutional Neural Network (CNN) can learn steering behavior from human driving demonstrations in the CARLA simulator.

The system collects driving data, processes camera images, trains a behavioral-cloning model, and evaluates its performance in a simulated driving environment. It also investigates whether image augmentation can improve the model's ability to predict steering angles under different driving conditions.

### Research Objective

**Research question:** How do training data quality and image augmentation influence steering prediction accuracy and autonomous driving performance in a CNN-based behavioral-cloning system?

## Technology Stack

| Component             | Technology                   |
| --------------------- | ---------------------------- |
| Driving Simulator     | CARLA 0.9.16                 |
| Programming Language  | Python 3.12                  |
| Deep Learning         | TensorFlow / Keras           |
| Image Processing      | OpenCV, NumPy                |
| Data Processing       | Pandas                       |
| Visualization         | Matplotlib                   |
| Interactive Dashboard | Streamlit, Plotly (optional) |

## Key Features

* **Human Driving Data Collection:** Record camera images and steering controls while driving manually in CARLA.
* **Autopilot Data Collection:** Generate additional training samples using CARLA's built-in autopilot.
* **CNN-Based Steering Prediction:** Train a neural network to predict steering commands from front-facing camera images.
* **Baseline vs. Augmented Training:** Compare model performance with and without image augmentation.
* **Model Evaluation:** Analyze steering prediction errors and driving performance using evaluation metrics and visualizations.
* **Autonomous Driving:** Test the trained model by allowing it to control a simulated vehicle.
* **Experiment Tracking:** Save training histories, evaluation results, model configurations, and plots for comparison.

## Experimental Methodology

The project evaluates two training approaches using the same dataset and experimental configuration.

| Experiment | Training Approach                 | Purpose                                              |
| ---------- | --------------------------------- | ---------------------------------------------------- |
| Baseline   | Original training images          | Establish reference model performance                |
| Augmented  | Training images with augmentation | Measure the effect of augmentation on generalization |

Both experiments use the same persisted 70/15/15 training, validation, and test split. Data augmentation is applied only to the training set, ensuring that validation and test results remain unaffected by augmentation.

The comparison considers steering prediction accuracy, validation performance, and autonomous driving behavior in CARLA.

## Expected Outcomes

The project aims to determine whether image augmentation improves steering prediction and helps the trained model generalize to different visual conditions.

The results will provide a practical comparison of the baseline and augmented models, highlighting the benefits and limitations o
