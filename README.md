# Electrical-energy-output-prediction

# Electrical Energy Output Prediction of Combined Cycle Power Plant Using ANN

This project predicts the electrical energy output of a Combined Cycle Power Plant (CCPP) using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

The dataset contains sensor readings collected from a real-world CCPP, and the goal is to accurately predict the energy output based on environmental conditions.

---

## Project Structure

├── ANN.ipynb # Jupyter Notebook with full code for model development
├── Folds5x2_pp.xlsx # Dataset used for training and testing
├── README.md # Project description and instructions

yaml
Copy
Edit

---

## Dataset Description

- **Source:** UCI Machine Learning Repository  
- **File:** `Folds5x2_pp.xlsx`  
- **Features:**
  - AT - Ambient Temperature (°C)
  - V - Exhaust Vacuum (cm Hg)
  - AP - Ambient Pressure (millibar)
  - RH - Relative Humidity (%)
- **Target:**
  - PE - Electrical Energy Output (MW)

The dataset contains 9568 samples of daily average sensor measurements and corresponding energy output.

---

## Technologies Used

- Python 3.x
- TensorFlow 2.x
- Keras (Sequential API)
- Scikit-learn
- Pandas & NumPy

---

## Model Overview

- Artificial Neural Network (ANN) for regression
- Input: 4 environmental variables (AT, V, AP, RH)
- Output: Electrical Energy Output (PE)
- Preprocessing: Feature Scaling using StandardScaler
- Architecture:
  - Two hidden layers with ReLU activation
  - Output layer with one neuron (regression output)
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Evaluation Metric: Mean Absolute Error (MAE)

---

## Workflow

1. Load dataset from `Folds5x2_pp.xlsx` using Pandas
2. Split data into features (X) and target (y)
3. Divide dataset into training and testing sets (80:20 ratio)
4. Apply feature scaling to improve model performance
5. Build and compile ANN using Keras Sequential API
6. Train the model on training data
7. Evaluate model on test data
8. Predict energy output for unseen samples
