# MNIST Handwritten Digit Classifier

A sequential deep learning model built with **TensorFlow** and **Keras** to classify handwritten digits (0-9) from the classic MNIST dataset. 

## 🎯 Performance Metrics
* **Final Validation Accuracy:** `97.77%`
* **Final Training Loss:** `0.0724`
* **Training Cycles (Epochs):** 5

## 🛠️ Architecture Setup
The network processes 28x28 pixel grayscale images through the following deep learning layers:
1. **Flatten Layer:** Reshapes the 2d image grid into a 784-element linear array.
2. **Dense Hidden Layer:** 128 neurons utilizing the `ReLU` activation function.
3. **Dropout Layer (0.2):** Randomly drops 20% of neural paths during iterations to prevent model overfitting.
4. **Dense Output Layer:** 10 nodes using `softmax` to determine the probability distribution for numbers 0-9.

## 🚀 How to Run Locally
Ensure you have the environment frameworks installed:
```bash
pip install tensorflow
python3 mnist_nm.py
```
