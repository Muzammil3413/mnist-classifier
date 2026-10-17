import tensorflow as tf

print("Initializing Neural Network Training Pipeline...")

# 1. Load the official dataset of handwritten numbers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the image pixel values from 0-255 down to 0.0-1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build the deep learning architecture layer-by-layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compile the model configuration structural rules
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the network for 5 iterations
print("\n--- Starting Model Training Loops ---")
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate accuracy against independent verification images
print("\n--- Evaluating Model Against Unseen Test Data ---")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(f"\nFinal Validation Model Accuracy: {test_acc * 100:.2f}%")
