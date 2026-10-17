import tensorflow as tf

print("Initializing Fast Fashion Image Classification Pipeline...")

# Load the high-speed clothing dataset array instantly
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel arrays
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build a fast visual neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Starting Model Training Loops ---")
model.fit(train_images, train_labels, epochs=5)

print("\nFashion Image Classification Complete!")