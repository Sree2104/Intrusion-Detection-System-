import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Load the MCAD dataset
d1 = pd.read_csv(r"C:\Users\sreet\Downloads\Python\work.csv")
d2 = pd.read_csv(r"C:\Users\sreet\Downloads\Python\work2.csv")
d3 = pd.read_csv(r"C:\Users\sreet\Downloads\Python\work3.csv")

# Concatenate datasets
data = pd.concat([d3], axis=0, ignore_index=True)

# Separate features and labels
X = data.drop(columns=["target"])
y = data["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions with PCA
pca = PCA(n_components=25)
X_pca = pca.fit_transform(X_scaled)

# Balance the dataset using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Load the pretrained model
pretrained_model = load_model('Second.keras')

# Create a new model to avoid duplicate layer names
model = Sequential()

# Clone layers from the pretrained model, updating names to avoid duplicates
for i, layer in enumerate(pretrained_model.layers[:-1]):  # Exclude the output layer
    cloned_layer = layer.__class__.from_config(layer.get_config())  # Clone the layer
    cloned_layer._name = f"{layer.name}_transfer_{i}"  # Update name to make it unique
    cloned_layer.trainable = False  # Freeze the cloned layer
    model.add(cloned_layer)

# Add a new output layer
model.add(Dense(6, activation='softmax', name="new_output_layer"))  # Adjust for new target class count

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('transferred_model_MCAD.keras', monitor='val_accuracy', save_best_only=True)

# Train on the MCAD dataset
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, 
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Transferred Model Test Accuracy: {test_acc:.4f}')
print(f'Transferred Model Test Loss: {test_loss:.4f}')
print(f"Transferred Model Test Accuracy: {test_acc * 100:.2f}%")
model.save('Final.keras') 