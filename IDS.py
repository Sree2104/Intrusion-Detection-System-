import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load the datasets and concatenate
d1 = pd.read_csv(r"C:\Users\sreet\Downloads\file\meta_pre.csv")
d2 = pd.read_csv(r"C:\Users\sreet\Downloads\file\Normal_pre.csv")
d3 = pd.read_csv(r"C:\Users\sreet\Downloads\file\torun_pre.csv")
d4 = pd.read_csv(r"C:\Users\sreet\Downloads\file\OVS_pre.csv")  
# Concatenate datasets
data = pd.concat([d1,d2,d3,d4], axis=0, ignore_index=True)

# Separate features and labels
X = data.drop(columns=["Label"])  # Assuming "Label" is the target column
y = data["Label"]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to reduce dimention
pca = PCA(n_components=25)
X_pca = pca.fit_transform(X_scaled)

# balance data using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape the input data for LSTM (timesteps=1)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Building Bi-LSTM model
model = Sequential()

# Input layer
model.add(Dense(128, activation='relu', input_shape=(1, X_train.shape[2])))

# Batch Normalization
model.add(BatchNormalization())

# Adding a Dropout layer for regularization
model.add(Dropout(0.3))

# Adding the first Bidirectional LSTM layer
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# Layer Normalization and Dropout
model.add(LayerNormalization())
model.add(Dropout(0.4))

# Adding second Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64, return_sequences=True)))

# Adding second Dropout and Batch Normalization layer
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Adding second Bidirectional LSTM layer
model.add(Bidirectional(LSTM(32)))

# Output layer (assuming multi-class classification with softmax)
model.add(Dense(6, activation='softmax'))  # Assuming 5 attack classes

# Compile the model with Adam optimizer and a learning rate scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print the results

print(f'Test Accuracy: {test_acc:.4f}')

print(f'Test Loss: {test_loss:.4f}')
print(f"Test Accuracy: {test_acc* 100:.2f}%")
model.save('Second.keras')