# Importing Libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Setting the parameters
batch_size = 100
epochs = 10
lr = 0.001

# Set location folder for images
parent_dir = 'C:/Users/user/OneDrive/Desktop/N'
data_dir = os.path.join(parent_dir, 'data_new')

# Preprocess images and split data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Split validation data into validation and test
test_size = len(validation_generator) // 2
test_generator = validation_generator.__getitem__(np.arange(test_size))
validation_generator = validation_generator.__getitem__(np.arange(test_size, len(validation_generator)))

# Customize ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
num_classes = train_generator.num_classes
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers except the last few
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    batch_size=batch_size
)

# Feature Extraction for SVM
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
train_features = feature_extractor.predict(train_generator)
test_features = feature_extractor.predict(test_generator)

train_labels = train_generator.classes
test_labels = test_generator.classes

# SVM Classification
svm = SVC(kernel='linear')
svm.fit(train_features, train_labels)
svm_predictions = svm.predict(test_features)

# Calculate Accuracy
accuracy = accuracy_score(test_labels, svm_predictions)
print(f'ResNet + SVM Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_mat = confusion_matrix(test_labels, svm_predictions)
print("Confusion Matrix:\n", conf_mat)

# ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(test_labels, svm.decision_function(test_features))
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(test_labels, svm.decision_function(test_features))
pr_auc = auc(recall, precision)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Plot SVM Decision Boundary
plt.figure()
plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels)
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
