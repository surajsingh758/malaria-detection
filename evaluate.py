import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 150
BATCH_SIZE = 32

# Load trained model
model = tf.keras.models.load_model("malaria_model.h5")

# Validation data (no augmentation)
datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Predictions
predictions = model.predict(val_data)
pred_classes = (predictions > 0.5).astype(int)
true_classes = val_data.classes

print("\nClassification Report:\n")
print(classification_report(true_classes, pred_classes))

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
print("\nConfusion Matrix:\n")
print(cm)

# ROC-AUC Score
roc_auc = roc_auc_score(true_classes, predictions)
print("\nROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(true_classes, predictions)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc_curve.png")

print("\nEvaluation Complete!")