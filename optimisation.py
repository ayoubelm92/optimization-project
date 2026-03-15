import sys
sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import json
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize


plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    "figure.figsize": (7,5),
    "figure.dpi": 140,
    "font.size": 11
})

# 1. DOSSIERS HORODATÉS
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
export_dir = f"exports/optimization/{timestamp}"
dataset_dir = os.path.join(export_dir, "dataset_npy")
dataset_img_dir = os.path.join(export_dir, "dataset_images")

os.makedirs(export_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(dataset_img_dir, exist_ok=True)

print(f"📁 Résultats exportés dans : {export_dir}")

# 2. CHARGEMENT DATASET MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("✅ Données prêtes :", x_train.shape)

# 3. EXPORT DATASET .NPY
np.save(os.path.join(dataset_dir, "x_train_sample.npy"), x_train[:5000])
np.save(os.path.join(dataset_dir, "y_train_sample.npy"), y_train[:5000])
np.save(os.path.join(dataset_dir, "x_test_sample.npy"), x_test[:1000])
np.save(os.path.join(dataset_dir, "y_test_sample.npy"), y_test[:1000])
print("✅ Dataset .npy exporté")

# 4. EXPORT DATASET EN IMAGES PNG
def save_images(x_data, y_data, subset="train"):
    subset_dir = os.path.join(dataset_img_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)
    for i, img in enumerate(x_data):
        label = y_data[i]
        label_dir = os.path.join(subset_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_uint8 = (img.squeeze() * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(label_dir, f"{i}.png"))

save_images(x_train[:5000], y_train[:5000], subset="train")
save_images(x_test[:1000], y_test[:1000], subset="test")
print("✅ Dataset exporté en images PNG lisibles")

# 5. QUANTUM ATTENTION STABLE
class QuantumAttention(tf.keras.layers.Layer):
    def __init__(self, reduction=8):
        super().__init__()
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // self.reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((1,1,channels))

    def call(self, x):
        attn = self.avg_pool(x)
        attn = self.fc1(attn)
        attn = self.fc2(attn)
        attn = self.reshape(attn)
        return x * attn

# 6. CNN BASELINE & OPTIMIZED
def build_baseline():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

def build_optimized():
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(32,3,activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,activation='relu', name="last_conv")(x)
    x = QuantumAttention()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# 7. TRAINING ET PLOTS
def plot_history(history, name):
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='val_acc', linewidth=2)
    plt.title(f"{name} Accuracy", weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{export_dir}/{name}_accuracy_curve.png")
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='val_loss', linewidth=2)
    plt.title(f"{name} Loss", weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{export_dir}/{name}_loss_curve.png")
    plt.close()

def train_model(model, name):
    print(f"\n🚀 Entraînement : {name}")
    log_dir = os.path.join("logs", timestamp, name)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    plot_history(history, name)
    return history

# 8. METRICS COMPLETES
def compute_metrics(model, name):
    print(f"\n📊 Evaluation : {name}")
    y_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix", weight='bold')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/{name}_confusion_matrix.png")
    plt.close()
    y_test_bin = label_binarize(y_test, classes=np.arange(10))
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.title(f"{name} ROC (AUC={roc_auc:.3f})", weight='bold')
    plt.grid(alpha=0.3)
    plt.savefig(f"{export_dir}/{name}_roc.png")
    plt.close()
    precision, recall_pr, _ = precision_recall_curve(y_test_bin.ravel(), y_probs.ravel())
    plt.figure()
    plt.plot(recall_pr, precision, linewidth=2)
    plt.title(f"{name} Precision-Recall", weight='bold')
    plt.grid(alpha=0.3)
    plt.savefig(f"{export_dir}/{name}_pr_curve.png")
    plt.close()
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{export_dir}/{name}_classification_report.csv")
    return {"Model": name, "Accuracy": acc, "Recall": rec, "F1-score": f1, "AUC": roc_auc}

# 9. GRAD-CAM
def gradcam(model, img, layer_name):
    grad_model = tf.keras.models.Model([model.inputs],[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/tf.reduce_max(heatmap)
    return heatmap.numpy()

# 🧠 EXTRACTION ATTENTION MAP (Quantum Attention)
def get_attention_map(model, img):
    """
    Extrait la carte d'attention depuis la couche QuantumAttention
    """
    try:
        attention_layer = None
        for layer in model.layers:
            if isinstance(layer, QuantumAttention):
                attention_layer = layer
                break

        if attention_layer is None:
            print("[WARN] QuantumAttention non trouvée")
            return None

        attn_model = tf.keras.Model(
            inputs=model.input,
            outputs=attention_layer.output
        )

        attn_output = attn_model.predict(img, verbose=0)[0]

        # moyenne sur les canaux
        attn_map = np.mean(attn_output, axis=-1)

        # normalisation
        attn_map = np.maximum(attn_map, 0)
        attn_map /= (attn_map.max() + 1e-8)

        return attn_map

    except Exception as e:
        print("[ERREUR attention]", e)
        return None
# 10. RUN MODELS
baseline_model = build_baseline()
optimized_model = build_optimized()
train_model(baseline_model, "Baseline_CNN")
train_model(optimized_model, "Optimized_CNN_QAttention")
metrics_base = compute_metrics(baseline_model, "Baseline_CNN")
metrics_opt = compute_metrics(optimized_model, "Optimized_CNN_QAttention")

# Grad-CAM example
heatmap = gradcam(optimized_model, x_test[0:1], "last_conv")
plt.figure()
plt.imshow(heatmap, cmap="jet")
plt.colorbar()
plt.savefig(f"{export_dir}/gradcam.png")
plt.close()

# 🎨 SUPER VISUALISATION COMPARATIVE 
def generate_super_visualization():
    print("\n[VISU] Génération figure comparative PRO...")

    sample_img = x_test[0:1]
    original = sample_img[0].squeeze()

    # --- GradCAM baseline ---
    try:
        # nom automatique de la dernière conv baseline
        last_conv_baseline = None
        for layer in reversed(baseline_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_baseline = layer.name
                break

        heatmap_base = gradcam(baseline_model, sample_img, last_conv_baseline)
    except Exception as e:
        print("[WARN] GradCAM baseline indisponible:", e)
        heatmap_base = None

    # --- GradCAM optimized ---
    heatmap_opt = gradcam(optimized_model, sample_img, "last_conv")

    # --- Attention map ---
    attention_map = get_attention_map(optimized_model, sample_img)

    plt.figure(figsize=(14, 8))

    # Image originale
    plt.subplot(2,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image", weight='bold')
    plt.axis('off')

    # GradCAM baseline
    plt.subplot(2,2,2)
    if heatmap_base is not None:
        plt.imshow(original, cmap='gray')
        plt.imshow(heatmap_base, cmap='jet', alpha=0.5)
        plt.title("Baseline Grad-CAM", weight='bold')
    else:
        plt.text(0.5,0.5,"Baseline GradCAM\nnon disponible", ha='center')
    plt.axis('off')

    # GradCAM optimized
    plt.subplot(2,2,3)
    plt.imshow(original, cmap='gray')
    plt.imshow(heatmap_opt, cmap='jet', alpha=0.5)
    plt.title("Optimized Grad-CAM", weight='bold')
    plt.axis('off')

    # Attention map
    plt.subplot(2,2,4)
    if attention_map is not None:
        plt.imshow(attention_map, cmap='jet')
        plt.title("Quantum Attention Map", weight='bold')
    else:
        plt.text(0.5,0.5,"Attention map\nnon disponible", ha='center')
    plt.axis('off')

    plt.suptitle(
        "Visual Explanation: Baseline vs Optimized CNN",
        fontsize=16,
        weight='bold'
    )

    plt.tight_layout()

    save_path = f"{export_dir}/SUPER_visual_explanation.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[OK] Figure PRO sauvegardée : {save_path}")
# 11. TABLEAU GLOBAL & JSON
df_metrics = pd.DataFrame([metrics_base, metrics_opt])
df_metrics.to_excel(f"{export_dir}/GLOBAL_metrics_report.xlsx", index=False)
with open(f"{export_dir}/metrics_summary.json","w") as f:
    json.dump({"baseline":metrics_base,"optimized":metrics_opt}, f, indent=4)

# 12. SUPER BAR CHART
labels = ['Accuracy','Recall','F1-score','AUC']
baseline_vals = [metrics_base[l] for l in labels]
opt_vals = [metrics_opt[l] for l in labels]
x = np.arange(len(labels))
width=0.35
plt.figure(figsize=(8,6))
bars1 = plt.bar(x-width/2, baseline_vals, width, label='Baseline CNN')
bars2 = plt.bar(x+width/2, opt_vals, width, label='Optimized CNN + QAM')
plt.xticks(x,labels)
plt.ylim(0,1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.title("Performance Comparison",weight='bold')
plt.savefig(f"{export_dir}/SUPER_comparison_barplot.png")
plt.close()

# 13. HEATMAP METRICS
heatmap_df = df_metrics.set_index("Model")
plt.figure(figsize=(6,4))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Metrics Heatmap", weight='bold')
plt.tight_layout()
plt.savefig(f"{export_dir}/metrics_heatmap.png")
plt.close()

# 14. RADAR CHART
angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
angles=np.concatenate([angles,[angles[0]]])
baseline_radar = baseline_vals+[baseline_vals[0]]
opt_radar = opt_vals+[opt_vals[0]]
plt.figure(figsize=(6,6))
ax=plt.subplot(111,polar=True)
ax.plot(angles,baseline_radar,label='Baseline CNN', linewidth=2)
ax.fill(angles,baseline_radar,alpha=0.1)
ax.plot(angles,opt_radar,label='Optimized CNN + QAM', linewidth=2)
ax.fill(angles,opt_radar,alpha=0.1)
ax.set_thetagrids(angles[:-1]*180/np.pi,labels)
plt.legend(loc='upper right')
plt.title("Radar Comparison", weight='bold')
plt.savefig(f"{export_dir}/radar_comparison.png")
plt.close()

# 🚀 VISUALISATION FINALE
generate_super_visualization()

print(f"\n🎉 PROJET FINAL TERMINÉ")
print(f"[EXPORT] Tous les résultats sont dans : {export_dir}")