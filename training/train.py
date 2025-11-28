import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ===================== CONFIG =====================
DATASET_PATHS = [
    "../data/groundnut",
    "../data/soyabean",
    "../data/sunflower"
]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 5
MODEL_SAVE_DIR = "../backend/crop_model"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")
CLASS_FILE = "../backend/class_indices.json"
SEED = 42
# ==================================================


def load_datasets():
    all_classes = set()
    # 1. Collect all class names globally to ensure consistent indexing
    for d in DATASET_PATHS:
        if os.path.exists(d):
            subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
            all_classes.update(subs)
    
    class_names = sorted(list(all_classes))
    print(f"Total classes found: {len(class_names)}")
    print("Classes:", class_names)
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    full_train_ds = None
    full_val_ds = None
    
    for d in DATASET_PATHS:
        if not os.path.exists(d):
            print(f"Warning: Directory not found: {d}")
            continue
            
        print(f"Loading from {d}...")
        # Load as individual images (batch_size=None) to allow mixing later
        ds_train_local = tf.keras.utils.image_dataset_from_directory(
            d,
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=None, 
            label_mode='int'
        )
        
        ds_val_local = tf.keras.utils.image_dataset_from_directory(
            d,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=None,
            label_mode='int'
        )
        
        local_classes = ds_train_local.class_names
        
        # Create mapping from local index to global index
        mapping = [class_to_idx[name] for name in local_classes]
        mapping_tensor = tf.constant(mapping, dtype=tf.int32)
        
        def remap_label(img, label):
            return img, tf.gather(mapping_tensor, label)
            
        ds_train_remapped = ds_train_local.map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val_remapped = ds_val_local.map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)
        
        if full_train_ds is None:
            full_train_ds = ds_train_remapped
            full_val_ds = ds_val_remapped
        else:
            full_train_ds = full_train_ds.concatenate(ds_train_remapped)
            full_val_ds = full_val_ds.concatenate(ds_val_remapped)

    AUTOTUNE = tf.data.AUTOTUNE
    
    # Shuffle, batch, and prefetch
    # Buffer size should be large enough to mix crops
    train_ds = full_train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = full_val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    # Data augmentation
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # EfficientNetB0 backbone
    base = EfficientNetB0(include_top=False,
                          weights="imagenet",
                          input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base.trainable = False  # Stage 1 training

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = data_aug(inputs)
    x = layers.Rescaling(1/255.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model, base


def main():
    # Load dataset
    train_ds, val_ds, class_names = load_datasets()

    # Build model
    model, base_model = build_model(len(class_names))
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    # ======================= STAGE 1 ===========================
    print("� Stage 1 Training (Freezing EfficientNet layers)…")
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS_STAGE1,
              callbacks=callbacks)

    # ======================= STAGE 2 ===========================
    print("� Stage 2 Fine-tuning (Unfreezing deeper layers)…")
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False  # freeze first 200 layers

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS_STAGE2,
              callbacks=callbacks)

    # ======================= SAVE MODEL =========================
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    model.save(os.path.join(MODEL_SAVE_DIR, "final_model.keras"))
    print(f"✅ Model saved at {MODEL_SAVE_DIR}")

    # Save class names
    class_map = {i: name for i, name in enumerate(class_names)}
    with open(CLASS_FILE, "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"✅ Class labels saved at {CLASS_FILE}")


if __name__ == "__main__":
    main()
