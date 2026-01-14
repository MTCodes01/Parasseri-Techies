"""
Enhanced Training Script for Emotion Detection Model
Includes data augmentation, callbacks, and better training practices.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import deeplake
import os
from datetime import datetime

# Set mode
mode = 'train'

def plot_model_history(model_history, save_path='training_plots'):
    """
    Plot and save accuracy and loss curves.
    
    Args:
        model_history: Training history object
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), 
                model_history.history['accuracy'], label='Train', linewidth=2)
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), 
                model_history.history['val_accuracy'], label='Validation', linewidth=2)
    axs[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Accuracy', fontsize=12)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, 
                                step=max(1, len(model_history.history['accuracy']) // 10)))
    axs[0].legend(loc='best', fontsize=10)
    axs[0].grid(True, alpha=0.3)

    # Plot loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), 
                model_history.history['loss'], label='Train', linewidth=2)
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), 
                model_history.history['val_loss'], label='Validation', linewidth=2)
    axs[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Loss', fontsize=12)
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, 
                                step=max(1, len(model_history.history['loss']) // 10)))
    axs[1].legend(loc='best', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = os.path.join(save_path, f'training_plot_{timestamp}.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to: {plot_file}")
    plt.show()


def augment_image(image):
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image (48x48)
        
    Returns:
        Augmented image
    """
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random rotation (-15 to 15 degrees)
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness adjustment
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.7, 1.3)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    return image


def deeplake_generator(ds, batch_size, augment=False):
    """
    Generate batches of data from DeepLake dataset.
    
    Args:
        ds: DeepLake dataset
        batch_size: Batch size
        augment: Whether to apply data augmentation
        
    Yields:
        Batches of (images, labels)
    """
    while True:
        batch_images = []
        batch_labels = []
        for sample in ds:
            image = sample.images.data()["value"]
            label = sample.labels.data()["value"]

            # Preprocess image
            image = cv2.resize(image, (48, 48))
            
            # Apply augmentation if enabled
            if augment:
                image = augment_image(image)
            
            # Normalize and add channel dimension
            image = np.expand_dims(image, axis=-1) / 255.0

            # One-hot encode the label
            label = tf.keras.utils.to_categorical(label, num_classes=7)
            label = np.squeeze(label)

            batch_images.append(image)
            batch_labels.append(label)

            # Yield the batch if it's full
            if len(batch_images) == batch_size:
                yield np.array(batch_images), np.array(batch_labels)
                batch_images, batch_labels = [], []

        # Handle the remaining samples in the last batch
        if batch_images:
            yield np.array(batch_images), np.array(batch_labels)


def create_model():
    """
    Create the emotion detection CNN model with batch normalization.
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Second convolutional block
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    return model


def main():
    """Main training function."""
    print("="*60)
    print("Emotion Detection Model Training")
    print("="*60)
    
    # Load DeepLake datasets
    print("\nLoading datasets...")
    try:
        # Try DeepLake 4.0+ API
        train_ds = deeplake.open('hub://activeloop/fer2013-train')
        val_ds = deeplake.open('hub://activeloop/fer2013-public-test')
        print("Using DeepLake 4.0+ API")
    except (AttributeError, TypeError, RuntimeError):
        # Fall back to DeepLake 3.x API
        try:
            train_ds = deeplake.load('hub://activeloop/fer2013-train')
            val_ds = deeplake.load('hub://activeloop/fer2013-public-test')
            print("Using DeepLake 3.x API")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("\nPlease install DeepLake 3.x with: pip install 'deeplake<4'")
            raise
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Training parameters
    batch_size = 64  # Reduced for better generalization
    num_epochs = 100
    initial_learning_rate = 0.001

    # Create data generators
    train_generator = deeplake_generator(train_ds, batch_size, augment=True)
    validation_generator = deeplake_generator(val_ds, batch_size, augment=False)

    # Calculate steps
    steps_per_epoch = len(train_ds) // batch_size
    validation_steps = len(val_ds) // batch_size
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    print(f"  Initial learning rate: {initial_learning_rate}")

    # Create the model
    print("\nBuilding model...")
    model = create_model()
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()

    # Create callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'best_model_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', timestamp),
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
    ]

    print("\nStarting training...")
    print("="*60)
    
    # Train the model
    model_info = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    # Plot and save training history
    print("\nGenerating training plots...")
    plot_model_history(model_info)

    # Save final model weights
    final_weights_path = 'model.weights.h5'
    model.save_weights(final_weights_path)
    print(f"\nFinal model weights saved to: {final_weights_path}")
    
    # Save full model
    final_model_path = f'emotion_model_{timestamp}.h5'
    model.save(final_model_path)
    print(f"Full model saved to: {final_model_path}")
    
    # Print final metrics
    final_train_acc = model_info.history['accuracy'][-1]
    final_val_acc = model_info.history['val_accuracy'][-1]
    final_train_loss = model_info.history['loss'][-1]
    final_val_loss = model_info.history['val_loss'][-1]
    
    print("\nFinal Metrics:")
    print(f"  Training Accuracy: {final_train_acc:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Training Loss: {final_train_loss:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    
    print("\nTo view training progress in TensorBoard, run:")
    print(f"  tensorboard --logdir=logs/{timestamp}")


if __name__ == "__main__":
    main()
