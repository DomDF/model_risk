import os
import jax, optax, pickle
import jax.numpy as jnp, jax.random as random
from flax import nnx

from einops import rearrange
import numpy as np

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import polars as pl

# Define the label mapping with lowercase English names
label_mapping = {
    "no anomaly": 0,     # formerly NoDifetto
    "anomaly type 1": 1, # formerly Difetto1
    "anomaly type 2": 2, # formerly Difetto2
    "anomaly type 3": 3  # formerly Difetto4
}

# Reverse mapping for confusion matrix labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# class SimpleWeldClassifier(nnx.Module):
#     """
#     CNN for classifying weld images with dropout regularization.
#     """
#     def __init__(self, num_classes: int, rngs: nnx.Rngs, dropout_rate=0.3):
#         # First convolutional layer
#         self.conv1 = nnx.Conv(
#             3,      # input channels (RGB)
#             32,     # output features
#             kernel_size=(3, 3),
#             rngs=rngs
#         )
        
#         # Second convolutional layer
#         self.conv2 = nnx.Conv(
#             32,     # input features
#             64,     # output features
#             kernel_size=(3, 3),
#             rngs=rngs
#         )
        
#         # Calculate the flattened size for the dense layer
#         # For 128x128 images after two 2x2 max pools: 128 -> 64 -> 32
#         # Final conv output will be 32x32x64
#         flat_size = 32 * 32 * 64
        
#         # Final dense layer for classification
#         self.dense = nnx.Linear(
#             in_features=flat_size,
#             out_features=num_classes,
#             rngs=rngs
#         )

#         # Store dropout rate but create dropout on-demand
#         self.dropout_rate = dropout_rate

#     def __call__(self, x, training=True, rngs=None):
#         # Normalize pixel values to [0, 1]
#         x = x / 255.0
        
#         # First conv block
#         x = self.conv1(x)
#         x = nnx.relu(x)
#         # Apply dropout after activation
#         if training and self.dropout_rate > 0:
#             dropout = nnx.Dropout(rate=self.dropout_rate)
#             x = dropout(x, deterministic=False, rngs=rngs)
#         x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        
#         # Second conv block
#         x = self.conv2(x)
#         x = nnx.relu(x)
#         # Apply dropout after activation
#         if training and self.dropout_rate > 0:
#             dropout = nnx.Dropout(rate=self.dropout_rate)
#             x = dropout(x, deterministic=False, rngs=rngs)
#         x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        
#         # Flatten the 3D tensor to 1D for the dense layer
#         x = rearrange(x, 'b c h w -> b (c h w)')
        
#         # Apply dropout before final classification layer
#         if training and self.dropout_rate > 0:
#             dropout = nnx.Dropout(rate=self.dropout_rate)
#             x = dropout(x, deterministic=False, rngs=rngs)
        
#         # Final classification layer
#         x = self.dense(x)

#         return x

class ImprovedWeldClassifier(nnx.Module):
    """
    Improved CNN for classifying weld images with:
    - Progressive feature extraction
    - Batch normalization
    - Pooling strategies
    (Spatial attention removed to reduce complexity)
    """
    def __init__(self, num_classes: int, rngs: nnx.Rngs, dropout_rate=0.2):
        # Progressive feature extraction with increasing channels
        # First convolutional block
        self.conv1 = nnx.Conv(
            3,      # input channels (RGB)
            32,     # output features
            kernel_size=(3, 3),
            padding=1,  # Preserve spatial dimensions
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        
        # Second convolutional block with increased channels
        self.conv2 = nnx.Conv(
            32,     # input features
            64,     # output features
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # Third convolutional block with further increased channels for deeper feature extraction
        # BENEFIT: Progressive feature extraction helps with hierarchical pattern recognition
        # MODEL RISK: Increased model capacity may lead to overfitting if training data is limited
        self.conv3 = nnx.Conv(
            64,     # input features
            128,    # output features
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs
        )
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        
        # Fourth convolutional block for high-level feature extraction
        self.conv4 = nnx.Conv(
            128,    # input features
            256,    # output features
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs
        )
        self.bn4 = nnx.BatchNorm(256, rngs=rngs)
        
        # Spatial attention mechanism removed to simplify model
        
        # Calculate the flattened size for the dense layer
        # Image size after pooling: 128 -> 64 -> 32 -> 16 -> 8
        # Final conv output will be 8x8x256
        flat_size = 8 * 8 * 256
        
        # Dense layers for classification with reduced dimensions
        self.dense1 = nnx.Linear(
            in_features=flat_size,
            out_features=512,
            rngs=rngs
        )
        
        self.dense2 = nnx.Linear(
            in_features=512,
            out_features=num_classes,
            rngs=rngs
        )

        # Store dropout rate but create dropout on-demand
        self.dropout_rate = dropout_rate

    def __call__(self, x, training=True, rngs=None):
        # Normalize pixel values to [0, 1]
        x = x / 255.0
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        # BENEFIT: Max pooling preserves strongest features and reduces spatial dimensions
        # MODEL RISK: Can lose spatial information about smaller defects
        x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), strides=(2, 2))
        
        # Fourth conv block (attention mechanism removed)
        x = self.conv4(x)
        x = self.bn4(x, use_running_average=not training)
        x = nnx.relu(x)
        
        # Final pooling with average pool
        # BENEFIT: Average pooling preserves more spatial information than max pooling
        # MODEL RISK: May dilute strong feature signals compared to max pooling
        x = nnx.avg_pool(x, (2, 2), strides=(2, 2))
        
        # Flatten the 3D tensor to 1D for the dense layer
        x = rearrange(x, 'b c h w -> b (c h w)')
        
        # Apply dropout before dense layers
        if training and self.dropout_rate > 0:
            dropout = nnx.Dropout(rate=self.dropout_rate)
            x = dropout(x, deterministic=False, rngs=rngs)
        
        # First dense layer
        x = self.dense1(x)
        x = nnx.relu(x)
        
        # Apply dropout between dense layers
        if training and self.dropout_rate > 0:
            dropout = nnx.Dropout(rate=self.dropout_rate)
            x = dropout(x, deterministic=False, rngs=rngs)
        
        # Final classification layer
        x = self.dense2(x)

        return x

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image using PIL and NumPy."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    
    # If grayscale, convert to 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=2)
    
    # Standardize using ImageNet means and stds
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    return img_array

def load_dataset_with_limit(directory, samples_per_class=None, target_size=(128, 128), n=1):
    """
    Load images with a limit on samples per class and option to take every nth image.
    
    Args:
        directory: Path to dataset directory
        samples_per_class: Maximum number of samples to load per class (None for all)
        target_size: Size to resize images to
        n: Load every nth image (n=1 loads all images, n=2 loads every other image, etc.)
    """
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all valid image files and sort them for consistent sampling
        img_files = sorted([f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Take every nth image
        sampled_files = img_files[::n]
        
        # Limit samples if specified (after n-sampling)
        if samples_per_class is not None and len(sampled_files) > samples_per_class:
            sampled_files = sampled_files[:samples_per_class]
        
        for img_name in sampled_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                img_array = load_and_preprocess_image(img_path, target_size)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {directory} (sampling 1/{n})")
    return np.array(images), np.array(labels), class_names

def batch_generator(X, y, batch_size=32, shuffle_data=True):
    """Generate batches of data."""
    n_samples = X.shape[0]
    if shuffle_data:
        X, y = shuffle(X, y)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        yield X[i:end], y[i:end]

def create_train_step():
    """Creates a JIT-compiled training step function."""
    @nnx.jit
    def train_step(model, optimizer, batch_images, batch_labels, rng):
        def loss_fn(model):
            # Split RNG for multiple dropout layers
            dropout_rng = nnx.Rngs(dropout=rng)
            # Forward pass with training=True and dropout RNG
            logits = model(batch_images, training=True, rngs=dropout_rng)
            # Calculate cross entropy loss
            num_classes = len(label_mapping)
            labels_onehot = jax.nn.one_hot(batch_labels, num_classes=num_classes)
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            return loss
        # Calculate loss and gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        # Update model parameters
        optimizer.update(grads)
        return loss, rng
    return train_step

def create_eval_step():
    """Creates a JIT-compiled evaluation step function."""
    @nnx.jit
    def eval_step(model, images, labels):
        # No need for RNGs during evaluation as dropout is disabled
        logits = model(images, training=False)
        labels_onehot = jax.nn.one_hot(labels, num_classes=len(label_mapping))
        loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == labels)
        return loss, accuracy, predictions
    return eval_step

def compute_confusion_matrix(true_labels, predictions, label_mapping):
    """
    Compute a confusion matrix using JAX/NumPy without sklearn.
    Handles non-consecutive label indices.
    
    Args:
        true_labels: Array of true class labels
        predictions: Array of predicted class labels
        label_mapping: Dictionary mapping class names to label indices
        
    Returns:
        confusion_matrix: A matrix of appropriate size
        label_indices: The indices used in the confusion matrix
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(true_labels, 'device_buffer'):
        true_labels = np.array(true_labels)
    if hasattr(predictions, 'device_buffer'):
        predictions = np.array(predictions)
    
    # Convert arrays to Python integers if needed
    true_labels = [int(x) for x in true_labels]
    predictions = [int(x) for x in predictions]
    
    # Get all unique label indices from the mapping
    label_indices = sorted(set(label_mapping.values()))
    
    # Create a mapping from actual label values to matrix indices
    label_to_idx = {int(label): i for i, label in enumerate(label_indices)}
    
    # Initialize confusion matrix with the correct size
    n_classes = len(label_indices)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    # Fill confusion matrix
    for t, p in zip(true_labels, predictions):
        t_idx = label_to_idx[t]
        p_idx = label_to_idx[p]
        cm[t_idx, p_idx] += 1
    
    return cm, label_indices

def print_confusion_matrix(cm, class_names, label_indices):
    """
    Print a confusion matrix and metrics without matplotlib.
    
    Args:
        cm: Confusion matrix array
        class_names: Dictionary mapping class indices to class names
        label_indices: The actual label indices used in the confusion matrix
    """
    # Create class labels for display
    class_labels = [class_names.get(i, f"Class {i}") for i in label_indices]
    
    # Print header
    print("\nConfusion Matrix:")
    print("----------------")
    
    # Print column headers
    header = "True\\Pred |"
    for label in class_labels:
        header += f" {label:10s} |"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, true_label in enumerate(class_labels):
        row = f"{true_label:10s} |"
        for j in range(len(class_labels)):
            row += f" {cm[i, j]:10d} |"
        print(row)
    print("-" * len(header))
    
    # Print classification report
    print("\nPer-class Metrics:")
    print("-----------------")
    for i, label_name in enumerate(class_labels):
        true_pos = cm[i, i]
        false_pos = np.sum(cm[:, i]) - true_pos
        false_neg = np.sum(cm[i, :]) - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
    
    # Print overall accuracy
    total = np.sum(cm)
    correct = np.sum(np.diag(cm))
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.4f}")

def summarize_model_improvements():
    """
    Provide a summary of the improvements in the ImprovedWeldClassifier and their impact on model risk.
    """
    improvements = [
        {
            "feature": "Progressive Feature Extraction",
            "implementation": "4 convolutional layers with increasing channel depth (32→64→128→256)",
            "benefits": "Hierarchical pattern recognition, better capture of complex weld defect features",
            "model_risk": "Increased model capacity may lead to overfitting with limited training data"
        },
        {
            "feature": "Batch Normalization",
            "implementation": "Applied after each convolutional layer",
            "benefits": "Stabilizes training, allows higher learning rates, reduces initialization sensitivity",
            "model_risk": "May reduce model robustness to distribution shifts, adds complexity to model deployment"
        },
        {
            "feature": "Mixed Pooling Strategy",
            "implementation": "Max pooling in early layers, average pooling in final layer",
            "benefits": "Preserves strong features early, retains more spatial information in final representation",
            "model_risk": "Can lose spatial information about smaller defects (max pool), or dilute strong signals (avg pool)"
        }
    ]
    
    print("\n===== MODEL IMPROVEMENT SUMMARY =====")
    for item in improvements:
        print(f"\n{item['feature']}:")
        print(f"  Implementation: {item['implementation']}")
        print(f"  Benefits: {item['benefits']}")
        print(f"  Model Risk: {item['model_risk']}")
    print("\n=====================================")

def train_model(base_dir=None, samples_per_class=50, num_epochs=50, batch_size=32, n=1,
                train_data=None, val_data=None, test_data=None):
    """
    Train a model using loaded data or loading with specified parameters.
    
    Args:
        base_dir: Base directory for the dataset (optional if data is provided)
        samples_per_class: Maximum samples per class
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        n: Take every nth image when loading
        train_data: Tuple of (images, labels, class_names) if already loaded
        val_data: Tuple of (images, labels, _) if already loaded
        test_data: Tuple of (images, labels, _) if already loaded
    """
    # Use provided data or load it
    if train_data is not None and val_data is not None and test_data is not None:
        train_images, train_labels, class_names = train_data
        val_images, val_labels, _ = val_data
        test_images, test_labels, _ = test_data
        print("Using provided datasets")
    elif base_dir is not None:
        # Load dataset with sample limit
        train_dir = os.path.join(base_dir, 'training')
        val_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'testing')
        
        print(f"Loading up to {samples_per_class} samples per class (taking every {n}th image)...")
        train_images, train_labels, class_names = load_dataset_with_limit(
            train_dir, samples_per_class=samples_per_class, n=n)
        val_images, val_labels, _ = load_dataset_with_limit(val_dir, n=n)
        test_images, test_labels, _ = load_dataset_with_limit(test_dir, n=n)
    else:
        raise ValueError("Either provide loaded data or a base_dir to load from")
    
    print(f"Training with {len(train_images)} images")
    print(f"Validating with {len(val_images)} images")
    print(f"Testing with {len(test_images)} images")
    print(f"Classes: {class_names}")
    
    # Initialize model with JAX
    key = random.PRNGKey(42)
    num_classes = len(label_mapping)
    model = ImprovedWeldClassifier(num_classes, nnx.Rngs(params=key), dropout_rate=0.2)
    
    # Calculate total training steps
    steps_per_epoch = len(train_images) // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    # Create learning rate schedule (cosine decay with warmup)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.001,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=0.0001
    )
    
    # Create optimizer with the schedule
    tx = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=1e-4  # Add L2 regularization
    )
    optimizer = nnx.Optimizer(model=model, tx=tx)
    
    # Create training and evaluation functions
    train_step_fn = create_train_step()
    eval_step_fn = create_eval_step()
    
    # Initialize an RNG for dropout
    dropout_rng = random.PRNGKey(seed=42)
    
    # Create lists to store metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    print("\nTraining progress:")
    print("-----------------")
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for batch_images, batch_labels in batch_generator(train_images, train_labels, batch_size):
            # Convert to JAX arrays
            batch_images = jnp.array(batch_images)
            batch_labels = jnp.array(batch_labels)
            
            # Get a new subkey for this batch
            dropout_rng, step_rng = random.split(dropout_rng)
            
            # Train step
            loss, step_rng = train_step_fn(model, optimizer, batch_images, batch_labels, step_rng)
            epoch_losses.append(loss)
            
            # Update the dropout RNG for the next batch
            dropout_rng = step_rng
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        val_loss, val_acc, _ = eval_step_fn(model, jnp.array(val_images), jnp.array(val_labels))
        
        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(float(train_loss))
        metrics['val_loss'].append(float(val_loss))
        metrics['val_accuracy'].append(float(val_acc))
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # Print final metrics in tabular form
    print("\nTraining Metrics:")
    print("-" * 65)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Accuracy':>12}")
    print("-" * 65)
    for i in range(len(metrics['epoch'])):
        print(f"{metrics['epoch'][i]:6d} | {metrics['train_loss'][i]:12.4f} | {metrics['val_loss'][i]:12.4f} | {metrics['val_accuracy'][i]:12.4f}")
    
    # Final evaluation on test set
    test_loss, test_acc, test_preds = eval_step_fn(model, jnp.array(test_images), jnp.array(test_labels))
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Compute and display confusion matrix
    cm, label_indices = compute_confusion_matrix(test_labels, test_preds, label_mapping)
    print_confusion_matrix(cm, reverse_label_mapping, label_indices)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Sample sanity check before entering the training loop
    print(f"Input image shape: {train_images.shape}")
    # Convert one sample for testing
    one_sample = jnp.array(train_images[:1])
    try:
        # First image should be for example (1, 128, 128, 3)
        # Set training=False to avoid needing RNGs for dropout during the test
        sample_output = model(one_sample, training=False)
        print(f"Model output shape for one sample: {sample_output.shape}")
    except Exception as e:
        print(f"Error during model test: {e}")
        print("This suggests a mismatch between your model architecture and input data shape")
        raise
    
    # Summarize model improvements
    summarize_model_improvements()
    
    # Save comprehensive metrics to CSV using Polars
    metrics_df = save_metrics_polars(
        model,
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
        filename="weld_classification_metrics.csv"
    )

    # Display the metrics table
    print("\nMetrics Summary:")
    print(metrics_df)

    return model, metrics

# Example usage
# Update this path to the correct location of your dataset
base_dir = "/Users/ddifrancesco/Github/model risk/weld_classification/Dataset_partitioned/DB - Copy/"

print("\nDataset contents:", os.listdir(base_dir))

# Load datasets with the correct size
n = 10  # take every nth image

# Load datasets using the existing directory structure
train_images, train_labels, class_names = load_dataset_with_limit(
    os.path.join(base_dir, 'training'), 
    samples_per_class=None, 
    target_size=(128, 128),  # Explicitly set size to match model
    n=n
)

# Also need to load validation and test data before passing to train_model
val_images, val_labels, _ = load_dataset_with_limit(
    os.path.join(base_dir, 'validation'), 
    target_size=(128, 128),  # Explicitly set size to match model
    n=n
)

test_images, test_labels, _ = load_dataset_with_limit(
    os.path.join(base_dir, 'testing'), 
    target_size=(128, 128),  # Explicitly set size to match model
    n=n
)

trained_model, training_metrics = train_model(
    train_data=(train_images, train_labels, class_names),
    val_data=(val_images, val_labels, None),
    test_data=(test_images, test_labels, None),
    batch_size=32,  # Smaller batch size may help with limited data
    num_epochs=100  # Train longer with early stopping
)


# Load full datasets without sampling or limits
full_train_images, full_train_labels, _ = load_dataset_with_limit(
    os.path.join(base_dir, 'training'), 
    samples_per_class=None,  # No limit
    target_size=(128, 128),
    n=50  # Take all images
)

full_val_images, full_val_labels, _ = load_dataset_with_limit(
    os.path.join(base_dir, 'validation'), 
    samples_per_class=None,
    target_size=(128, 128),
    n=1
)

full_test_images, full_test_labels, _ = load_dataset_with_limit(
    os.path.join(base_dir, 'testing'), 
    samples_per_class=None,
    target_size=(128, 128),
    n=1
)

# Then check the distributions
full_train_class_counts = np.bincount(full_train_labels)
for i, count in enumerate(full_train_class_counts):
    print(f"{reverse_label_mapping[i]}: {count} training images ({count/len(full_train_labels)*100:.1f}%)")

full_val_class_counts = np.bincount(full_val_labels)
for i, count in enumerate(full_val_class_counts):
    print(f"{reverse_label_mapping[i]}: {count} validation images ({count/len(full_val_labels)*100:.1f}%)")

full_test_class_counts = np.bincount(full_test_labels)
for i, count in enumerate(full_test_class_counts):
    print(f"{reverse_label_mapping[i]}: {count} test images ({count/len(full_test_labels)*100:.1f}%)")

trained_model, training_metrics = train_model(
    train_data=(full_train_images, full_train_labels, class_names),
    val_data=(full_val_images, full_val_labels, None),
    test_data=(full_test_images, full_test_labels, None),
    batch_size=32,  # Smaller batch size may help with limited data
    num_epochs=10  # Train longer with early stopping?
)

def calculate_metrics(model, images, labels, dataset_name="Dataset"):
    """Calculate comprehensive metrics for a dataset using Polars."""
    # Convert to JAX arrays if needed
    if not isinstance(images, jnp.ndarray):
        images = jnp.array(images)
    if not isinstance(labels, jnp.ndarray):
        labels = jnp.array(labels)
    
    # Get predictions
    logits = model(images, training=False)
    predictions = jnp.argmax(logits, axis=1)
    
    # Convert to numpy for easier handling
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Overall accuracy
    overall_accuracy = np.mean(predictions == labels)
    
    # Per-class metrics
    metrics_data = []
    
    # Calculate metrics for each class
    for cls_idx, cls_name in reverse_label_mapping.items():
        # True positives, false positives, false negatives
        true_pos = np.sum((predictions == cls_idx) & (labels == cls_idx))
        false_pos = np.sum((predictions == cls_idx) & (labels != cls_idx))
        false_neg = np.sum((predictions != cls_idx) & (labels == cls_idx))
        
        # Class metrics
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Class support (number of samples)
        support = np.sum(labels == cls_idx)
        
        # Add to metrics data
        metrics_data.append({
            'Dataset': dataset_name,
            'Class': cls_name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': int(support),
            'Accuracy': overall_accuracy  # Same for all classes within a dataset
        })
    
    # Add overall metrics (macro average)
    macro_precision = np.mean([m['Precision'] for m in metrics_data])
    macro_recall = np.mean([m['Recall'] for m in metrics_data])
    macro_f1 = np.mean([m['F1'] for m in metrics_data])
    total_support = np.sum([m['Support'] for m in metrics_data])
    
    metrics_data.append({
        'Dataset': dataset_name,
        'Class': 'Overall (macro avg)',
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1': macro_f1,
        'Support': int(total_support),
        'Accuracy': overall_accuracy
    })
    
    return metrics_data

def save_metrics_polars(model, train_data, val_data, test_data, filename="model_metrics.csv"):
    """
    Calculate metrics for all datasets and save to CSV using Polars.
    
    Args:
        model: Trained model
        train_data: Tuple of (images, labels)
        val_data: Tuple of (images, labels)
        test_data: Tuple of (images, labels)
        filename: Output CSV filename
    """
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    # Calculate metrics for each dataset
    train_metrics = calculate_metrics(model, train_images, train_labels, "Training")
    val_metrics = calculate_metrics(model, val_images, val_labels, "Validation")
    test_metrics = calculate_metrics(model, test_images, test_labels, "Test")
    
    # Combine all metrics
    all_metrics = train_metrics + val_metrics + test_metrics
    
    # Create Polars DataFrame
    metrics_df = pl.DataFrame(all_metrics)
    
    # Format floating point columns
    for col in ['Precision', 'Recall', 'F1', 'Accuracy']:
        metrics_df = metrics_df.with_columns(
            pl.col(col).map_elements(lambda x: f"{x:.4f}").alias(col)
        )
    
    # Save to CSV
    metrics_df.write_csv(filename)
    print(f"Metrics saved to {filename}")
    
    return metrics_df

"""

-----------------------------------------------------------------
 Epoch |   Train Loss |     Val Loss | Val Accuracy
-----------------------------------------------------------------
     1 |       1.1906 |       1.5886 |       0.2455
     2 |       1.0898 |       1.8105 |       0.2455
     3 |       1.0678 |       2.3924 |       0.2455
     4 |       0.9935 |       2.9915 |       0.2455
     5 |       0.9289 |       2.9536 |       0.2455
     6 |       0.9325 |       3.1001 |       0.2455
     7 |       0.8601 |       2.7732 |       0.2455
     8 |       0.8262 |       3.3698 |       0.2455
     9 |       0.7755 |       3.5766 |       0.2455
    10 |       0.6890 |       3.4601 |       0.2471
    11 |       0.6605 |       3.5591 |       0.2471
    12 |       0.5995 |       3.1489 |       0.3126
    13 |       0.5230 |       4.4606 |       0.3126
    14 |       0.4982 |       3.3592 |       0.3126
    15 |       0.3802 |       2.2891 |       0.2946
    16 |       0.3698 |       2.5085 |       0.2750
    17 |       0.3587 |       5.0025 |       0.2570
    18 |       0.2811 |       2.8534 |       0.3159
    19 |       0.3141 |      12.2290 |       0.3126
    20 |       0.1987 |       4.2747 |       0.3142
    21 |       0.1611 |       2.4542 |       0.3306
    22 |       0.1794 |       3.0853 |       0.2439
    23 |       0.1431 |       3.5357 |       0.1997
    24 |       0.1571 |       2.4904 |       0.2602
    25 |       0.1124 |       9.7362 |       0.2193
    26 |       0.1456 |      20.6819 |       0.1833
    27 |       0.1186 |       9.6439 |       0.2684
    28 |       0.0995 |      10.7408 |       0.1833
    29 |       0.0783 |       2.5213 |       0.3110
    30 |       0.0826 |       3.7614 |       0.3142
    31 |       0.0788 |      11.6001 |       0.2995
    32 |       0.0747 |       3.0095 |       0.3110
    33 |       0.0656 |       0.8967 |       0.6350
    34 |       0.0799 |       4.4573 |       0.2586
    35 |       0.0673 |       7.5838 |       0.2635
    36 |       0.0370 |       1.2496 |       0.5025
    37 |       0.0406 |       2.6823 |       0.3699
    38 |       0.0436 |      13.5701 |       0.2602
    39 |       0.0357 |       7.9930 |       0.2668
    40 |       0.0470 |       3.8594 |       0.4386
    41 |       0.0358 |       6.4262 |       0.3650
    42 |       0.0327 |       2.6316 |       0.3748
    43 |       0.0290 |      11.8909 |       0.2700
    44 |       0.0387 |       8.0534 |       0.2881
    45 |       0.0349 |       4.2531 |       0.2144
    46 |       0.0173 |       2.6802 |       0.3322
    47 |       0.0284 |       3.2264 |       0.3470
    48 |       0.0362 |       3.1173 |       0.3699
    49 |       0.0470 |       6.9349 |       0.3273
    50 |       0.0301 |       5.0401 |       0.2766
    51 |       0.0212 |       4.7269 |       0.3993
    52 |       0.0202 |       2.3598 |       0.4861
    53 |       0.0243 |       2.1298 |       0.4943
    54 |       0.0198 |       3.6179 |       0.3993
    55 |       0.0104 |       4.7796 |       0.3764
    56 |       0.0056 |       3.0391 |       0.4812
    57 |       0.0046 |       1.6318 |       0.6187
    58 |       0.0076 |       2.1635 |       0.6023
    59 |       0.0087 |       2.4738 |       0.6219
    60 |       0.0057 |       1.4831 |       0.6923
    61 |       0.0066 |       4.9311 |       0.4763
    62 |       0.0037 |       2.3433 |       0.6661
    63 |       0.0044 |       1.5562 |       0.7692
    64 |       0.0081 |       2.9634 |       0.6563
    65 |       0.0035 |       1.7737 |       0.7512
    66 |       0.0050 |       6.2404 |       0.4386
    67 |       0.0032 |       9.2946 |       0.3568
    68 |       0.0062 |       9.4672 |       0.3650
    69 |       0.0077 |       6.6005 |       0.4141
    70 |       0.0035 |       1.1239 |       0.6956
    71 |       0.0031 |       1.5988 |       0.6399
    72 |       0.0019 |       1.6513 |       0.6268
    73 |       0.0042 |       0.5123 |       0.8543
    74 |       0.0033 |       0.3987 |       0.9034
    75 |       0.0017 |       0.3713 |       0.9133
    76 |       0.0027 |       2.1158 |       0.5303
    77 |       0.0056 |       1.4728 |       0.6088
    78 |       0.0045 |       3.0423 |       0.4664
    79 |       0.0046 |       1.4365 |       0.6318
    80 |       0.0038 |       0.7918 |       0.7512
    81 |       0.0044 |       1.0431 |       0.7889
    82 |       0.0016 |       0.7263 |       0.8331
    83 |       0.0018 |       0.5024 |       0.8838
    84 |       0.0018 |       0.4843 |       0.8887
    85 |       0.0012 |       0.4649 |       0.8936
    86 |       0.0010 |       0.4676 |       0.9002
    87 |       0.0011 |       0.4331 |       0.8936
    88 |       0.0009 |       0.4097 |       0.9067
    89 |       0.0011 |       0.4167 |       0.8985
    90 |       0.0021 |       0.4068 |       0.9116
    91 |       0.0013 |       0.4066 |       0.9100
    92 |       0.0016 |       0.4362 |       0.9034
    93 |       0.0017 |       0.4080 |       0.9083
    94 |       0.0011 |       0.4148 |       0.9116
    95 |       0.0036 |       0.6242 |       0.8609
    96 |       0.0017 |       0.7211 |       0.8576
    97 |       0.0008 |       0.5486 |       0.8887
    98 |       0.0008 |       0.3991 |       0.9051
    99 |       0.0019 |       0.3919 |       0.9051
   100 |       0.0008 |       0.3911 |       0.9067

Final Test Accuracy: 0.9390

Confusion Matrix:
----------------
True\Pred | no anomaly | anomaly type 1 | anomaly type 2 | anomaly type 3 |
---------------------------------------------------------------------------
no anomaly |         72 |          1 |          4 |          0 |
anomaly type 1 |          2 |         62 |          0 |          0 |
anomaly type 2 |          7 |          0 |         37 |          1 |
anomaly type 3 |          0 |          0 |          0 |         60 |
---------------------------------------------------------------------------

Per-class Metrics:
-----------------
no anomaly:
  Precision: 0.8889
  Recall: 0.9351
  F1-score: 0.9114
anomaly type 1:
  Precision: 0.9841
  Recall: 0.9688
  F1-score: 0.9764
anomaly type 2:
  Precision: 0.9024
  Recall: 0.8222
  F1-score: 0.8605
anomaly type 3:
  Precision: 0.9836
  Recall: 1.0000
  F1-score: 0.9917

Overall Accuracy: 0.9390
Input image shape: (1588, 128, 128, 3)
Model output shape for one sample: (1, 4)


"""

trained_model = 