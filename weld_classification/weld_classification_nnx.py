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

print(jax.devices()); print(jax.device_count('gpu'))

# Define the label mapping with lowercase English names
label_mapping = {
    "no anomaly": 0,     # formerly NoDifetto
    "anomaly type 1": 1, # formerly Difetto1
    "anomaly type 2": 2, # formerly Difetto2
    "anomaly type 3": 3  # formerly Difetto4
}

# Reverse mapping for confusion matrix labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

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
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x, use_running_average=not training)
        x = nnx.relu(x)
        
        # Final pooling with average pool
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

def calculate_metrics_batched(model, images, labels, dataset_name="Dataset", batch_size=32):
    """Calculate comprehensive metrics for a dataset using batched processing to save memory."""
    # Convert to numpy arrays for consistent handling
    if not isinstance(images, np.ndarray):
        images = np.array(images)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Process in batches to avoid OOM errors
    n_samples = len(images)
    all_predictions = []
    
    print(f"Processing {n_samples} images from {dataset_name} in batches of {batch_size}...")
    
    # Get predictions batch by batch
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_images = jnp.array(images[i:end])
        
        # Forward pass
        logits = model(batch_images, training=False)
        batch_predictions = jnp.argmax(logits, axis=1)
        
        # Convert to numpy and append
        all_predictions.extend(np.array(batch_predictions))
        
        # Print progress
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n_samples} images")
    
    # Convert to numpy array
    predictions = np.array(all_predictions)
    
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

def save_metrics_polars_batched(model, train_data, val_data, test_data, filename="model_metrics.csv", batch_size=32):
    """
    Calculate metrics for all datasets in batches and save to CSV using Polars.
    Uses batched processing to avoid memory issues.
    """
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    # Calculate metrics for each dataset with batching
    try:
        print("\nCalculating training metrics...")
        train_metrics = calculate_metrics_batched(model, train_images, train_labels, "Training", batch_size)
        
        print("\nCalculating validation metrics...")
        val_metrics = calculate_metrics_batched(model, val_images, val_labels, "Validation", batch_size)
        
        print("\nCalculating test metrics...")
        test_metrics = calculate_metrics_batched(model, test_images, test_labels, "Test", batch_size)
        
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
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print("This is non-critical as model has already been saved.")
        return None

def save_confusion_matrix_csv(cm, class_names, label_indices, filename="confusion_matrix.csv"):
    """Save confusion matrix to CSV using Polars."""
    class_labels = [class_names.get(i, f"Class {i}") for i in label_indices]
    
    # Convert confusion matrix to a list of dictionaries
    cm_data = []
    for i, true_label in enumerate(class_labels):
        row_data = {'True': true_label}
        for j, pred_label in enumerate(class_labels):
            row_data[f'Pred_{pred_label}'] = int(cm[i, j])
        cm_data.append(row_data)
    
    # Create Polars DataFrame
    cm_df = pl.DataFrame(cm_data)
    
    # Save to CSV
    cm_df.write_csv(filename)
    print(f"Confusion matrix saved to {filename}")
    
    return cm_df

def save_model(model):
    """
    Save model using dill to a unique timestamped directory.
    
    Args:
        model: The model to save
    
    Returns:
        The path to the saved model
    """
    import dill
    from datetime import datetime
    import time
    import uuid
    
    # Generate a truly unique directory name using both timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID for readability
    ckpt_dir = os.path.join(os.getcwd(), f"model_save_{timestamp}_{unique_id}")
    
    # Create the directory
    os.makedirs(ckpt_dir)
    
    # Split the model into its computational graph and state
    _, state = nnx.split(model)
    
    # Save the state using dill
    with open(os.path.join(ckpt_dir, "model_state.dill"), "wb") as f:
        dill.dump(state, f)
    
    print(f"Model saved to {ckpt_dir} using dill")
    return ckpt_dir

def save_training_history(metrics, filename="training_history.csv"):
    """
    Save training metrics history to a CSV file using Polars.
    
    Args:
        metrics: Dictionary containing training metrics (epoch, losses, accuracy)
        filename: Path to save the CSV file
    """
    # Create a list of dictionaries for each epoch
    history_data = []
    for i in range(len(metrics['epoch'])):
        epoch_data = {
            'Epoch': metrics['epoch'][i],
            'TrainLoss': metrics['train_loss'][i],
            'ValLoss': metrics['val_loss'][i],
            'ValAccuracy': metrics['val_accuracy'][i]
        }
        history_data.append(epoch_data)
    
    # Create Polars DataFrame
    history_df = pl.DataFrame(history_data)
    
    # Format floating point columns with specified return_dtype
    for col in ['TrainLoss', 'ValLoss', 'ValAccuracy']:
        history_df = history_df.with_columns(
            pl.col(col).map_elements(lambda x: f"{x:.6f}", return_dtype=pl.Utf8).alias(col)
        )
    
    # Save to CSV
    history_df.write_csv(filename)
    print(f"Training history saved to {filename}")
    
    return history_df

def train_model(base_dir=None, samples_per_class=None, num_epochs=50, batch_size=32, n=1,
                train_data=None, val_data=None, test_data=None):
    """
    Train a model using loaded data or loading with specified parameters.
    Returns the trained model and training metrics.
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
        
        print(f"Loading{'up to ' + str(samples_per_class) + ' samples per class ' if samples_per_class else ' all samples '}(taking every {n}th image)...")
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
    
    # Final evaluation on test set
    test_loss, test_acc, test_preds = eval_step_fn(model, jnp.array(test_images), jnp.array(test_labels))
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Compute and display confusion matrix
    cm, label_indices = compute_confusion_matrix(test_labels, test_preds, label_mapping)
    print_confusion_matrix(cm, reverse_label_mapping, label_indices)
    
    # Save confusion matrix to CSV
    save_confusion_matrix_csv(cm, reverse_label_mapping, label_indices, "confusion_matrix.csv")
    
    # Save training history
    save_training_history(metrics, "training_history.csv")
    
    # IMPORTANT: Save the model immediately after basic evaluation
    save_model(model)
    print("Model parameters saved to model_checkpoints folder")
    
    # Try to calculate and save metrics in batches
    try:
        # Calculate and save comprehensive metrics using batched version
        metrics_df = save_metrics_polars_batched(  # Changed to batched version
            model,
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_labels),
            filename="weld_classification_metrics.csv",
            batch_size=batch_size  # Use same batch size as training
        )
        if metrics_df is not None:
            print("Comprehensive metrics calculated and saved successfully")
        else:
            print("Warning: Metrics calculation returned None")
            
    except Exception as e:
        print(f"Warning: Could not save full metrics due to error: {e}")
        print("This is non-critical as model and basic metrics are already saved.")

    return model, metrics

# Main execution
if __name__ == "__main__":
    # Set the path to your dataset
    base_dir = "DB - Copy/"

    print("\nDataset contents:", os.listdir(base_dir))

    # Load full datasets without sampling or limits
    print("\nLoading full datasets...")
    full_train_images, full_train_labels, class_names = load_dataset_with_limit(
        os.path.join(base_dir, 'training'), 
        samples_per_class=None,  # No limit
        target_size=(128, 128),
        n=1  # Take all images
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

    # Display class distributions
    print("\nClass distribution in datasets:")
    full_train_class_counts = np.bincount(full_train_labels)
    for i, count in enumerate(full_train_class_counts):
        print(f"{reverse_label_mapping[i]}: {count} training images ({count/len(full_train_labels)*100:.1f}%)")

    full_val_class_counts = np.bincount(full_val_labels)
    for i, count in enumerate(full_val_class_counts):
        print(f"{reverse_label_mapping[i]}: {count} validation images ({count/len(full_val_labels)*100:.1f}%)")

    full_test_class_counts = np.bincount(full_test_labels)
    for i, count in enumerate(full_test_class_counts):
        print(f"{reverse_label_mapping[i]}: {count} test images ({count/len(full_test_labels)*100:.1f}%)")

    # Train on full dataset
    trained_model, training_metrics = train_model(
        train_data=(full_train_images, full_train_labels, class_names),
        val_data=(full_val_images, full_val_labels, None),
        test_data=(full_test_images, full_test_labels, None),
        batch_size=32,
        num_epochs=1
    )
    
    print("\nTraining complete! Model and metrics have been saved.") 

    