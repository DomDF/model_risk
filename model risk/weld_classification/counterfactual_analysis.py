import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flax import nnx
import optax
from matplotlib.animation import FuncAnimation
import os
import polars as pl
import jax.random as random
import pandas as pd
from datetime import datetime
from einops import rearrange  # Import einops for tensor reshaping

# Import the model class and label mappings
# Path to the NPZ model file
MODEL_PATH = "weld_classifier_model.npz"

# Define the label mapping with lowercase English names (same as in weld_classification_nnx.py)
label_mapping = {
    "no anomaly": 0,     # formerly NoDifetto
    "anomaly type 1": 1, # formerly Difetto1
    "anomaly type 2": 2, # formerly Difetto2
    "anomaly type 3": 3  # formerly Difetto4
}

# Reverse mapping for confusion matrix labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Directory name to class name mapping
dir_to_class = {
    "NoDifetto": "no anomaly",
    "Difetto1": "anomaly type 1",
    "Difetto2": "anomaly type 2",
    "Difetto4": "anomaly type 3"
}

# Reverse mapping: class name to directory name
class_to_dir = {v: k for k, v in dir_to_class.items()}

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



class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis on images using a trained model.
    Modifies input images to make them classified as target classes.
    """
    def __init__(self, model, label_mapping, learning_rate=500000, steps=1000):
        """
        Initialize the analyzer.
        
        Args:
            model: Trained classifier model
            label_mapping: Dictionary mapping class names to indices
            learning_rate: Step size for image modification
            steps: Number of steps for the analysis
        """
        self.model = model
        self.label_mapping = label_mapping
        self.reverse_mapping = {v: k for k, v in label_mapping.items()}
        self.learning_rate = learning_rate
        self.steps = steps
        # Define pixel valid range (0-255 for images)
        self.clip_range = (0, 255)
        self.num_classes = len(label_mapping)
        self.setup_functions()
    
    def setup_functions(self):
        """Set up the JAX functions for gradient computation and prediction."""
    
        # Function to get model predictions (probabilities)
        def predict_fn(image):
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image[None, ...]
            
            # The model expects float32 in range 0-255 (normalization happens inside)
            try:
                # Get logits from the model
                logits = self.model(image, training=False)
                
                # Convert to probabilities with softmax
                probs = jax.nn.softmax(logits, axis=-1)
                return probs[0]  # Remove batch dimension
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                # If there's an error, attempt to fix BatchNorm layers
                fixed = False
                for name, module in self.model.__dict__.items():
                    if isinstance(module, nnx.BatchNorm):
                        # Try to fix BatchNorm parameters
                        if isinstance(module.mean, np.ndarray) and module.mean.size == 1 and hasattr(module.mean.item(), 'value'):
                            module.mean = module.mean.item().value
                            fixed = True
                        if isinstance(module.var, np.ndarray) and module.var.size == 1 and hasattr(module.var.item(), 'value'):
                            module.var = module.var.item().value
                            fixed = True
                
                if fixed:
                    print("Fixed BatchNorm layers, retrying prediction...")
                    # Try again after fixing
                    logits = self.model(image, training=False)
                    probs = jax.nn.softmax(logits, axis=-1)
                    return probs[0]
                else:
                    raise ValueError("Failed to make prediction even after attempting to fix BatchNorm layers") from e
        
        self.predict = predict_fn
        
        # Function to compute loss for a target class
        def loss_fn(image, target_class):
            probs = predict_fn(image)
            # Negative log probability of the target class
            return -jnp.log(probs[target_class] + 1e-10)
        
        # Function to compute gradients of the loss w.r.t. the input image
        def grad_fn(image, target_class):
            return jax.grad(lambda x: loss_fn(x, target_class))(image)
        
        self.grad_fn = grad_fn
        
        # Function to update the image based on gradients
        def update_fn(image, target_class):
            grads = grad_fn(image, target_class)
            # Update in the direction that decreases the loss (increases target class probability)
            updated_image = image - self.learning_rate * grads
            # Clip to valid pixel range (0-255)
            updated_image = jnp.clip(updated_image, self.clip_range[0], self.clip_range[1])
            return updated_image
            
        self.update = update_fn
    
    def analyze_image(self, image, true_label):
        """
        Perform counterfactual analysis on an image for all classes except the true class.
        
        Args:
            image: Input image as numpy array (H, W, C)
            true_label: True class index of the image
            
        Returns:
            Dictionary of results for each target class
        """
        results = {}
        
        # Get all class indices (now consecutive 0,1,2,3)
        all_classes = list(range(self.num_classes))
        
        # Convert image to JAX array if needed
        if not isinstance(image, jnp.ndarray):
            image = jnp.array(image, dtype=jnp.float32)
        
        # Initial predictions
        initial_probs = self.predict(image)
        print("Initial predictions:")
        for class_idx in range(self.num_classes):
            class_name = self.reverse_mapping.get(class_idx, f"Class {class_idx}")
            print(f"  {class_name}: {initial_probs[class_idx]:.4f}")
        
        # For each class except the true class
        for target_class in all_classes:
            if target_class == true_label:
                continue
                
            target_name = self.reverse_mapping.get(target_class, f"Class {target_class}")
            print(f"Analyzing counterfactual for target class: {target_name}")
            
            # Initialize storage for tracking
            images = [image.copy()]
            probabilities = [self.predict(image)]
            
            # Current working image
            current_image = image.copy()
            
            # Perform the analysis steps
            for step in range(self.steps):
                # Update the image
                current_image = self.update(current_image, target_class)
                
                # Store the updated image and probabilities
                images.append(current_image.copy())
                probabilities.append(self.predict(current_image))
                
                # Print progress every 10 steps
                if (step + 1) % 10 == 0:
                    probs = probabilities[-1]
                    pred_idx = jnp.argmax(probs).item()
                    predicted_name = self.reverse_mapping.get(pred_idx, f"Class {pred_idx}")
                    
                    print(f"  Step {step+1}/{self.steps}: Predicted as {predicted_name} "
                          f"(p={probs[pred_idx]:.4f}), Target {target_name} "
                          f"(p={probs[target_class]:.4f})")
            
            # Store results for this target class
            results[target_class] = {
                'images': images,
                'probabilities': probabilities
            }
            
        return results
    
    def analyze_specific_counterfactual(self, image, source_class, target_class):
        """
        Perform counterfactual analysis for a specific source to target transformation.
        
        Args:
            image: Input image as numpy array (H, W, C)
            source_class: Source class name
            target_class: Target class name
            
        Returns:
            Dictionary with tabular results and modified images
        """
        # Get class indices
        source_idx = self.label_mapping[source_class]
        target_idx = self.label_mapping[target_class]
        
        # Convert image to JAX array if needed
        if not isinstance(image, jnp.ndarray):
            image = jnp.array(image, dtype=jnp.float32)
        
        # Initialize storage for tabular results
        tabular_results = {
            'step': [],
            'predicted_class': [],
            'predicted_prob': [],
            'target_prob': [],
            'loss': []
        }
        
        # Store images for visualization
        images = [image.copy()]
        
        # Current working image
        current_image = image.copy()
        
        # Initial prediction
        initial_probs = self.predict(current_image)
        initial_pred = jnp.argmax(initial_probs).item()
        initial_pred_name = self.reverse_mapping[initial_pred]
        
        print(f"Initial prediction: {initial_pred_name} (p={initial_probs[initial_pred]:.4f})")
        print(f"Target class: {target_class} (p={initial_probs[target_idx]:.4f})")
        
        # Store initial results
        tabular_results['step'].append(0)
        tabular_results['predicted_class'].append(initial_pred_name)
        tabular_results['predicted_prob'].append(float(initial_probs[initial_pred]))
        tabular_results['target_prob'].append(float(initial_probs[target_idx]))
        tabular_results['loss'].append(float(-jnp.log(initial_probs[target_idx] + 1e-10)))
        
        # Perform the analysis steps
        for step in range(self.steps):
            # Update the image
            current_image = self.update(current_image, target_idx)
            
            # Store the updated image
            images.append(current_image.copy())
            
            # Get current predictions
            probs = self.predict(current_image)
            pred_idx = jnp.argmax(probs).item()
            pred_name = self.reverse_mapping[pred_idx]
            
            # Calculate loss
            loss = float(-jnp.log(probs[target_idx] + 1e-10))
            
            # Store results
            tabular_results['step'].append(step + 1)
            tabular_results['predicted_class'].append(pred_name)
            tabular_results['predicted_prob'].append(float(probs[pred_idx]))
            tabular_results['target_prob'].append(float(probs[target_idx]))
            tabular_results['loss'].append(loss)
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{self.steps}: Predicted as {pred_name} "
                      f"(p={probs[pred_idx]:.4f}), Target {target_class} "
                      f"(p={probs[target_idx]:.4f}), Loss: {loss:.4f}")
        
        return {
            'images': images,
            'tabular_results': tabular_results
        }
    
    def visualize_results(self, image, true_label, results, save_path=None):
        """
        Visualize the counterfactual analysis results.
        
        Args:
            image: Original image
            true_label: True class index
            results: Results from analyze_image
            save_path: Optional path to save visualizations
        """
        # Create a directory for saving if needed
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Get class names
        true_class_name = self.reverse_mapping.get(true_label, f"Class {true_label}")
        
        # Plot original image with its prediction
        original_probs = self.predict(image)
        pred_idx = jnp.argmax(original_probs).item()
        predicted_name = self.reverse_mapping.get(pred_idx, f"Class {pred_idx}")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image.astype(np.uint8))
        plt.title(f"Original Image\nTrue: {true_class_name}, Predicted: {predicted_name} "
                 f"(p={original_probs[pred_idx]:.4f})")
        plt.axis('off')
        if save_path:
            plt.savefig(os.path.join(save_path, "original_image.png"))
        plt.show()
        
        # For each target class, visualize the progression
        for target_class, data in results.items():
            target_name = self.reverse_mapping.get(target_class, f"Class {target_class}")
            
            # Create a figure for the image progression
            plt.figure(figsize=(15, 8))
            
            # Select a subset of steps to display
            num_images = min(5, len(data['images']))
            step_indices = [0] + [i * (len(data['images'])-1) // (num_images-1) for i in range(1, num_images)]
            
            # Plot images at selected steps
            for i, idx in enumerate(step_indices):
                probs = data['probabilities'][idx]
                
                # Get the predicted class
                pred_idx = jnp.argmax(probs).item()
                current_pred_name = self.reverse_mapping.get(pred_idx, f"Class {pred_idx}")
                
                plt.subplot(1, num_images, i+1)
                plt.imshow(data['images'][idx].astype(np.uint8))
                if i == 0:
                    plt.title(f"Original\nPred: {predicted_name}\n(p={probs[pred_idx]:.2f})")
                else:
                    plt.title(f"Step {idx}\nPred: {current_pred_name}\n(p={probs[pred_idx]:.2f})")
                plt.axis('off')
            
            plt.suptitle(f"Counterfactual Progression: {true_class_name} → {target_name}")
            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, f"progression_{true_class_name}_to_{target_name}.png"))
            plt.show()
            
            # Plot probability changes over time
            plt.figure(figsize=(10, 6))
            probs_array = jnp.array(data['probabilities'])
            
            # Plot for each class
            for class_idx in range(self.num_classes):
                class_name = self.reverse_mapping.get(class_idx, f"Class {class_idx}")
                plt.plot(probs_array[:, class_idx], label=class_name)
            
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            plt.xlabel('Step')
            plt.ylabel('Probability')
            plt.title(f'Class Probabilities Over Time\nTarget: {target_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(os.path.join(save_path, f"probabilities_{true_class_name}_to_{target_name}.png"))
            plt.show()
            
            # Create an animation of the image changing
            if save_path:
                self._create_animation(data['images'], data['probabilities'], target_class, 
                                      os.path.join(save_path, f"animation_{true_class_name}_to_{target_name}.gif"))
    
    def visualize_specific_counterfactual(self, results, source_class, target_class, save_path=None):
        """
        Visualize the results of a specific counterfactual transformation.
        
        Args:
            results: Results dictionary from analyze_specific_counterfactual
            source_class: Source class name
            target_class: Target class name
            save_path: Directory to save visualizations
        """
        # Create save directory if needed
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        images = results['images']
        tabular_results = results['tabular_results']
        
        # Create a figure for the image progression
        plt.figure(figsize=(15, 8))
        
        # Select a subset of steps to display
        num_images = 5
        step_indices = [0] + [i * (len(images)-1) // (num_images-1) for i in range(1, num_images)]
        
        # Plot images at selected steps
        for i, idx in enumerate(step_indices):
            plt.subplot(1, num_images, i+1)
            plt.imshow(images[idx].astype(np.uint8))
            
            pred_class = tabular_results['predicted_class'][idx]
            pred_prob = tabular_results['predicted_prob'][idx]
            target_prob = tabular_results['target_prob'][idx]
            
            if i == 0:
                plt.title(f"Original\nPred: {pred_class}\n(p={pred_prob:.2f})")
            else:
                plt.title(f"Step {idx}\nPred: {pred_class}\n(p={pred_prob:.2f})")
            plt.axis('off')
        
        plt.suptitle(f"Counterfactual Progression: {source_class} → {target_class}")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "image_progression.png"))
        plt.show()
        
        # Plot probability and loss changes over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot probabilities
        steps = tabular_results['step']
        ax1.plot(steps, tabular_results['predicted_prob'], label='Predicted Class Prob')
        ax1.plot(steps, tabular_results['target_prob'], label=f'Target Class ({target_class}) Prob')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Probability')
        ax1.set_title('Class Probabilities Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(steps, tabular_results['loss'], color='red')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Counterfactual Loss: {source_class} → {target_class}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "metrics_progression.png"))
        plt.show()
        
        # Create an animation of the image changing
        if save_path:
            self._create_counterfactual_animation(images, tabular_results, source_class, target_class, 
                                               os.path.join(save_path, "animation.gif"))
        
        # Save tabular results as CSV using Polars
        results_df = pl.DataFrame(tabular_results)
        csv_path = os.path.join(save_path, "counterfactual_results.csv")
        results_df.write_csv(csv_path)
        print(f"Tabular results saved to {csv_path}")
    
    def _create_animation(self, images, probabilities, target_class, save_path, fps=10):
        """Create an animation of the image transformation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Prepare data for probability plot
        probs_array = jnp.array(probabilities)
        steps = range(len(images))
        lines = []
        
        # Set up the probability plot
        for class_idx in range(self.num_classes):
            class_name = self.reverse_mapping.get(class_idx, f"Class {class_idx}")
            line, = ax2.plot(steps[:1], probs_array[:1, class_idx], label=class_name)
            lines.append(line)
        
        ax2.set_xlim(0, len(steps)-1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Initialize the image plot
        im = ax1.imshow(images[0].astype(np.uint8))
        ax1.set_title('Image Transformation')
        ax1.axis('off')
        
        target_name = self.reverse_mapping.get(target_class, f"Class {target_class}")
        fig.suptitle(f'Counterfactual Analysis: Target = {target_name}')
        
        # Animation update function
        def update(frame):
            # Update image
            im.set_array(images[frame].astype(np.uint8))
            
            # Update probability lines
            for i, line in enumerate(lines):
                line.set_data(steps[:frame+1], probs_array[:frame+1, i])
            
            # Update title with current prediction
            probs = probabilities[frame]
            pred_idx = jnp.argmax(probs).item()
            current_pred_name = self.reverse_mapping.get(pred_idx, f"Class {pred_idx}")
            
            # Display both the current prediction and target probability
            ax1.set_title(f'Step {frame}: Predicted as {current_pred_name} (p={probs[pred_idx]:.2f})\n'
                          f'Target {target_name} (p={probs[target_class]:.2f})')
            
            return [im] + lines
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(images), blit=True)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close(fig)
    
    def _create_counterfactual_animation(self, images, results, source_class, target_class, save_path, fps=10):
        """Create an animation of the image transformation with loss visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Set up the image plot
        im = ax1.imshow(images[0].astype(np.uint8))
        ax1.set_title('Image Transformation')
        ax1.axis('off')
        
        # Set up the loss plot
        steps = results['step']
        line, = ax2.plot(steps[:1], results['loss'][:1], color='red')
        
        ax2.set_xlim(0, max(steps))
        ax2.set_ylim(0, max(results['loss']) * 1.1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Counterfactual Loss')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Counterfactual Analysis: {source_class} → {target_class}')
        
        # Animation update function
        def update(frame):
            # Update image
            im.set_array(images[frame].astype(np.uint8))
            
            # Update loss line
            line.set_data(steps[:frame+1], results['loss'][:frame+1])
            
            # Update title with current prediction
            pred_class = results['predicted_class'][frame]
            pred_prob = results['predicted_prob'][frame]
            target_prob = results['target_prob'][frame]
            
            ax1.set_title(f'Step {frame}: Predicted as {pred_class} (p={pred_prob:.2f})\n'
                         f'Target: {target_class} (p={target_prob:.2f})')
            
            return [im, line]
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(images), blit=True)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close(fig)


def load_image(image_path, size=(128, 128)):
    """Load and preprocess an image for analysis."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(size)
        return np.array(img)

def select_example_images(base_dir, n=1):
    """
    Select and display one example image from each class.
    
    Args:
        base_dir: Base directory containing the dataset
        n: Which image to select (e.g., n=1 selects the first image)
    
    Returns:
        Dictionary mapping class names to image paths
    """
    # Use the training directory
    train_dir = os.path.join(base_dir, 'training')
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(train_dir) 
                 if os.path.isdir(os.path.join(train_dir, d))]
    
    example_images = {}
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(class_dirs), figsize=(16, 4))
    
    # For each class, select an example image
    for i, class_name in enumerate(class_dirs):
        class_dir = os.path.join(train_dir, class_name)
        
        # Get all image files
        img_files = sorted([f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Select the nth image
        if n < len(img_files):
            img_path = os.path.join(class_dir, img_files[n])
            
            # Map directory name to class name
            if class_name in dir_to_class:
                mapped_class = dir_to_class[class_name]
                example_images[mapped_class] = img_path
            else:
                print(f"Warning: Unknown directory name {class_name}")
            example_images[class_name] = img_path

            # Display the image
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"{dir_to_class.get(class_name, class_name)}")
            axes[i].axis('off')
        else:
            print(f"Warning: Class {class_name} has fewer than {n+1} images")
    
    plt.tight_layout()
    plt.show()
    
    return example_images

def perform_counterfactual(model, image_path, source_class, target_class, 
                          learning_rate=500000, steps=1000, save_dir = "counterfactual_results"):
    """
    Perform a specific counterfactual transformation.
    
    Args:
        model: Trained model
        image_path: Path to the source image
        source_class: Source class name (e.g., "no anomaly")
        target_class: Target class name (e.g., "anomaly type 1")
        learning_rate: Step size for gradient updates
        steps: Number of steps to run
        save_dir: Directory to save results
    """
    # Load the image
    image = load_image(image_path)
    
    # Create analyzer
    analyzer = CounterfactualAnalyzer(model, label_mapping, learning_rate=learning_rate, steps=steps)
    
    # Run analysis for just this specific transformation
    print(f"Starting counterfactual analysis: {source_class} → {target_class}")
    
    # Perform the specific counterfactual analysis
    results = analyzer.analyze_specific_counterfactual(image, source_class, target_class)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{source_class}_to_{target_class}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None
    
    # Visualize the results
    analyzer.visualize_specific_counterfactual(results, source_class, target_class, save_path)
    
    return results

def analyze_and_visualize(model, image_path, true_class_name, save_dir=None, learning_rate=500000, steps=1000):
    """
    Perform counterfactual analysis on a single image and visualize results.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        true_class_name: Name of the true class (e.g., "no anomaly")
        save_dir: Directory to save visualizations
        learning_rate: Step size for image modifications
        steps: Number of steps for the analysis
    """
    # Load the image
    image = load_image(image_path)
    
    # Get the true class index
    true_label = label_mapping.get(true_class_name)
    if true_label is None:
        raise ValueError(f"Unknown class name: {true_class_name}. Valid names are: {list(label_mapping.keys())}")
    
    # Create analyzer
    analyzer = CounterfactualAnalyzer(model, label_mapping, learning_rate=learning_rate, steps=steps)
    
    # Run analysis
    print(f"Starting counterfactual analysis for image: {image_path}")
    print(f"True class: {true_class_name} (index {true_label})")
    
    results = analyzer.analyze_image(image, true_label)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"counterfactual_{image_name}_{true_class_name}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None
    
    # Visualize results
    analyzer.visualize_results(image, true_label, results, save_path)
    
    return results

def explain_counterfactual_purpose():
    """Print explanation of the purpose of counterfactual analysis for model risk."""
    print("\n=== Purpose of Counterfactual Analysis for Model Risk ===")
    print("""
    Counterfactual analysis helps us understand model risk in several ways:
    
    1. Identifying Decision Boundaries: By observing how images need to change to cross 
       decision boundaries, we gain insights into what features the model considers important
       for classification.
    
    2. Detecting Adversarial Vulnerabilities: The minimal changes needed to flip a classification
       can reveal potential adversarial attack vectors that could be exploited in real-world scenarios.
    
    3. Improving Model Robustness: Understanding these vulnerabilities allows us to enhance training
       with adversarial examples, making the model more robust to small perturbations.
    
    4. Interpretability: The visual changes highlight what the model "thinks" makes one class
       different from another, improving our understanding of its decision-making process.
    
    5. Fairness Assessment: Counterfactual analysis can reveal if the model relies on irrelevant
       or potentially biased features for classification.
    
    In the context of weld classification, this analysis helps ensure the model is focusing on
    actual defect characteristics rather than irrelevant image features, which is crucial for
    safety-critical applications where false negatives could lead to structural failures.
    """)

def run_planned_counterfactuals(model, base_dir, image_index=0, save_dir="counterfactual_results"):
    """
    Run the planned series of counterfactual analyses.
    
    Args:
        model: Trained model
        base_dir: Base directory containing the dataset
        image_index: Index of the image to use from each class
        save_dir: Directory to save results
    """
    # Step 1: Select example images from each class
    example_images = select_example_images(base_dir, n=image_index)
    
    # Verify we have all the required classes
    required_classes = ["no anomaly", "anomaly type 1", "anomaly type 2", "anomaly type 3"]
    for cls in required_classes:
        if cls not in example_images:
            print(f"Error: No example image found for class '{cls}'")
            return
    
    # Step 2: Define the counterfactual transformations to perform
    transformations = [
        # Starting with just the first one as requested
        {"source": "no anomaly", "target": "anomaly type 1"}
        
        # The rest can be uncommented when ready to proceed
        # {"source": "no anomaly", "target": "anomaly type 2"},
        # {"source": "no anomaly", "target": "anomaly type 3"},
        # {"source": "anomaly type 1", "target": "no anomaly"},
        # {"source": "anomaly type 2", "target": "no anomaly"},
        # {"source": "anomaly type 3", "target": "no anomaly"}
    ]
    
    # Step 3: Perform each counterfactual analysis
    for transform in transformations:
        source = transform["source"]
        target = transform["target"]
        
        print(f"\n{'='*80}")
        print(f"Analyzing counterfactual: {source} → {target}")
        print(f"{'='*80}\n")
        
        # Get the image path
        image_path = example_images[source]
        
        # Perform the counterfactual analysis
        perform_counterfactual(
            model,
            image_path,
            source,
            target,
            learning_rate=500000,
            steps=1000,
            save_dir=save_dir
        )
    
    # Step 5: Explain the purpose of counterfactual analysis
    explain_counterfactual_purpose()

def load_dill(model_path, num_classes=4):
    """
    Load a saved model from a dill file
    
    Args:
        model_path: Path to the directory containing the model_state.dill file
        num_classes: Number of classes the model was trained for (default 4)
    
    Returns:
        The loaded model with restored parameters
    """
    import dill
    import jax.random as random
    
    # Path to the dill file
    dill_path = os.path.join(model_path, "model_state.dill")
    
    # Load the saved state
    with open(dill_path, "rb") as f:
        saved_state = dill.load(f)
    
    # Create a new model instance with the same architecture
    key = random.PRNGKey(0)  # Seed doesn't matter for loading
    model = ImprovedWeldClassifier(num_classes=num_classes, rngs=nnx.Rngs(params=key), dropout_rate=0.2)
    
    # Split the model
    model_vars, _ = nnx.split(model)
    
    # Merge the loaded state with the new model
    model = nnx.merge(model_vars, saved_state)
    
    print(f"Model successfully loaded from {dill_path}")
    return model

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image to match the training script exactly."""
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure RGB
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    # Standardize using ImageNet means and stds (as in training script)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    return img_array  # Return in standardized range as in training

def create_model_wrapper(model):
    """
    Create a wrapper function for the model that handles normalization differences.
    The training script normalizes images to standardized values, but the model expects
    inputs in [0, 255] range and does its own division by 255.0 internally.
    """
    def wrapped_model(inputs, training=False):
        # Training script normalized to standardized range
        # But the model expects 0-255 range and does its own division
        # So we need to un-normalize, then multiply by 255
        
        # First, get the standardized mean and std used in preprocessing
        # Use JAX arrays for compatibility with traced arrays
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        
        # Check if already in 0-255 range - use jnp operations for JAX compatibility
        if jnp.max(jnp.abs(inputs)) > 5.0:  # Heuristic for already being in 0-255 range
            # Already in expected range, pass directly
            return model(inputs, training=training)
        else:
            # Convert from standardized to 0-255 range
            # First un-standardize
            inputs_unnormalized = inputs * std + mean
            # Then scale to 0-255
            inputs_scaled = inputs_unnormalized * 255.0
            # Pass to model which will divide by 255.0 internally
            return model(inputs_scaled, training=training)
    
    return wrapped_model

def predict_image(model, image):
    """Get model predictions for an image."""
    import jax.numpy as jnp
    import jax.nn as nn
    
    # Create a wrapped model that handles normalization differences
    wrapped_model = create_model_wrapper(model)
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image[None, ...]
    
    # Get model predictions (logits)
    logits = wrapped_model(image, training=False)
    
    # Convert to probabilities
    probs = nn.softmax(logits, axis=-1)
    
    # Get the predicted class
    pred_class = jnp.argmax(probs, axis=-1)[0]
    
    return pred_class, probs[0]

def find_correctly_classified_image(model, class_name="no anomaly", base_dir="DB - Copy"):
    """Find an image that is correctly classified as the specified class."""
    # Get directory name for the class
    dir_name = class_to_dir[class_name]  # Use the reverse mapping
    expected_label = label_mapping[class_name]
    
    # Path to class directory
    class_dir = os.path.join(base_dir, "training", dir_name)
    
    # List all images in the directory
    image_files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Searching through {len(image_files)} images...")
    
    # Check each image
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        image = load_and_preprocess_image(img_path)
        pred_class, probs = predict_image(model, image)
        
        # Convert JAX array to a standard Python integer
        pred_class_int = pred_class.item() if hasattr(pred_class, 'item') else int(pred_class)
        
        if pred_class_int == expected_label:
            print(f"Found correctly classified image: {img_file}")
            print(f"Prediction: {class_name} with probability {probs[pred_class_int]:.4f}")
            return img_path, image
    
    print("No correctly classified images found.")
    return None, None

def perform_counterfactual_analysis(model, image, source_class, target_class, 
                                  steps=1000, learning_rate=500000.0, save_dir="counterfactual_results"):
    """
    Gradually modify an image to change its classification from source to target class.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    source_label = label_mapping[source_class]
    target_label = label_mapping[target_class]
    
    # Convert image to JAX array if needed
    if not isinstance(image, jnp.ndarray):
        image = jnp.array(image, dtype=jnp.float32)
    
    # Ensure image has proper preprocessing like in training
    # Print the image statistics to debug normalization
    print(f"Image stats - min: {jnp.min(image):.4f}, max: {jnp.max(image):.4f}, mean: {jnp.mean(image):.4f}")
    
    # Create a wrapped model that handles normalization correctly
    wrapped_model = create_model_wrapper(model)
    
    # Visualization of original image
    # Convert from standardized range to 0-255 for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Function to convert standardized image to display format
    def to_display_format(img):
        # Un-standardize
        display_img = img * std + mean
        # Scale to 0-255
        display_img = display_img * 255.0
        # Clip to valid range and convert to uint8
        return np.clip(display_img, 0, 255).astype(np.uint8)
    
    # Save the visualization of the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(to_display_format(image))
    plt.title(f"Original {source_class} Image")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "original_image.png"))
    plt.close()
    
    # Initialize storage for tracking progress
    saved_images = [image.copy()]
    saved_probs = []
    saved_losses = []
    
    # Get initial prediction
    _, initial_probs = predict_image(model, image)
    saved_probs.append(initial_probs)
    
    # Calculate initial loss (negative log probability of target class)
    initial_loss = -jnp.log(initial_probs[target_label] + 1e-10)
    saved_losses.append(float(initial_loss))
    
    print(f"Starting counterfactual analysis: {source_class} → {target_class}")
    print(f"Initial probabilities: {source_class}: {initial_probs[source_label]:.4f}, "
          f"{target_class}: {initial_probs[target_label]:.4f}")
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Define loss function using our wrapped model
    def loss_fn(img):
        # Forward pass (add batch dimension)
        if len(img.shape) == 3:
            img_batch = img[None, ...]
        else:
            img_batch = img
            
        # Use the wrapped model
        logits = wrapped_model(img_batch, training=False)
        probs = jax.nn.softmax(logits, axis=-1)[0]
        return -jnp.log(probs[target_label] + 1e-10)
    
    # Sanity check: Can the loss change with a perturbed image?
    print("\n=== DEBUGGING: Sanity Check for Model Sensitivity ===")
    perturbed_image = image.copy()
    
    # Add noise to the image - directly in standardized space
    # Create noise in the standardized space
    noise_mask = jnp.zeros_like(perturbed_image)
    # Apply stronger perturbation (since we're in standardized space with smaller values)
    noise_mask = noise_mask.at[50:70, 50:70, :].set(0.5)  # Smaller value for standardized space
    perturbed_image = perturbed_image + noise_mask
    
    # Get original loss and prediction
    orig_logits = wrapped_model(image[None, ...], training=False)
    orig_probs = jax.nn.softmax(orig_logits, axis=-1)[0]
    orig_pred = jnp.argmax(orig_probs)
    orig_pred_int = orig_pred.item() if hasattr(orig_pred, 'item') else int(orig_pred)
    orig_loss = -jnp.log(orig_probs[target_label] + 1e-10)
    
    # Get perturbed loss and prediction
    perturbed_logits = wrapped_model(perturbed_image[None, ...], training=False)
    perturbed_probs = jax.nn.softmax(perturbed_logits, axis=-1)[0]
    perturbed_pred = jnp.argmax(perturbed_probs)
    perturbed_pred_int = perturbed_pred.item() if hasattr(perturbed_pred, 'item') else int(perturbed_pred)
    perturbed_loss = -jnp.log(perturbed_probs[target_label] + 1e-10)
    
    print(f"Original prediction: {reverse_label_mapping[orig_pred_int]} ({orig_probs[orig_pred_int]:.6f})")
    print(f"Perturbed prediction: {reverse_label_mapping[perturbed_pred_int]} ({perturbed_probs[perturbed_pred_int]:.6f})")
    print(f"Original loss: {orig_loss:.6f}")
    print(f"Perturbed loss: {perturbed_loss:.6f}")
    print(f"Loss difference: {perturbed_loss - orig_loss:.6f}")
    print("\nDetailed probability changes:")
    for class_idx in range(len(label_mapping)):
        delta = perturbed_probs[class_idx] - orig_probs[class_idx]
        print(f"  {reverse_label_mapping[class_idx]}: {orig_probs[class_idx]:.6f} → {perturbed_probs[class_idx]:.6f} (Δ: {delta:.6f})")
    
    if jnp.abs(perturbed_loss - orig_loss) < 1e-5:
        print("\nWARNING: Model doesn't show sensitivity to image changes!")
        print("This suggests the model may not be suitable for counterfactual analysis.")
    print("=== END DEBUGGING ===\n")
    
    # Create gradient function
    grad_fn = jax.grad(loss_fn)
    
    # Initial gradient check
    initial_grads = grad_fn(image)
    grad_magnitude = jnp.mean(jnp.abs(initial_grads))
    grad_min = jnp.min(initial_grads)
    grad_max = jnp.max(initial_grads)
    grad_abs_max = jnp.max(jnp.abs(initial_grads))
    non_zero = jnp.sum(jnp.abs(initial_grads) > 1e-10)
    grad_std = jnp.std(initial_grads)
    
    print("\n=== Initial Gradient Statistics ===")
    print(f"Mean absolute gradient: {grad_magnitude:.8f}")
    print(f"Min gradient: {grad_min:.8f}")
    print(f"Max gradient: {grad_max:.8f}")
    print(f"Max absolute gradient: {grad_abs_max:.8f}")
    print(f"Gradient standard deviation: {grad_std:.8f}")
    print(f"Non-zero gradients: {non_zero}/{initial_grads.size} ({non_zero/initial_grads.size*100:.4f}%)")
    
    # Debug: Check if the model has BatchNorm in train mode
    model_has_bn = False
    for name, module in model.__dict__.items():
        if isinstance(module, nnx.BatchNorm):
            model_has_bn = True
            break
    
    if model_has_bn:
        print("\nModel has BatchNorm layers. Checking if training mode affects gradients...")
        # Try generating gradients with training=True to see if that changes anything
        def loss_fn_train(img):
            logits = model(img[None, ...], training=True)
            probs = jax.nn.softmax(logits, axis=-1)[0]
            return -jnp.log(probs[target_label] + 1e-10)
        
        try:
            grad_fn_train = jax.grad(loss_fn_train)
            train_grads = grad_fn_train(image)
            train_non_zero = jnp.sum(jnp.abs(train_grads) > 1e-10)
            print(f"Training mode non-zero gradients: {train_non_zero}/{train_grads.size}")
            print(f"Max absolute gradient in training mode: {jnp.max(jnp.abs(train_grads)):.8f}")
        except Exception as e:
            print(f"Error computing gradients in training mode: {str(e)}")
    
    # Fallback: try jax.jit on the gradient function
    print("\nAttempting to JIT compile the gradient function...")
    try:
        jitted_grad_fn = jax.jit(grad_fn)
        jitted_grads = jitted_grad_fn(image)
        jitted_non_zero = jnp.sum(jnp.abs(jitted_grads) > 1e-10)
        print(f"JIT compiled gradient non-zero count: {jitted_non_zero}/{jitted_grads.size}")
    except Exception as e:
        print(f"Error JIT compiling gradient function: {str(e)}")
    
    # If we have no usable gradients at all, try a random perturbation approach
    use_random_search = non_zero == 0
    if use_random_search:
        print("\nWARNING: No usable gradients detected. Switching to random search approach.")
        print("This will be much less efficient but might still find useful counterfactuals.")
        
        # Also try direct adversarial perturbation as a more targeted approach
        print("\nTrying direct adversarial perturbation first...")
        
        # Create an adversarial perturbation directly pushing toward the target class
        # This adds noise specifically in one channel (e.g., making it "brighter" in one channel)
        adversarial_image = image.copy()
        
        # Try different perturbation strengths
        for strength in [50.0, 100.0, 150.0, 200.0]:
            # Apply a structured perturbation (not random) - try different patterns
            patterns = [
                # Central square
                lambda img: img.at[40:88, 40:88, 0].add(strength),
                # Horizontal bar
                lambda img: img.at[54:74, :, 1].add(strength),
                # Vertical bar
                lambda img: img.at[:, 54:74, 2].add(strength),
                # Checkerboard
                lambda img: img.at[::2, ::2, :].add(strength)
            ]
            
            for i, pattern_fn in enumerate(patterns):
                # Apply the pattern and clip to valid range
                test_image = jnp.clip(pattern_fn(adversarial_image), 0, 255)
                
                # Get prediction for this perturbation
                _, test_probs = predict_image(model, test_image)
                target_prob = test_probs[target_label].item() if hasattr(test_probs[target_label], 'item') else float(test_probs[target_label])
                
                # Calculate the log loss (make sure to convert to Python scalar)
                test_loss = -jnp.log(test_probs[target_label] + 1e-10)
                test_loss_val = test_loss.item() if hasattr(test_loss, 'item') else float(test_loss)
                
                print(f"Pattern {i+1}, strength {strength}: Target prob: {target_prob:.6f}, Loss: {test_loss_val:.6f}")
                
                # If this improves target probability, use it as starting point
                if target_prob > 0.01:  # Even a small probability is better than 0
                    print(f"Found promising perturbation! Using as starting point.")
                    current_image = test_image
                    # Get the new prediction with this image
                    _, current_probs = predict_image(model, current_image)
                    saved_images.append(current_image.copy())
                    saved_probs.append(current_probs)
                    
                    # Update loss value (safely converted to Python scalar)
                    current_loss_value = -jnp.log(current_probs[target_label] + 1e-10)
                    if hasattr(current_loss_value, 'item'):
                        current_loss_value = current_loss_value.item()
                    else:
                        current_loss_value = float(current_loss_value)
                    saved_losses.append(current_loss_value)
                    break
            
            # If we found a good perturbation, break from the strength loop too
            if len(saved_images) > 1:
                break
                
        # Print status after trying direct perturbations
        if len(saved_images) > 1:
            print("Direct adversarial perturbation successful! Continuing with random search...")
        else:
            print("Direct adversarial perturbation failed. Trying pure random search...")
    
    # Perform gradient descent on the input image
    current_image = image.copy() if len(saved_images) <= 1 else saved_images[-1].copy()
    original_image = image.copy()
    
    for step in range(steps):
        if not use_random_search:
            # Standard gradient-based approach
            grads = grad_fn(current_image)
            
            # Debug: Print detailed gradient info
            if step % 10 == 0:
                grad_magnitude = jnp.mean(jnp.abs(grads))
                grad_min = jnp.min(grads)
                grad_max = jnp.max(grads)
                non_zero = jnp.sum(jnp.abs(grads) > 1e-10)
                print(f"Step {step}: Gradient stats - mean: {grad_magnitude:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}, non-zero: {non_zero}/{grads.size}")
            
            # Update image with much higher learning rate
            current_image = current_image - learning_rate * grads
        else:
            # Random search approach - add random perturbations
            if step % 10 == 0:
                print(f"Step {step}: Using random search approach")
            
            # Create a random perturbation using numpy instead of JAX to avoid tracer issues
            np_seed = step % 10000
            np.random.seed(np_seed)
            random_noise = np.random.normal(0, 5.0, current_image.shape)  # Using numpy instead
            
            # Convert to JAX array
            random_noise = jnp.array(random_noise)
            
            # Try the perturbation
            test_image = jnp.clip(current_image + random_noise, 0, 255)
            _, test_probs = predict_image(model, test_image)
            
            # Convert target probabilities to Python scalars
            test_target_prob = test_probs[target_label]
            if hasattr(test_target_prob, 'item'):
                test_target_prob = test_target_prob.item()
            current_target_prob = saved_probs[-1][target_label]
            if hasattr(current_target_prob, 'item'):
                current_target_prob = current_target_prob.item()
            
            # If the perturbation improves the target probability, keep it
            if test_target_prob > current_target_prob:
                current_image = test_image
                if step % 10 == 0:
                    print(f"  Improved target probability: {current_target_prob:.6f} → {test_target_prob:.6f}")
        
        # Clip to valid pixel range
        current_image = jnp.clip(current_image, 0, 255)
        
        # Save the current image
        saved_images.append(current_image.copy())
        
        # Get current prediction
        _, current_probs = predict_image(model, current_image)
        saved_probs.append(current_probs)
        
        # Calculate loss - ensure we convert JAX values to Python scalars
        current_loss_value = -jnp.log(current_probs[target_label] + 1e-10)
        if hasattr(current_loss_value, 'item'):
            current_loss_value = current_loss_value.item()
        saved_losses.append(float(current_loss_value))
        
        # Print progress periodically
        if (step + 1) % 10 == 0 or step == 0:
            # Convert JAX array to NumPy or Python primitive
            if hasattr(current_probs, 'numpy'):
                current_probs_np = current_probs.numpy()
            else:
                current_probs_np = np.array(current_probs)
            
            # Get prediction class using NumPy
            pred_idx = np.argmax(current_probs_np)
            # Convert to Python primitive for dict lookup
            if hasattr(pred_idx, 'item'):
                pred_idx = pred_idx.item()
            else:
                pred_idx = int(pred_idx)
                
            pred_name = reverse_label_mapping[pred_idx]
            
            # Convert values to Python primitives for safe printing
            curr_pred_prob = float(current_probs_np[pred_idx])
            curr_target_prob = float(current_probs_np[target_label])
            curr_loss_val = float(current_loss_value)
            
            print(f"Step {step+1}/{steps}: Predicted as {pred_name} "
                  f"(p={curr_pred_prob:.4f}), Target {target_class} "
                  f"(p={curr_target_prob:.4f}), Loss: {curr_loss_val:.4f}")
            
            # Save intermediate images periodically and check if they're changing
            if (step + 1) % 20 == 0:
                # Convert to NumPy for PIL
                if hasattr(current_image, 'numpy'):
                    img_array = current_image.numpy().astype(np.uint8)
                else:
                    img_array = np.array(current_image).astype(np.uint8)
                    
                Image.fromarray(img_array).save(
                    os.path.join(save_dir, f"step_{step+1}.png"))
        
        if step % 10 == 0:
            # Calculate image difference
            image_diff_jax = jnp.sum(jnp.abs(current_image - original_image))
            # Convert to Python primitive
            if hasattr(image_diff_jax, 'item'):
                image_diff = image_diff_jax.item()
            else:
                image_diff = float(image_diff_jax)
                
            print(f"Step {step}: Total image change: {image_diff:.2f}")
    
    # Save final results
    
    # Save key images: original, middle, final
    plt.figure(figsize=(15, 5))
    
    # Function to convert JAX array to NumPy for matplotlib
    def to_numpy_image(img):
        if hasattr(img, 'numpy'):
            return img.numpy().astype(np.uint8)
        else:
            return np.array(img).astype(np.uint8)
    
    # Function to safely extract probabilities
    def get_safe_prob(probs, idx):
        if hasattr(probs, 'numpy'):
            probs_np = probs.numpy()
        else:
            probs_np = np.array(probs)
        return float(probs_np[idx])
    
    # Function to safely get prediction index
    def get_pred_idx(probs):
        if hasattr(probs, 'numpy'):
            probs_np = probs.numpy()
        else:
            probs_np = np.array(probs)
        pred_idx = np.argmax(probs_np)
        return pred_idx.item() if hasattr(pred_idx, 'item') else int(pred_idx)
    
    plt.subplot(1, 3, 1)
    plt.imshow(to_numpy_image(saved_images[0]))
    plt.title(f"Original\n{source_class} (p={get_safe_prob(saved_probs[0], source_label):.4f})")
    plt.axis('off')
    
    middle_idx = steps // 2
    plt.subplot(1, 3, 2)
    plt.imshow(to_numpy_image(saved_images[middle_idx]))
    mid_pred = get_pred_idx(saved_probs[middle_idx])
    mid_name = reverse_label_mapping[mid_pred]
    plt.title(f"Step {middle_idx}\n{mid_name} (p={get_safe_prob(saved_probs[middle_idx], mid_pred):.4f})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(to_numpy_image(saved_images[-1]))
    final_pred = get_pred_idx(saved_probs[-1])
    final_name = reverse_label_mapping[final_pred]
    plt.title(f"Final (Step {steps})\n{final_name} (p={get_safe_prob(saved_probs[-1], final_pred):.4f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "image_progression.png"))
    
    # Plot probabilities over time - convert JAX array to NumPy
    plt.figure(figsize=(10, 6))
    if hasattr(saved_probs[0], 'numpy'):
        # Convert list of JAX arrays to NumPy
        probs_list = [p.numpy() if hasattr(p, 'numpy') else np.array(p) for p in saved_probs]
        probs_array = np.array(probs_list)
    else:
        # Already NumPy arrays or convertible
        probs_array = np.array(saved_probs)
    
    for class_name, class_idx in label_mapping.items():
        plt.plot(probs_array[:, class_idx], label=class_name)
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Step')
    plt.ylabel('Probability')
    plt.title(f'Class Probabilities During Counterfactual: {source_class} → {target_class}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "probabilities.png"))
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(saved_losses, color='red')
    plt.xlabel('Step')
    plt.ylabel('Loss (-log probability of target class)')
    plt.title(f'Counterfactual Loss: {source_class} → {target_class}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    
    # Save data to CSV
    results_data = {
        'step': list(range(steps + 1)),
        'loss': saved_losses
    }
    
    # Add probabilities for each class - ensure we convert JAX arrays to NumPy
    for class_name, class_idx in label_mapping.items():
        # Extract the probability column and ensure it's a regular Python list
        class_probs = [float(p[class_idx]) for p in probs_array]
        results_data[f'prob_{class_name}'] = class_probs
    
    # Save as CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(save_dir, "counterfactual_data.csv"), index=False)
    
    print(f"Counterfactual analysis complete. Results saved to {save_dir}")
    return saved_images, saved_probs, saved_losses

def create_temperature_scaled_model(model, temperature=10.0):
    """
    Creates a wrapped model that applies temperature scaling to soften predictions.
    
    Args:
        model: The original model
        temperature: Temperature parameter (higher = softer predictions)
    
    Returns:
        A wrapped model function with temperature scaling
    """
    def scaled_model(image, training=False):
        # Forward pass through the original model
        logits = model(image, training=training)
        
        # Apply temperature scaling to logits
        # This divides the logits by temperature before softmax
        # Higher temperature = softer probability distribution
        scaled_logits = logits / temperature
        
        return scaled_logits
    
    return scaled_model

def predict_with_temperature(model, image, temperature=10.0):
    """Get model predictions with temperature scaling applied."""
    import jax.numpy as jnp
    import jax.nn as nn
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image[None, ...]
    
    # Get model predictions (logits)
    logits = model(image, training=False)
    
    # Apply temperature scaling and convert to probabilities
    scaled_logits = logits / temperature
    probs = nn.softmax(scaled_logits, axis=-1)
    
    # Get the predicted class
    pred_class = jnp.argmax(probs, axis=-1)[0]
    
    return pred_class, probs[0]

def visualize_model_activations(model, image, save_path=None):
    """
    Visualize the model's activation maps to see what features it focuses on.
    Uses guided backpropagation to create a saliency map.
    
    Args:
        model: The model to analyze
        image: Input image as JAX array (H, W, C)
        save_path: Optional path to save the visualization
    """
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image_batch = image[None, ...]
    else:
        image_batch = image
    
    # Define a function to compute gradients w.r.t input
    def get_gradients(image):
        # Forward pass
        def forward(x):
            logits = model(x, training=False)
            pred_class = jnp.argmax(logits, axis=-1)[0]
            return logits[0, pred_class]  # Return the prediction score for the top class
        
        # Compute gradients
        return jax.grad(forward)(image)
    
    # Get the gradients
    try:
        grads = get_gradients(image_batch)
        
        # Take the absolute value and reduce to a single channel by taking the max across channels
        saliency_map = jnp.abs(grads[0])
        saliency_map = jnp.max(saliency_map, axis=-1)
        
        # Normalize to [0, 1] for visualization
        saliency_map = (saliency_map - jnp.min(saliency_map)) / (jnp.max(saliency_map) - jnp.min(saliency_map) + 1e-8)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.set_title("Original Image")
        ax1.imshow(image.astype(np.uint8))
        ax1.axis('off')
        
        # Saliency map
        ax2.set_title("Saliency Map")
        ax2.imshow(saliency_map, cmap='jet')
        ax2.axis('off')
        
        # Overlay saliency on original image
        ax3.set_title("Overlay")
        ax3.imshow(image.astype(np.uint8))
        ax3.imshow(saliency_map, cmap='jet', alpha=0.5)
        ax3.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return saliency_map
    except Exception as e:
        print(f"Error generating activation visualization: {str(e)}")
        return None

def verify_preprocessing_consistency(image_path):
    """
    Verify that our preprocessing matches what was done during training.
    
    Args:
        image_path: Path to a test image
    """
    print("\n=== VERIFYING PREPROCESSING CONSISTENCY ===")
    
    # Load the image using our preprocessing approach
    our_preprocessed = load_and_preprocess_image(image_path)
    
    # Convert from JAX array to NumPy if needed
    if hasattr(our_preprocessed, 'numpy'):
        our_preprocessed_np = our_preprocessed.numpy()
    else:
        our_preprocessed_np = np.array(our_preprocessed)
    
    # Load using an approach that mimics the training script exactly
    def training_script_preprocess(image_path, target_size=(128, 128)):
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Standardize using ImageNet means and stds (as in training script)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        return img_array  # Note: Training keeps values in standardized range
    
    training_preprocessed = training_script_preprocess(image_path)
    
    # Compare statistics with NumPy operations on NumPy arrays
    print("Our preprocessing:")
    print(f"  Range: [{np.min(our_preprocessed_np):.4f}, {np.max(our_preprocessed_np):.4f}]")
    print(f"  Mean: {np.mean(our_preprocessed_np):.4f}")
    print(f"  Std: {np.std(our_preprocessed_np):.4f}")
    
    print("\nTraining script preprocessing:")
    print(f"  Range: [{np.min(training_preprocessed):.4f}, {np.max(training_preprocessed):.4f}]")
    print(f"  Mean: {np.mean(training_preprocessed):.4f}")
    print(f"  Std: {np.std(training_preprocessed):.4f}")
    
    # Calculate difference with NumPy operations on NumPy arrays
    diff = np.abs(our_preprocessed_np/255.0 - training_preprocessed)  # Normalize ours back to compare
    print(f"\nMax absolute difference: {np.max(diff):.8f}")
    print(f"Mean absolute difference: {np.mean(diff):.8f}")
    
    # Note about model's internal normalization
    print("\nNOTE: The model internally divides inputs by 255.0 before processing")
    print("Our preprocessing: Applies ImageNet normalization then scales back to [0-255]")
    print("Training script: Applies ImageNet normalization and keeps values in normalized range")
    print("This means when the model divides by 255.0, our preprocessing needs compensation")
    
    print("===================================================\n")
    
    return our_preprocessed, training_preprocessed

def analyze_model_predictions(model, base_dir="DB - Copy", num_samples=10):
    """
    Analyze the model's prediction patterns across multiple classes to check for saturation issues.
    
    Args:
        model: The loaded model
        base_dir: Base directory containing the dataset
        num_samples: Number of samples to check per class
    """
    print("\n==== MODEL PREDICTION PATTERN ANALYSIS ====")
    
    all_probs = []
    saturated_count = 0
    total_count = 0
    
    # Check each class
    for class_name, class_idx in label_mapping.items():
        dir_name = class_to_dir[class_name]
        class_dir = os.path.join(base_dir, "training", dir_name)
        
        # List all images
        image_files = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Take up to num_samples images
        samples = image_files[:num_samples]
        print(f"\nAnalyzing {len(samples)} '{class_name}' images:")
        
        for img_file in samples:
            img_path = os.path.join(class_dir, img_file)
            image = load_and_preprocess_image(img_path)
            _, probs = predict_image(model, image)
            
            # Convert JAX array to NumPy for analysis
            # Always ensure we convert JAX arrays to NumPy before using NumPy operations
            if hasattr(probs, 'numpy'):
                probs_np = probs.numpy()  # For newer JAX versions
            else:
                probs_np = np.array(probs)  # Fallback
            
            all_probs.append(probs_np)
            
            # Check if probabilities are saturated using NumPy operations on NumPy arrays
            is_saturated = np.any(probs_np > 0.99) or np.any(probs_np < 0.01)
            total_count += 1
            if is_saturated:
                saturated_count += 1
            
            # Format output - convert values to Python primitives for string formatting
            probs_str = ", ".join([f"{name}: {float(p):.4f}" for name, p in zip(label_mapping.keys(), probs_np)])
            sat_marker = "⚠️ SATURATED" if is_saturated else ""
            print(f"  {img_file}: {probs_str} {sat_marker}")
    
    # Overall statistics - we're working with NumPy arrays at this point
    all_probs = np.vstack(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)
    
    print("\nOverall Statistics:")
    print(f"Samples with saturated probabilities: {saturated_count}/{total_count} ({saturated_count/total_count*100:.1f}%)")
    
    print("\nProbability distributions by class:")
    for i, class_name in enumerate(label_mapping.keys()):
        print(f"  {class_name}: mean={mean_probs[i]:.4f}, std={std_probs[i]:.4f}")
    
    # Visualize the probability distributions with Matplotlib (using NumPy arrays)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(label_mapping.keys()):
        plt.hist(all_probs[:, i], alpha=0.6, bins=20, label=class_name)
    
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Model Probability Distributions')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Use a timestamp for the filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"model_probability_analysis_{timestamp}.png")
    
    print("\nDIAGNOSIS:")
    if saturated_count / total_count > 0.7:
        print("  The model shows strong probability saturation (very high confidence)")
        print("  This makes gradient-based counterfactual analysis difficult as there's no 'smoothness'")
        print("  RECOMMENDATION: Apply temperature scaling to soften the model's predictions")
    else:
        print("  The model shows reasonable probability distributions")
        print("  The issue might be with specific model operations blocking gradient flow")
        print("  RECOMMENDATION: Check for non-differentiable operations in the model")
    
    print("=========================================")

# Report explaining the core issues found with counterfactual analysis on this model
def explain_counterfactual_issues():
    """Print a detailed explanation of why counterfactual analysis might be failing on this model."""
    print("\n==== MODEL RELIABILITY ISSUES FOR COUNTERFACTUAL ANALYSIS ====")
    print("""
    The counterfactual analysis is failing for several fundamental reasons:

    1. MODEL INSENSITIVITY
       The model shows no change in outputs even when inputs are significantly
       perturbed. This indicates the model has either become too confident in its
       predictions or has learned features that don't vary with the kinds of input
       changes we're making.

    2. ZERO GRADIENTS
       The model produces zero gradients when trying to optimize inputs, making
       gradient-based methods impossible to use. This is likely because the model
       has learned to make very hard decisions with no ambiguity, leading to
       saturated probabilities of 0 or 1.

    3. LOW MODEL ACCURACY
       The model has poor accuracy (around 30%) on the source class images, indicating
       it's not reliably identifying the patterns it should. Counterfactual analysis
       requires a well-trained model that can accurately classify the source images.

    4. NUMERICAL STABILITY ISSUES
       JAX's automatic differentiation might be having issues computing meaningful 
       gradients through the model's computation graph, particularly if the model
       contains operations that are numerically unstable.

    RECOMMENDATION:
    For counterfactual analysis to work properly, you should:
      - Consider retraining the model with regularization to avoid overfitting
      - Add temperature scaling to soften the model's predictions
      - Verify the model achieves good classification accuracy (>80%) on all classes
      - Try a simpler model architecture that supports smooth gradient computation
    """)
    print("==============================================================\n")

# This section contains the main execution code
if __name__ == "__main__":
    # Path to the directory containing your saved model
    model_dir = "model_save_20250407_134634_db0b8816"  # Replace with your model directory
    
    # Path to your dataset
    base_dir = "/Users/ddifrancesco/Github/model risk/weld_classification/DB - Copy"  # Replace with your dataset directory
    
    # Find a test image for preprocessing verification
    test_image_path = None
    for class_dir in ["NoDifetto", "Difetto1", "Difetto2", "Difetto4"]:
        dir_path = os.path.join(base_dir, "training", class_dir)
        if os.path.exists(dir_path):
            image_files = [f for f in os.listdir(dir_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                test_image_path = os.path.join(dir_path, image_files[0])
                break
    
    if test_image_path:
        # Verify preprocessing consistency
        verify_preprocessing_consistency(test_image_path)
    
    # Load the model
    model = load_dill(model_dir)
    
    # Explain the core issues found with counterfactual analysis
    explain_counterfactual_issues()
    
    # Analyze model prediction patterns
    analyze_model_predictions(model, base_dir, num_samples=5)
    
    # Ask user if they want to run with temperature scaling
    print("\nRunning counterfactual analysis with and without temperature scaling:")
    print("1. First with standard model")
    print("2. Then with temperature scaling (T=10) to soften predictions")
    print("\n========== STANDARD MODEL ANALYSIS ==========")
    
    # Try without temperature scaling (standard approach)
    standard_results = run_counterfactual_analysis(
        model=model,
        source_class="anomaly type 1",
        target_class="no anomaly",
        base_dir=base_dir,
        steps=25,  # Reduced for faster debugging
        learning_rate=500000.0,
        temperature=None  # No temperature scaling
    )
    
    print("\n========== TEMPERATURE SCALED MODEL ANALYSIS ==========")
    print("Running with temperature scaling to soften model predictions")
    
    # Then try with temperature scaling
    scaled_results = run_counterfactual_analysis(
        model=model,
        source_class="anomaly type 1", 
        target_class="no anomaly",
        base_dir=base_dir,
        steps=25,  # Reduced for faster debugging
        learning_rate=500000.0,
        temperature=10.0  # Apply temperature scaling
    )
    
    # Visualize results if any were successful
    if standard_results is not None:
        print("\nVisualizing standard model results:")
        saved_images, saved_probs, _ = standard_results
        visualize_counterfactual_steps(
            saved_images, 
            saved_probs, 
            num_steps=10, 
            save_path="counterfactual_standard_visualization.png"
        )
    
    if scaled_results is not None:
        print("\nVisualizing temperature-scaled model results:")
        saved_images, saved_probs, _ = scaled_results
        visualize_counterfactual_steps(
            saved_images, 
            saved_probs, 
            num_steps=10, 
            save_path="counterfactual_temperature_scaled_visualization.png"
        )

def run_counterfactual_analysis(model, source_class, target_class, base_dir="DB - Copy",
                             steps=1000, learning_rate=500000.0, temperature=None):
    """Run the complete counterfactual analysis pipeline."""
    # Apply temperature scaling if requested
    if temperature is not None and temperature > 1.0:
        print(f"\nUsing temperature scaling with T={temperature}")
        scaled_model = create_temperature_scaled_model(model, temperature)
        working_model = scaled_model
    else:
        working_model = model  # Use original model
    
    # Debug: Check if the model can properly classify images from source_class
    print(f"\n=== DEBUGGING: Model Performance on {source_class} ===")
    dir_name = class_to_dir[source_class]
    class_dir = os.path.join(base_dir, "training", dir_name)
    
    # List all images in the directory
    image_files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    correct_count = 0
    total_tested = min(10, len(image_files))  # Test up to 10 images
    
    print(f"Testing model on {total_tested} sample images from {source_class} class...")
    
    for i, img_file in enumerate(image_files[:total_tested]):
        img_path = os.path.join(class_dir, img_file)
        image = load_and_preprocess_image(img_path)
        
        # Use appropriate prediction function based on temperature setting
        if temperature is not None and temperature > 1.0:
            pred_class, probs = predict_with_temperature(model, image, temperature)
        else:
            pred_class, probs = predict_image(model, image)
            
        # Convert JAX array to a standard Python integer for dictionary lookup
        if hasattr(pred_class, 'item'):
            pred_class_int = pred_class.item()
        else:
            pred_class_int = int(pred_class)
            
        pred_name = reverse_label_mapping[pred_class_int]
        expected_label = label_mapping[source_class]
        
        # Convert probability to Python primitive if needed
        pred_prob = probs[pred_class_int]
        if hasattr(pred_prob, 'item'):
            pred_prob = pred_prob.item()
        
        print(f"Image {i+1}: {img_file}")
        print(f"  Prediction: {pred_name} (p={pred_prob:.4f})")
        print(f"  Expected: {source_class}")
        
        if pred_class_int == expected_label:
            correct_count += 1
    
    accuracy = correct_count / total_tested if total_tested > 0 else 0
    print(f"\nAccuracy on {source_class} class: {correct_count}/{total_tested} ({accuracy*100:.1f}%)")
    
    if accuracy < 0.5:
        print(f"WARNING: Model has poor accuracy on {source_class} class!")
        print("This may explain why counterfactual generation is not working well.")
    print("=== END DEBUGGING ===\n")
    
    # Find a correctly classified image
    print(f"Finding a correctly classified image from class '{source_class}'...")
    
    # Need to implement a version that uses temperature scaling if needed
    if temperature is not None and temperature > 1.0:
        # Custom search for properly classified image with temperature scaling
        dir_name = class_to_dir[source_class]
        expected_label = label_mapping[source_class]
        class_dir = os.path.join(base_dir, "training", dir_name)
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            image = load_and_preprocess_image(img_path)
            pred_class, probs = predict_with_temperature(model, image, temperature)
            if hasattr(pred_class, 'item'):
                pred_class_int = pred_class.item()
            else:
                pred_class_int = int(pred_class)
            
            if pred_class_int == expected_label:
                # Convert probability to Python scalar
                pred_prob = probs[pred_class_int]
                if hasattr(pred_prob, 'item'):
                    pred_prob = pred_prob.item()
                    
                print(f"Found correctly classified image with temperature scaling: {img_file}")
                print(f"Prediction: {source_class} with probability {pred_prob:.4f}")
                image_path = img_path
                # Make sure to load the image
                image = load_and_preprocess_image(img_path)
                break
        else:
            print("Could not find a correctly classified image. Aborting.")
            return None
    else:
        # Use the standard function
        image_path, image = find_correctly_classified_image(model, source_class, base_dir)
        if image_path is None:
            print("Could not find a correctly classified image. Aborting.")
            return None
    
    # Create directory for results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(image_path).split('.')[0]
    temp_suffix = f"_temp{temperature}" if temperature is not None else ""
    save_dir = f"counterfactual_{source_class}_to_{target_class}{temp_suffix}_{image_name}_{timestamp}"
    
    # Execute the analysis with fewer steps for debugging
    results = perform_counterfactual_analysis(
        model=working_model,  # Use either scaled or original model
        image=image, 
        source_class=source_class,
        target_class=target_class,
        steps=steps,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    return results

def visualize_counterfactual_steps(saved_images, saved_probs, num_steps=10, save_path=None):
    """Create a visualization showing the image at evenly spaced steps."""
    total_steps = len(saved_images)
    indices = [0] + [i * total_steps // num_steps for i in range(1, num_steps)]
    
    # Ensure we include the last step
    if indices[-1] != total_steps - 1:
        indices[-1] = total_steps - 1
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8)) if num_steps >= 10 else plt.subplots(1, num_steps, figsize=(4*num_steps, 4))
    axes = axes.flatten()
    
    # Function to convert standardized image to display format
    def to_display_format(img):
        # First convert JAX array to NumPy if needed
        if hasattr(img, 'numpy'):
            img_np = img.numpy()
        else:
            img_np = np.array(img)
            
        # Check if image is already in display range using NumPy operations
        if np.max(np.abs(img_np)) > 5.0:
            return img_np.astype(np.uint8)
        
        # Un-standardize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_img = img_np * std + mean
        # Scale to 0-255
        display_img = display_img * 255.0
        # Clip to valid range and convert to uint8
        return np.clip(display_img, 0, 255).astype(np.uint8)
    
    for i, idx in enumerate(indices):
        if i < len(axes):
            # Display image - convert from standardized to display format
            axes[i].imshow(to_display_format(saved_images[idx]))
            
            # Get prediction - convert JAX array to Python primitive for dictionary lookup
            if hasattr(saved_probs[idx], 'numpy'):
                prob_np = saved_probs[idx].numpy()
            else:
                prob_np = np.array(saved_probs[idx])
                
            pred_idx = np.argmax(prob_np)
            
            # Ensure we have a Python integer for dict lookup
            if hasattr(pred_idx, 'item'):
                pred_idx = pred_idx.item()
            else:
                pred_idx = int(pred_idx)
                
            pred_name = reverse_label_mapping[pred_idx]
            confidence = float(prob_np[pred_idx])  # Convert to Python float for formatting
            
            # Set title
            step_label = "Original" if idx == 0 else f"Step {idx}"
            axes[i].set_title(f"{step_label}\n{pred_name}\n(p={confidence:.4f})")
            axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()