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
    def __init__(self, model, label_mapping, learning_rate=0.01, steps=1000):
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
                
                # Print progress periodically
                if (step + 1) % 10 == 0 or step == 0:
                    # Get prediction class safely with stop_gradient
                    safe_probs = jax.lax.stop_gradient(probabilities[-1])
                    pred_idx = int(jnp.argmax(safe_probs))
                    pred_name = self.reverse_mapping.get(pred_idx, f"Class {pred_idx}")
                    
                    # Get probability values safely
                    curr_pred_prob = float(safe_probs[pred_idx])
                    curr_target_prob = float(safe_probs[target_class])
                    curr_loss = float(jax.lax.stop_gradient(loss_fn(image, target_class)))
                    
                    print(f"Step {step+1}/{self.steps}: Predicted as {pred_name} "
                          f"(p={curr_pred_prob:.4f}), Target {target_name} "
                          f"(p={curr_target_prob:.4f}), Loss: {curr_loss:.4f}")
            
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
            
            # Print progress periodically
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step+1}/{self.steps}: Predicted as {pred_name} "
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
                          learning_rate=0.01, steps=1000, save_dir = "counterfactual_results"):
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

def analyze_and_visualize(model, image_path, true_class_name, save_dir=None, learning_rate=0.01, steps=1000):
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
            learning_rate=0.01,
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
                                  steps=1000, learning_rate=0.01, save_dir="counterfactual_results"):
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
    
    # Fix for initial output
    # Stop gradients for safe output and get scalar values for printing
    initial_probs_safe = jax.lax.stop_gradient(initial_probs)
    source_prob = float(initial_probs_safe[source_label])
    target_prob = float(initial_probs_safe[target_label])
    
    print(f"Starting counterfactual analysis: {source_class} → {target_class}")
    print(f"Initial probabilities: {source_class}: {source_prob:.4f}, "
          f"{target_class}: {target_prob:.4f}")
    
    # Calculate initial loss (negative log probability of target class)
    initial_loss = -jnp.log(initial_probs[target_label] + 1e-10)
    # Convert to Python scalar with stop_gradient to avoid tracer issues
    initial_loss_val = float(jax.lax.stop_gradient(initial_loss))
    print(f"Initial loss: {initial_loss_val:.4f}")
    saved_losses.append(initial_loss_val)
    
    # Define loss function focusing directly on maximizing target class probability
    def loss_fn(img):
        # Forward pass (add batch dimension)
        if len(img.shape) == 3:
            img_batch = img[None, ...]
        else:
            img_batch = img
            
        # Use the wrapped model
        logits = wrapped_model(img_batch, training=False)
        probs = jax.nn.softmax(logits, axis=-1)[0]
        
        # Simple negative log probability loss - clean and direct
        loss = -jnp.log(probs[target_label] + 1e-10)
        
        # Higher regularization weight to make changes more gradual and visible
        reg_weight = 0.001  # Increased from 0.0001
        l2_penalty = jnp.sum((img - image) ** 2)
        
        return loss + reg_weight * l2_penalty
    
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
    
    # Skip BatchNorm and JIT checks for simplicity
    
    # For gradient approach when gradients are very small, use larger learning rate
    if non_zero < initial_grads.size * 0.01:  # If less than 1% of gradients are non-zero
        learning_rate = learning_rate * 5.0  # Increase learning rate but don't go too extreme
        print(f"Increasing learning rate to {learning_rate:.1f} to compensate for small gradients.")
    
    # Simple gradient descent without any manual perturbation
    current_image = image.copy()
    original_image = image.copy()
    
    # Perform gradient descent
    for step in range(steps):
        # Standard gradient-based approach
        grads = grad_fn(current_image)
        
        # Debug: Print detailed gradient info
        if step % 10 == 0:
            grad_magnitude = jnp.mean(jnp.abs(grads))
            grad_min = jnp.min(grads)
            grad_max = jnp.max(grads)
            non_zero = jnp.sum(jnp.abs(grads) > 1e-10)
            print(f"Step {step}: Gradient stats - mean: {grad_magnitude:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}, non-zero: {non_zero}/{grads.size}")
        
        # Update image with current learning rate
        current_image = current_image - learning_rate * grads
        
        # Clip to valid pixel range (in standardized space)
        current_image = jnp.clip(current_image, -3.0, 3.0)  # Clip to reasonable standardized range
        
        # Save the current image
        saved_images.append(current_image.copy())
        
        # Get current prediction
        _, current_probs = predict_image(model, current_image)
        saved_probs.append(current_probs)
        
        # Calculate loss - ensure we convert JAX values to Python scalars
        current_loss_value = loss_fn(current_image)
        if hasattr(current_loss_value, 'item'):
            current_loss_value = current_loss_value.item()
        saved_losses.append(float(current_loss_value))
        
        # Print progress periodically
        if (step + 1) % 10 == 0 or step == 0:
            # Get prediction class safely with stop_gradient
            safe_probs = jax.lax.stop_gradient(current_probs)
            pred_idx = int(jnp.argmax(safe_probs))
            pred_name = reverse_label_mapping[pred_idx]
            
            # Get probability values safely
            curr_pred_prob = float(safe_probs[pred_idx])
            curr_target_prob = float(safe_probs[target_label])
            curr_loss = float(jax.lax.stop_gradient(current_loss_value))
            
            print(f"Step {step+1}/{steps}: Predicted as {pred_name} "
                  f"(p={curr_pred_prob:.4f}), Target {target_class} "
                  f"(p={curr_target_prob:.4f}), Loss: {curr_loss:.4f}")
        
        # Save intermediate images more frequently
        if (step + 1) % 5 == 0 or step == 0:  # Every 5 steps instead of 20
            # Convert to display format for saving
            img_array = to_display_format(current_image)
            Image.fromarray(img_array).save(os.path.join(save_dir, f"step_{step+1}.png"))
    
    # Save visualization of the results
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(1, 3, 1)
    plt.imshow(to_display_format(saved_images[0]))
    safe_probs = jax.lax.stop_gradient(saved_probs[0])
    plt.title(f"Original\n{source_class} (p={float(safe_probs[source_label]):.4f})")
    plt.axis('off')
    
    # Show middle step
    middle_idx = len(saved_images) // 2
    plt.subplot(1, 3, 2)
    plt.imshow(to_display_format(saved_images[middle_idx]))
    safe_probs = jax.lax.stop_gradient(saved_probs[middle_idx])
    mid_pred = int(jnp.argmax(safe_probs))
    mid_name = reverse_label_mapping[mid_pred]
    plt.title(f"Step {middle_idx}\n{mid_name} (p={float(safe_probs[mid_pred]):.4f})")
    plt.axis('off')
    
    # Show final image
    plt.subplot(1, 3, 3)
    plt.imshow(to_display_format(saved_images[-1]))
    safe_probs = jax.lax.stop_gradient(saved_probs[-1])
    final_pred = int(jnp.argmax(safe_probs))
    final_name = reverse_label_mapping[final_pred]
    plt.title(f"Final (Step {steps})\n{final_name} (p={float(safe_probs[final_pred]):.4f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "image_progression.png"))
    
    # Plot probabilities over time
    plt.figure(figsize=(10, 6))
    probs_array = []
    for p in saved_probs:
        # Convert to numpy safely
        p_safe = jax.lax.stop_gradient(p)
        if hasattr(p_safe, 'numpy'):
            probs_array.append(p_safe.numpy())
        else:
            probs_array.append(np.array([float(p_safe[i]) for i in range(len(p_safe))]))
    
    # Convert to numpy array
    probs_array = np.array(probs_array)
    
    # Plot each class probability
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
    plt.ylabel('Loss')
    plt.title(f'Counterfactual Loss: {source_class} → {target_class}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    
    # Save results to CSV
    results_data = {
        'step': list(range(len(saved_losses))),
        'loss': saved_losses
    }
    
    # Add class probabilities
    for class_name, class_idx in label_mapping.items():
        results_data[f'prob_{class_name}'] = [float(p[class_idx]) for p in probs_array]
    
    # Print final statistics
    print(f"\nFinal results:")
    print(f"  Initial loss: {saved_losses[0]:.4f}")
    print(f"  Final loss: {saved_losses[-1]:.4f}")
    print(f"  Change: {saved_losses[-1] - saved_losses[0]:.4f}")
    
    # Save to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(save_dir, "counterfactual_data.csv"), index=False)
    
    print(f"Counterfactual analysis complete. Results saved to {save_dir}")
    return saved_images, saved_probs, saved_losses

def find_unsaturated_example(model, class_name="no anomaly", base_dir="DB - Copy", max_confidence=0.95):
    """
    Find an image that is correctly classified as the specified class but with probability less than max_confidence.
    This helps avoid completely saturated predictions that lead to zero gradients.
    
    Args:
        model: The trained model
        class_name: Target class to find examples for
        base_dir: Dataset directory
        max_confidence: Maximum confidence threshold to consider
        
    Returns:
        Tuple of (image_path, image) or (None, None) if no suitable example found
    """
    # Get directory name for the class
    dir_name = class_to_dir[class_name]  # Use the reverse mapping
    expected_label = label_mapping[class_name]
    
    # Path to class directory
    class_dir = os.path.join(base_dir, "training", dir_name)
    
    # List all images in the directory
    image_files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Searching through {len(image_files)} images for a non-saturated example...")
    
    # Check each image
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        image = load_and_preprocess_image(img_path)
        pred_class, probs = predict_image(model, image)
        
        # Convert JAX array to a standard Python integer
        pred_class_int = pred_class.item() if hasattr(pred_class, 'item') else int(pred_class)
        
        # Check if correctly classified
        if pred_class_int == expected_label:
            # Convert probability to Python float
            pred_prob = float(probs[pred_class_int]) if hasattr(probs[pred_class_int], 'item') else float(probs[pred_class_int])
            
            # Check if probability is less than max_confidence
            if pred_prob < max_confidence:
                print(f"Found non-saturated correctly classified image: {img_file}")
                print(f"Prediction: {class_name} with probability {pred_prob:.4f}")
                return img_path, image
            
    print("No suitable non-saturated examples found.")
    return None, None

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
    diff = np.abs(our_preprocessed_np - training_preprocessed)
    print(f"\nMax absolute difference: {np.max(diff):.8f}")
    print(f"Mean absolute difference: {np.mean(diff):.8f}")
    
    # Note about model's internal normalization
    print("\nNOTE: The model internally divides inputs by 255.0 before processing")
    print("Our preprocessing: Applies ImageNet normalization then keeps values in normalized range")
    print("Training script: Applies ImageNet normalization and keeps values in normalized range")
    
    print("===================================================\n")
    
    return our_preprocessed, training_preprocessed

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
    For counterfactual analysis to work properly with non-saturated examples, we will:
      - Find examples where the model's prediction probabilities are not completely saturated
      - Use a clean gradient-based approach with proper regularization
      - Ensure stable gradient flow through the network
    """)
    print("==============================================================\n")

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
        print("  RECOMMENDATION: We'll focus on finding non-saturated examples for counterfactual analysis")
    else:
        print("  The model shows reasonable probability distributions")
        print("  The issue might be with specific model operations blocking gradient flow")
        print("  RECOMMENDATION: We'll proceed with gradient-based counterfactual analysis")
    
    print("=========================================\n")

def visualize_counterfactual_steps(saved_images, saved_probs, num_steps=15, save_path=None):
    """Create a visualization showing the image at evenly spaced steps."""
    total_steps = len(saved_images)
    
    # For better visualization of the transition, use linearly spaced indices
    indices = [0]  # Start with the first image
    
    # Find where probabilities flip (if they do)
    flip_index = None
    prev_pred = None
    for i in range(total_steps):
        safe_probs = jax.lax.stop_gradient(saved_probs[i])
        if hasattr(safe_probs, 'numpy'):
            prob_np = safe_probs.numpy()
        else:
            prob_np = np.array([float(safe_probs[j]) for j in range(len(safe_probs))])
        
        curr_pred = np.argmax(prob_np)
        
        if prev_pred is not None and curr_pred != prev_pred:
            flip_index = i
            break
        
        prev_pred = curr_pred
    
    # If we found a flip, concentrate indices around it
    if flip_index:
        print(f"Found prediction flip at step {flip_index}")
        # Add indices before the flip
        before_flip = max(1, flip_index - 2)
        indices.extend(list(range(max(1, before_flip), flip_index)))
        
        # Add the flip index and a few after
        indices.extend(list(range(flip_index, min(flip_index + 3, total_steps))))
        
        # Add remaining indices spread out
        remaining = num_steps - len(indices)
        if remaining > 0 and flip_index + 3 < total_steps:
            step_size = (total_steps - (flip_index + 3)) / remaining
            indices.extend([int(flip_index + 3 + i * step_size) for i in range(remaining)])
    else:
        # No flip found, use evenly spaced indices
        indices.extend([i * total_steps // (num_steps-1) for i in range(1, num_steps)])
    
    # Ensure we include the last step and remove duplicates
    if indices[-1] != total_steps - 1:
        indices.append(total_steps - 1)
    indices = sorted(list(set(indices)))
    
    # Create figure with 3 rows if needed
    num_cols = min(5, len(indices))
    num_rows = (len(indices) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])  # Make it indexable
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
            # Create a 2x1 subplot grid within each axes position
            axes[i].remove()  # Remove the original axes
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            
            # Get the image and convert to display format
            img = to_display_format(saved_images[idx])
            ax.imshow(img)
            
            # Get prediction - safely stop gradients before conversion
            safe_probs = jax.lax.stop_gradient(saved_probs[idx])
            
            # Convert JAX array to Python primitive for dictionary lookup
            if hasattr(safe_probs, 'numpy'):
                prob_np = safe_probs.numpy()
            else:
                prob_np = np.array([float(safe_probs[j]) for j in range(len(safe_probs))])
                
            pred_idx = np.argmax(prob_np)
            pred_name = reverse_label_mapping[pred_idx]
            confidence = float(prob_np[pred_idx])  # Convert to Python float for formatting
            
            # Set title with more emphasis on transition steps
            step_label = "Original" if idx == 0 else f"Step {idx}"
            if flip_index and idx == flip_index:
                # Highlight the transition step in red
                ax.set_title(f"TRANSITION\n{step_label}\n{pred_name}\n(p={confidence:.4f})", color='red')
                # Add a red border to the image
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            else:
                ax.set_title(f"{step_label}\n{pred_name}\n(p={confidence:.4f})")
            ax.axis('off')
            
            # Add a small inset showing the difference from original image
            if idx > 0:  # Skip for the first image
                # Create an inset axes in the bottom right corner
                axins = ax.inset_axes([0.65, 0.65, 0.35, 0.35])
                
                # Calculate absolute difference from original image
                original_img = to_display_format(saved_images[0])
                diff = np.abs(img.astype(np.float32) - original_img.astype(np.float32))
                
                # Enhance contrast by scaling the difference
                scaling_factor = 10  # Amplify differences by 10x
                enhanced_diff = np.clip(diff * scaling_factor, 0, 255).astype(np.uint8)
                
                # Show the difference image with a different colormap to highlight changes
                axins.imshow(enhanced_diff, cmap='hot')
                axins.set_title(f"Diff (x{scaling_factor})", fontsize=8)
                axins.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Path to the directory containing your saved model
    model_dir = "model_save_20250324_143403_24db4de6"  # Replace with your model directory
    
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
    
    # Run proper gradient-based counterfactual analysis with non-saturated examples
    print("\nLooking for non-saturated examples to enable proper gradient-based counterfactual analysis...")
    
    # Search for non-saturated examples of anomaly type 1
    image_path, image = find_unsaturated_example(
        model=model,
        class_name="anomaly type 1", 
        base_dir=base_dir,
        max_confidence=0.95
    )
    
    if image_path is None:
        # Try with a higher threshold if we can't find any below 0.99
        print("Trying with a higher saturation threshold (0.999)...")
        image_path, image = find_unsaturated_example(
            model=model,
            class_name="anomaly type 1", 
            base_dir=base_dir,
            max_confidence=0.99
        )
    
    if image_path is not None:
        # Create directory for results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(image_path).split('.')[0]
        save_dir = f"gradual_counterfactual_anomaly_type_1_to_no_anomaly_{image_name}_{timestamp}"
        
        # Configure counterfactual analysis parameters
        print("Running proper gradient-based counterfactual analysis...")
        # Use MUCH smaller learning rate and more steps for gradual transition
        learning_rate = 0.01  # Much smaller learning rate
        steps = 100           # More steps
        
        # Use the clean counterfactual analysis with JAX's autograd
        results = perform_counterfactual_analysis(
            model=model,
            image=image,
            source_class="anomaly type 1",
            target_class="no anomaly",
            steps=steps,
            learning_rate=learning_rate,
            save_dir=save_dir
        )
        
        # Visualize results
        if results is not None:
            print("\nVisualizing counterfactual results:")
            saved_images, saved_probs, _ = results
            visualize_counterfactual_steps(
                saved_images,
                saved_probs,
                num_steps=15,
                save_path="gradual_counterfactual_visualization.png"
            )
    else:
        print("Could not find any non-saturated examples. Counterfactual analysis may not work properly.")
        print("Consider retraining the model with regularization to avoid extreme probability saturation.")



