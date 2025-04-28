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