import re

# Read the original file
with open("counterfactual_analysis.py", "r") as f:
    content = f.read()

# Find the main code section
main_section_match = re.search(r"if __name__ == \"__main__\":", content)
if main_section_match:
    main_section_start = main_section_match.start()
    
    # Extract functions defined after main
    functions_after_main = content[main_section_start:]
    
    # Extract the run_counterfactual_analysis function
    run_cf_match = re.search(r"def run_counterfactual_analysis\(.*?\):.*?return results", 
                       functions_after_main, re.DOTALL)
    
    if run_cf_match:
        run_cf_func = run_cf_match.group(0)
        
        # Extract the visualize_counterfactual_steps function
        vis_cf_match = re.search(r"def visualize_counterfactual_steps\(.*?\):.*?plt\.show\(\)", 
                           functions_after_main, re.DOTALL)
        
        if vis_cf_match:
            vis_cf_func = vis_cf_match.group(0)
            
            # 1. Fix the print progress area in perform_counterfactual_analysis
            fix1 = """# Print progress periodically
        if (step + 1) % 10 == 0 or step == 0:
            # Need to stop gradient optimization for printing
            # Get a safe copy of current_probs for analysis
            jax.lax.stop_gradient(current_probs)  # Make sure no gradients flow through here
            
            # Get prediction class - with stop_gradient to avoid tracer issues
            pred_idx = jnp.argmax(current_probs, axis=0)
            pred_idx = int(jax.lax.stop_gradient(pred_idx))
            pred_name = reverse_label_mapping[pred_idx]
            
            # Get probabilities as Python scalars with stop_gradient
            curr_pred_prob = float(jax.lax.stop_gradient(current_probs[pred_idx]))
            curr_target_prob = float(jax.lax.stop_gradient(current_probs[target_label]))
            curr_loss_val = float(jax.lax.stop_gradient(current_loss_value))"""
            
            # 2. Fix the random search approach for target probabilities
            fix2 = """# If the perturbation improves the target probability, keep it
            test_target_prob = jax.lax.stop_gradient(test_probs[target_label])
            if hasattr(test_target_prob, 'item'):
                test_target_prob = test_target_prob.item()
            else:
                test_target_prob = float(jax.lax.stop_gradient(test_target_prob))
                
            current_target_prob = jax.lax.stop_gradient(saved_probs[-1][target_label])
            if hasattr(current_target_prob, 'item'):
                current_target_prob = current_target_prob.item()
            else:
                current_target_prob = float(jax.lax.stop_gradient(current_target_prob))"""
                
            # 3. Fix the helper visualization functions
            fix3 = """# Function to safely extract probabilities
    def get_safe_prob(probs, idx):
        # First stop gradients for safe extraction
        safe_probs = jax.lax.stop_gradient(probs)
        # Then extract the value as a float
        return float(safe_probs[idx])
    
    # Function to safely get prediction index
    def get_pred_idx(probs):
        # First stop gradients for safe operations
        safe_probs = jax.lax.stop_gradient(probs)
        # Get the index of maximum value
        pred_idx = jnp.argmax(safe_probs)
        # Convert to Python int
        return int(pred_idx)"""
                
            # 4. Fix the visualization code for plotting probabilities
            fix4 = """# Plot probabilities over time - convert JAX array to NumPy
    plt.figure(figsize=(10, 6))
    # Convert and detach from computation graph
    probs_array = []
    for p in saved_probs:
        # Safely convert to numpy first stopping gradient flow
        p_safe = jax.lax.stop_gradient(p)
        if hasattr(p_safe, 'numpy'):
            probs_array.append(p_safe.numpy())
        else:
            probs_array.append(np.array([float(p_safe[i]) for i in range(len(p_safe))]))
    
    # Now it's safe to work with the numpy arrays
    probs_array = np.array(probs_array)"""
                
            # 5. Fix the CSV conversion code - ensure all arrays have same length
            fix5 = """# Save data to CSV
    # Make sure all arrays have the same length - use actual number of steps completed
    actual_steps = len(saved_losses) - 1  # subtract 1 for initial state
    results_data = {
        'step': list(range(len(saved_losses))),
        'loss': saved_losses
    }
    
    # Add probabilities for each class - ensure we convert JAX arrays to NumPy safely
    # and ensure all arrays have the same length
    for class_name, class_idx in label_mapping.items():
        # Extract probability values for the completed steps only
        class_probs = []
        for p in probs_array[:len(saved_losses)]:  # Match length with saved_losses
            class_probs.append(float(p[class_idx]))
        results_data[f'prob_{class_name}'] = class_probs
        
    # Print final statistics
    print(f"Final results:")
    print(f"  Initial loss: {saved_losses[0]:.4f}")
    print(f"  Final loss: {saved_losses[-1]:.4f}")
    print(f"  Change: {saved_losses[-1] - saved_losses[0]:.4f}")"""
                
            # 6. Add a fix for the initial_loss conversion to float
            fix6 = """# Calculate initial loss (negative log probability of target class)
    initial_loss = -jnp.log(initial_probs[target_label] + 1e-10)
    # Convert to Python scalar with stop_gradient to avoid tracer issues
    initial_loss_val = float(jax.lax.stop_gradient(initial_loss))
    saved_losses.append(initial_loss_val)"""
            
            # 7. Add a fix for the f-string formatting issue
            fix7 = """# Get initial prediction
    _, initial_probs = predict_image(model, image)
    
    # Stop gradients for safe output
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
    saved_losses.append(initial_loss_val)"""
            
            # Replace the problematic areas
            fixed_content = content
            
            # 1. Progress printing fix
            fixed_content = re.sub(r"# Print progress periodically.*?curr_loss_val = float\(current_loss_value\)",
                           fix1, fixed_content, flags=re.DOTALL)
            
            # 2. Random search approach fix
            fixed_content = re.sub(r"# Convert target probabilities to Python scalars.*?current_target_prob = saved_probs\[-1\]\[target_label\].*?if hasattr\(current_target_prob, 'item'\):.*?current_target_prob = current_target_prob.item\(\).*?else:.*?current_target_prob = float\(current_target_prob\)",
                           fix2, fixed_content, flags=re.DOTALL)
            
            # 3. Helper functions fix
            fixed_content = re.sub(r"# Function to safely extract probabilities.*?def get_pred_idx\(probs\):.*?return pred_idx\.item\(\) if hasattr\(pred_idx, 'item'\) else int\(pred_idx\)",
                           fix3, fixed_content, flags=re.DOTALL)
            
            # 4. Plotting probabilities fix
            fixed_content = re.sub(r"# Plot probabilities over time - convert JAX array to NumPy.*?if hasattr\(saved_probs\[0\], 'numpy'\):.*?probs_array = np\.array\(saved_probs\)",
                           fix4, fixed_content, flags=re.DOTALL)
            
            # 5. CSV conversion fix - complete replacement
            fixed_content = re.sub(r"# Save data to CSV.*?results_data = \{.*?'loss': saved_losses.*?\}.*?# Add probabilities for each class.*?results_data\[f'prob_\{class_name\}'\] = class_probs",
                           fix5, fixed_content, flags=re.DOTALL)
            
            # 6. Initial loss fix
            fixed_content = re.sub(r"# Calculate initial loss \(negative log probability of target class\).*?saved_losses\.append\(float\(initial_loss\)\)",
                           fix6, fixed_content, flags=re.DOTALL)
            
            # 7. f-string formatting fix
            fixed_content = re.sub(r"# Get initial prediction.*?print\(f\"Initial probabilities: \{source_class\}: \{source_prob:.4f\}, \".*?f\"\{target_class\}: \{target_prob:.4f\}\"\).*?# Calculate initial loss \(negative log probability of target class\).*?saved_losses\.append\(initial_loss_val\)",
                           fix7, fixed_content, flags=re.DOTALL)
            
            # Now insert the functions in the right place
            main_section_start = re.search(r"if __name__ == \"__main__\":", fixed_content).start()
            final_content = (fixed_content[:main_section_start] + 
                           run_cf_func + "\n\n" +
                           vis_cf_func + "\n\n" +
                           fixed_content[main_section_start:main_section_start+1000] +
                           # Skip the duplicate function definitions in the remaining content
                           fixed_content[main_section_start+1000:].replace(run_cf_func, "").replace(vis_cf_func, ""))
            
            # Write the fixed content to a new file
            with open("fixed_counterfactual.py", "w") as f:
                f.write(final_content)
            
            print("Successfully fixed the file with comprehensive JAX conversion fixes!")
        else:
            print("Could not find visualize_counterfactual_steps function")
    else:
        print("Could not find run_counterfactual_analysis function")
else:
    print("Could not find main section") 