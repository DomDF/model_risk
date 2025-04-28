using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.build()

using Flux, Enzyme, CUDA, LinearAlgebra
using Random, Images, ImageDraw, FileIO, Statistics, Dates
using BSON: @save, @load
using CSV, DataFrames, DataFramesMeta
using ProgressMeter
using Flux: Chain, Conv, BatchNorm, MaxPool, MeanPool, Dense, Dropout, relu

#########################
#
# training weld classifier
#
#########################

# label mapping with numeric and string ids
label_mapping = Dict(
    "no anomaly" => 0,     # formerly NoDifetto
    "anomaly type 1" => 1, # formerly Difetto1
    "anomaly type 2" => 2, # formerly Difetto2
    "anomaly type 3" => 3  # formerly Difetto4
)

# reverse mapping required for confusion matrix labels
reverse_label_mapping = Dict(v => k for (k, v) in label_mapping)

# model architecture
"""
a CNN for classifying weld images with:
- progressive feature extraction
- batch normalization
- pooling strategies
"""
function build_weld_classifier(num_classes::Int; dropout_rate=0.2)
    model = Chain(
        # Input data is now (width, height, channels, batch)
        
        # First convolutional block
        Conv((3, 3), 3 => 32, pad=(1, 1)),
        BatchNorm(32, relu),
        MaxPool((2, 2), stride=(2, 2)),
        
        # Second convolutional block
        Conv((3, 3), 32 => 64, pad=(1, 1)),
        BatchNorm(64, relu),
        MaxPool((2, 2), stride=(2, 2)),
        
        # Third convolutional block
        Conv((3, 3), 64 => 128, pad=(1, 1)),
        BatchNorm(128, relu),
        MaxPool((2, 2), stride=(2, 2)),
        
        # Fourth convolutional block
        Conv((3, 3), 128 => 256, pad=(1, 1)),
        BatchNorm(256, relu),
        MeanPool((2, 2), stride=(2, 2)),
        
        # Flatten the 3D tensor to 1D
        Flux.flatten,
        
        # First dense layer with dropout
        Dropout(dropout_rate),
        Dense(8 * 8 * 256 => 512, relu),
        
        # Final classification layer with dropout
        Dropout(dropout_rate),
        Dense(512 => num_classes)
    )
    
    return model
end

"""
load and preprocess an image using Images.jl
"""
function load_and_preprocess_image(image_path, target_size=(128, 128))
    # Load image and resize
    img = load(image_path)
    img = imresize(img, target_size)
    
    # Convert to normalized array
    img_array = Float32.(channelview(RGB.(img)))
    
    # Rearrange dimensions to be (width, height, channels) for Flux Conv layers
    img_array = permutedims(img_array, (3, 2, 1))
    
    # Normalize using ImageNet means and stds
    mean_vals = Float32[0.485, 0.456, 0.406]
    std_vals = Float32[0.229, 0.224, 0.225]
    
    for c in 1:3
        img_array[:, :, c] = (img_array[:, :, c] .- mean_vals[c]) ./ std_vals[c]
    end
    
    return img_array
end

"""
load images with a limit on samples per class and option to take every nth image.
"""
function load_dataset_with_limit(directory, samples_per_class=nothing, target_size=(128, 128), n=1)
    images = []
    labels = []
    
    # Get class directories
    class_dirs = filter(d -> isdir(joinpath(directory, d)), readdir(directory))
    sort!(class_dirs)
    class_to_idx = Dict(cls_name => i-1 for (i, cls_name) in enumerate(class_dirs))
    
    @info "Found classes: $class_dirs"
    
    for class_name in class_dirs
        class_dir = joinpath(directory, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all valid image files and sort them for consistent sampling
        img_files = filter(f -> any(endswith.(lowercase(f), [".png", ".jpg", ".jpeg"])), 
                           readdir(class_dir))
        sort!(img_files)
        
        # Take every nth image
        sampled_files = img_files[1:n:end]
        
        # Limit samples if specified
        if !isnothing(samples_per_class) && length(sampled_files) > samples_per_class
            sampled_files = sampled_files[1:samples_per_class]
        end
        
        # Process images with progress bar
        p = Progress(length(sampled_files), desc="Loading $(class_name) images: ")
        for img_name in sampled_files
            img_path = joinpath(class_dir, img_name)
            try
                img_array = load_and_preprocess_image(img_path, target_size)
                push!(images, img_array)
                push!(labels, class_idx)
                next!(p)
            catch e
                @warn "Error loading $img_path: $e"
            end
        end
    end
    
    # Convert to arrays
    if !isempty(images)
        # Get dimensions from the first image
        h, w, c = size(images[1])
        n_images = length(images)
        
        # Pre-allocate the 4D array
        X = Array{Float32, 4}(undef, h, w, c, n_images)
        
        # Fill the pre-allocated array
        for i in 1:n_images
            X[:, :, :, i] = images[i]
        end
        
        # Combine arrays along a new dimension for batching (width,height,channels,batch)
        # X = cat(images..., dims=4) # <-- Removed this line causing stack overflow
        y = Int.(labels)
        
        @info "Loaded $(size(X, 4)) images from $directory (sampling 1/$n)"
        return X, y, class_dirs
    else
        @error "No images loaded from $directory"
        return nothing, nothing, class_dirs
    end
end

"""
create batches of data.
"""
function batch_generator(X, y, batch_size=32; shuffle_data=true)
    n_samples = size(X, 4)
    indices = collect(1:n_samples)
    
    if shuffle_data
        indices = shuffle(indices)
    end
    
    batches = []
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_indices = indices[i:end_idx]
        push!(batches, (X[:, :, :, batch_indices], y[batch_indices]))
    end
    
    return batches
end

"""
one-hot encoding version of the labels.
"""
function one_hot_encode(labels, num_classes)
    return Flux.onehotbatch(labels, 0:num_classes-1)
end

"""
define loss function logit x-entropy
"""
function loss_function(model, x, y, num_classes)
    # Forward pass
    logits = model(x)
    # Calculate cross entropy loss with one-hot encoding
    labels_onehot = one_hot_encode(y, num_classes)
    loss = Flux.logitcrossentropy(logits, labels_onehot)
    return loss
end

"""
compute confusion matrix.
"""
function compute_confusion_matrix(true_labels, predictions, label_mapping)
    # Get all unique label indices from the mapping
    label_indices = sort(collect(Set(values(label_mapping))))
    
    # Create a mapping from actual label values to matrix indices
    label_to_idx = Dict(Int(label) => i for (i, label) in enumerate(label_indices))
    
    # Initialize confusion matrix with the correct size
    n_classes = length(label_indices)
    cm = zeros(Int, n_classes, n_classes)
    
    # Fill confusion matrix
    for (t, p) in zip(true_labels, predictions)
        t_idx = label_to_idx[t]
        p_idx = label_to_idx[p]
        cm[t_idx, p_idx] += 1
    end
    
    return cm, label_indices
end

"""
print confusion matrix and metrics
"""
function print_confusion_matrix(cm, class_names, label_indices)
    # Create class labels for display
    class_labels = [get(class_names, i, "Class $i") for i in label_indices]
    
    # Print header
    println("\nConfusion Matrix:")
    println("----------------")
    
    # Print column headers
    header = "True\\Pred |"
    for label in class_labels
        header *= " $(rpad(label, 10)) |"
    end
    println(header)
    println("-" ^ length(header))
    
    # Print rows
    for (i, true_label) in enumerate(class_labels)
        row = "$(rpad(true_label, 10)) |"
        for j in 1:length(class_labels)
            row *= " $(rpad(string(cm[i, j]), 10)) |"
        end
        println(row)
    end
    println("-" ^ length(header))
    
    # Print classification report
    println("\nPer-class Metrics:")
    println("-----------------")
    for (i, label_name) in enumerate(class_labels)
        true_pos = cm[i, i]
        false_pos = sum(cm[:, i]) - true_pos
        false_neg = sum(cm[i, :]) - true_pos
        
        precision = true_pos / (true_pos + false_pos) 
        precision = isnan(precision) ? 0.0 : precision
        
        recall = true_pos / (true_pos + false_neg)
        recall = isnan(recall) ? 0.0 : recall
        
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = isnan(f1) ? 0.0 : f1
        
        println("$label_name:")
        println("  Precision: $(round(precision, digits=4))")
        println("  Recall: $(round(recall, digits=4))")
        println("  F1-score: $(round(f1, digits=4))")
    end
    
    # Print overall accuracy
    total = sum(cm)
    correct = sum(diag(cm))
    accuracy = correct / total
    println("\nOverall Accuracy: $(round(accuracy, digits=4))")
    
    return accuracy
end

"""
calculate metrics for a dataset
"""
function calculate_metrics(model, X, y, dataset_name="Dataset")
    # Forward pass for predictions
    logits = model(X)
    predictions = Flux.onecold(logits) .- 1  # Adjust for 0-based indexing
    
    # Overall accuracy
    overall_accuracy = mean(predictions .== y)
    
    # Per-class metrics
    metrics_data = []
    
    # Calculate metrics for each class
    for (cls_idx, cls_name) in reverse_label_mapping
        # True positives, false positives, false negatives
        true_pos = sum((predictions .== cls_idx) .& (y .== cls_idx))
        false_pos = sum((predictions .== cls_idx) .& (y .!= cls_idx))
        false_neg = sum((predictions .!= cls_idx) .& (y .== cls_idx))
        
        # Class metrics
        precision = true_pos / (true_pos + false_pos)
        precision = isnan(precision) ? 0.0 : precision
        
        recall = true_pos / (true_pos + false_neg)
        recall = isnan(recall) ? 0.0 : recall
        
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = isnan(f1) ? 0.0 : f1
        
        # Class support (number of samples)
        support = sum(y .== cls_idx)
        
        # Add to metrics data
        push!(metrics_data, (
            Dataset = dataset_name,
            Class = cls_name,
            Precision = precision,
            Recall = recall,
            F1 = f1,
            Support = Int(support),
            Accuracy = overall_accuracy
        ))
    end
    
    # Add overall metrics (macro average)
    macro_precision = mean([m.Precision for m in metrics_data])
    macro_recall = mean([m.Recall for m in metrics_data])
    macro_f1 = mean([m.F1 for m in metrics_data])
    total_support = sum([m.Support for m in metrics_data])
    
    push!(metrics_data, (
        Dataset = dataset_name,
        Class = "Overall (macro avg)",
        Precision = macro_precision,
        Recall = macro_recall,
        F1 = macro_f1,
        Support = Int(total_support),
        Accuracy = overall_accuracy
    ))
    
    return metrics_data, predictions
end

"""
calculate metrics in batches to save memory
"""
function calculate_metrics_batched(model, X, y, dataset_name="Dataset", batch_size=32)
    n_samples = size(X, 4)
    all_predictions = []
    
    println("Processing $(n_samples) images from $(dataset_name) in batches of $(batch_size)...")
    
    # Get predictions batch by batch
    @showprogress for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_indices = i:end_idx
        batch_X = X[:, :, :, batch_indices]
        
        # Forward pass
        logits = model(batch_X)
        batch_predictions = Flux.onecold(logits) .- 1  # Adjust for 0-based indexing
        
        # Collect predictions
        append!(all_predictions, batch_predictions)
    end
    
    # Overall accuracy
    overall_accuracy = mean(all_predictions .== y)
    
    # Per-class metrics
    metrics_data = []
    
    # Calculate metrics for each class
    for (cls_idx, cls_name) in reverse_label_mapping
        # True positives, false positives, false negatives
        true_pos = sum((all_predictions .== cls_idx) .& (y .== cls_idx))
        false_pos = sum((all_predictions .== cls_idx) .& (y .!= cls_idx))
        false_neg = sum((all_predictions .!= cls_idx) .& (y .== cls_idx))
        
        # Class metrics
        precision = true_pos / (true_pos + false_pos)
        precision = isnan(precision) ? 0.0 : precision
        
        recall = true_pos / (true_pos + false_neg)
        recall = isnan(recall) ? 0.0 : recall
        
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = isnan(f1) ? 0.0 : f1
        
        # Class support (number of samples)
        support = sum(y .== cls_idx)
        
        # Add to metrics data
        push!(metrics_data, (
            Dataset = dataset_name,
            Class = cls_name,
            Precision = precision,
            Recall = recall,
            F1 = f1,
            Support = Int(support),
            Accuracy = overall_accuracy
        ))
    end
    
    # Add overall metrics (macro average)
    macro_precision = mean([m.Precision for m in metrics_data])
    macro_recall = mean([m.Recall for m in metrics_data])
    macro_f1 = mean([m.F1 for m in metrics_data])
    total_support = sum([m.Support for m in metrics_data])
    
    push!(metrics_data, (
        Dataset = dataset_name,
        Class = "Overall (macro avg)",
        Precision = macro_precision,
        Recall = macro_recall,
        F1 = macro_f1,
        Support = Int(total_support),
        Accuracy = overall_accuracy
    ))
    
    return metrics_data, all_predictions
end

"""
save metrics to CSV
"""
function save_metrics_csv(metrics_data, filename="model_metrics_full.csv")
    # Create DataFrame
    df = DataFrame(metrics_data)
    
    # Format floating point columns
    for col in [:Precision, :Recall, :F1, :Accuracy]
        df[!, col] = round.(df[!, col], digits=4)
    end
    
    # Save to CSV
    CSV.write(filename, df)
    println("Metrics saved to $filename")
    
    return df
end

"""
save confusion matrix to CSV
"""
function save_confusion_matrix_csv(cm, class_names, label_indices, filename="confusion_matrix_full.csv")
    class_labels = [get(class_names, i, "Class $i") for i in label_indices]
    
    # Create DataFrame
    df = DataFrame()
    df[!, :True] = class_labels
    
    for (j, pred_label) in enumerate(class_labels)
        col_name = Symbol("Pred_$pred_label")
        df[!, col_name] = cm[:, j]
    end
    
    # Save to CSV
    CSV.write(filename, df)
    println("Confusion matrix saved to $filename")
    
    return df
end

"""
save model to BSON file
"""
function save_model(model, metrics)
    # Generate a unique directory name
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    unique_id = randstring(8)
    save_dir = joinpath(pwd(), "model_save_$(timestamp)_$(unique_id)")
    
    # Create the directory
    mkpath(save_dir)
    
    # Save the model
    model_path = joinpath(save_dir, "model.bson")
    @save model_path model
    
    # Save metrics
    metrics_path = joinpath(save_dir, "metrics.bson")
    @save metrics_path metrics
    
    println("Model saved to $save_dir")
    return save_dir
end

"""
save training history to CSV
"""
function save_training_history(metrics, filename="training_history_full.csv")
    # Create arrays from the metrics
    epochs = metrics[:epoch]
    train_losses = metrics[:train_loss]
    val_losses = metrics[:val_loss]
    val_accuracies = metrics[:val_accuracy]
    
    # Create DataFrame
    df = DataFrame(
        Epoch = epochs,
        TrainLoss = round.(train_losses, digits=6),
        ValLoss = round.(val_losses, digits=6),
        ValAccuracy = round.(val_accuracies, digits=6)
    )
    
    # Save to CSV
    CSV.write(filename, df)
    println("Training history saved to $filename")
    
    return df
end

"""
train model using Enzyme gradients
"""
function train_model(base_dir=nothing; samples_per_class=nothing, num_epochs=50, batch_size=32, n=1,
                    train_data=nothing, val_data=nothing, test_data=nothing, use_gpu::Bool=false)
    # Use provided data or load it
    if !isnothing(train_data) && !isnothing(val_data) && !isnothing(test_data)
        train_X, train_y, class_names = train_data
        val_X, val_y, _ = val_data
        test_X, test_y, _ = test_data
        println("Using provided datasets")
    elseif !isnothing(base_dir)
        # Load dataset with sample limit
        train_dir = joinpath(base_dir, "training")
        val_dir = joinpath(base_dir, "validation")
        test_dir = joinpath(base_dir, "testing")
        
        if isnothing(samples_per_class)
            println("Loading all samples (taking every $(n)th image)...")
        else
            println("Loading up to $(samples_per_class) samples per class (taking every $(n)th image)...")
        end
        
        train_X, train_y, class_names = load_dataset_with_limit(
            train_dir, samples_per_class, (128, 128), n)
        val_X, val_y, _ = load_dataset_with_limit(val_dir, nothing, (128, 128), n)
        test_X, test_y, _ = load_dataset_with_limit(test_dir, nothing, (128, 128), n)
    else
        error("Either provide loaded data or a base_dir to load from")
    end
    
    println("Training with $(size(train_X, 4)) images")
    println("Validating with $(size(val_X, 4)) images")
    println("Testing with $(size(test_X, 4)) images")
    println("Classes: $(class_names)")
    
    # Initialize model
    num_classes = length(label_mapping)
    model = build_weld_classifier(num_classes, dropout_rate=0.2)
    
    # Create a Duplicated model for Enzyme
    dup_model = Enzyme.Duplicated(model)
    
    # Create optimizer (use Adam with a cosine decay)
    opt = ADAM(0.001)
    opt_state = Flux.setup(opt, model)  # Setup optimizer state for the original model
    
    println("\nTraining progress:")
    println("-----------------")
    
    # Create lists to store metrics
    metrics = Dict(
        :epoch => Int[],
        :train_loss => Float64[],
        :val_loss => Float64[],
        :val_accuracy => Float64[]
    )
    
    for epoch in 1:num_epochs
        # Training
        epoch_losses = Float64[]
        
        # Create batches for this epoch
        batches = batch_generator(train_X, train_y, batch_size)
        
        for (batch_X, batch_y) in batches
            # Define loss function
            loss_fn(m, x, y) = Flux.logitcrossentropy(m(x), one_hot_encode(y, num_classes))
            
            # Compute gradients using Enzyme via Flux's gradient
            # This follows the pattern from your example
            grads = Flux.gradient((m, x, y) -> loss_fn(m, x, y), 
                                   dup_model, batch_X, batch_y)
            
            # Calculate loss for reporting
            loss_val = loss_fn(model, batch_X, batch_y)
            push!(epoch_losses, loss_val)
            
            # Update using the optimizer state and the original model
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Calculate average loss for the epoch
        train_loss = mean(epoch_losses)
        
        # Validation
        val_logits = model(val_X)
        val_labels_onehot = one_hot_encode(val_y, num_classes)
        val_loss = Flux.logitcrossentropy(val_logits, val_labels_onehot)
        val_predictions = Flux.onecold(val_logits) .- 1  # Adjust for 0-based indexing
        val_accuracy = mean(val_predictions .== val_y)
        
        # Store metrics
        push!(metrics[:epoch], epoch)
        push!(metrics[:train_loss], train_loss)
        push!(metrics[:val_loss], val_loss)
        push!(metrics[:val_accuracy], val_accuracy)
        
        # Print progress
        if epoch % 5 == 0 || epoch == 1 || epoch == num_epochs
            println("Epoch $(epoch)/$(num_epochs): train_loss=$(round(train_loss, digits=4)), val_loss=$(round(val_loss, digits=4)), val_acc=$(round(val_accuracy, digits=4))")
        end
    end
    
    # Final evaluation on test set
    test_logits = model(test_X)
    test_labels_onehot = one_hot_encode(test_y, num_classes)
    test_loss = Flux.logitcrossentropy(test_logits, test_labels_onehot)
    test_predictions = Flux.onecold(test_logits) .- 1  # Adjust for 0-based indexing
    test_accuracy = mean(test_predictions .== test_y)
    
    println("\nFinal Test Accuracy: $(round(test_accuracy, digits=4))")
    
    # Compute and display confusion matrix
    cm, label_indices = compute_confusion_matrix(test_y, test_predictions, label_mapping)
    print_confusion_matrix(cm, reverse_label_mapping, label_indices)
    
    # Save confusion matrix to CSV
    save_confusion_matrix_csv(cm, reverse_label_mapping, label_indices, "confusion_matrix.csv")
    
    # Save training history
    save_training_history(metrics, "training_history.csv")
    
    # Save the model
    save_model(model, metrics)
    println("Model and metrics saved")
    
    # Try to calculate and save metrics
    try
        # Calculate metrics for all datasets
        train_metrics, _ = calculate_metrics_batched(model, train_X, train_y, "Training", batch_size)
        val_metrics, _ = calculate_metrics_batched(model, val_X, val_y, "Validation", batch_size)
        test_metrics, _ = calculate_metrics_batched(model, test_X, test_y, "Test", batch_size)
        
        # Combine all metrics
        all_metrics = vcat(train_metrics, val_metrics, test_metrics)
        
        # Save to CSV
        metrics_df = save_metrics_csv(all_metrics, "weld_classification_metrics.csv")
        println("Comprehensive metrics calculated and saved successfully")
    catch e
        println("Warning: Could not save full metrics due to error: $e")
        println("This is non-critical as model and basic metrics are already saved.")
    end
    
    return model, metrics
end

# Set the path to your dataset
cd("/Users/ddifrancesco/Github/model risk/weld_classification/")
base_dir = "DB - Copy/"

if !isdir(base_dir)
    println("Warning: Dataset directory not found. Please update the path.")
    return
end

println("\nDataset contents:", readdir(base_dir))

# Load full datasets without sampling or limits
println("\nLoading full datasets...")
full_train_X, full_train_y, class_names = load_dataset_with_limit(
    joinpath(base_dir, "training"), 
    nothing,  # No limit
    (128, 128),
    1 # Take every nth image
)

full_val_X, full_val_y, _ = load_dataset_with_limit(
    joinpath(base_dir, "validation"), 
    nothing,
    (128, 128),
    1
)

full_test_X, full_test_y, _ = load_dataset_with_limit(
    joinpath(base_dir, "testing"), 
    nothing,
    (128, 128),
    1 # load the full test set
)

# Display class distributions
println("\nClass distribution in datasets:")

# Training set distribution
for i in 0:length(reverse_label_mapping)-1
    count = sum(full_train_y .== i)
    percentage = count / length(full_train_y) * 100
    println("$(reverse_label_mapping[i]): $(count) training images ($(round(percentage, digits=1))%)")
end

# Validation set distribution
for i in 0:length(reverse_label_mapping)-1
    count = sum(full_val_y .== i)
    percentage = count / length(full_val_y) * 100
    println("$(reverse_label_mapping[i]): $(count) validation images ($(round(percentage, digits=1))%)")
end

# Test set distribution
for i in 0:length(reverse_label_mapping)-1
    count = sum(full_test_y .== i)
    percentage = count / length(full_test_y) * 100
    println("$(reverse_label_mapping[i]): $(count) test images ($(round(percentage, digits=1))%)")
end

# Train on full dataset
trained_model, training_metrics = train_model(
    train_data=(full_train_X, full_train_y, class_names),
    val_data=(full_val_X, full_val_y, nothing),
    test_data=(full_test_X, full_test_y, nothing),
    batch_size=32,
    num_epochs=50,  # More epochs for better results
    use_gpu=false   # Change to true to use GPU if available
)

using BSON: @load
using DataFrames # Add this if you want to convert to DataFrame later

model_save_directory = "model_save_20250421_152724_pekgzn0h" # Use your actual directory

load_results = function(path::String = model_save_directory; return_model::Bool = false)
    # Define local variables; initially they don't hold the loaded data
    local model = nothing
    local metrics = nothing

    # Construct paths
    metrics_bson_path = joinpath(path, "metrics.bson")
    model_bson_path = joinpath(path, "model.bson")

    # Load data into local variables 'metrics' and 'model' if files exist
    if isfile(metrics_bson_path)
        @load metrics_bson_path metrics # Loads into local 'metrics'
    else
        println("Warning: Metrics file not found at $metrics_bson_path")
    end

    if isfile(model_bson_path)
        @load model_bson_path model     # Loads into local 'model'
    else
        println("Warning: Model file not found at $model_bson_path")
        # If model loading is essential, you might want to handle this more robustly
        if return_model
             println("Error: Cannot return model as file was not found.")
             # Decide how to proceed, e.g., return nothing, throw error
             # return nothing, metrics # Return nothing for model
        end
    end

    # Now 'model' and 'metrics' hold the loaded data (or nothing if files weren't found)
    if return_model
        return model, metrics
    else
        return metrics
    end
end

model, metrics = load_results(return_model = true)

#########################
#
# counterfactual analysis
#
#########################


using Flux: logitcrossentropy, onehot, onecold, softmax

println("Ensuring model is in evaluation mode (for Dropout, BatchNorm)")
Flux.testmode!(model) # Important for consistent predictions/gradients

num_classes = length(label_mapping)

# --- 1. Identify Target Image ---
println("Finding a suitable test image...")
test_logits = model(full_test_X)
test_preds = Flux.onecold(test_logits) .- 1 # Get 0-based predicted labels

original_label_idx = label_mapping["anomaly type 2"]
target_label_idx = label_mapping["no anomaly"]     # Should be 0

# --- Find ALL images CORRECTLY classified as 'anomaly type 2' ---
println("Searching for all test images correctly classified as 'anomaly type 2'...")
correctly_classified_indices = Int[]
for i in 1:length(full_test_y)
    # Condition checks for both true label and predicted label
    if full_test_y[i] == original_label_idx && test_preds[i] == original_label_idx
        push!(correctly_classified_indices, i)
    end
end
println("Found $(length(correctly_classified_indices)) images correctly classified as 'anomaly type 2'.")
# --- End Search ---

# --- Select Index ---
if isempty(correctly_classified_indices)
    # If this error occurs despite the confusion matrix, it might indicate
    # the test set used here differs from the one used for the matrix,
    # or an issue with label indexing.
    error("Could not find any test image correctly classified as 'anomaly type 1', despite confusion matrix indicating they exist. Check data consistency.")
elseif length(correctly_classified_indices) == 1
    global found_idx = correctly_classified_indices[1]
    println("Only one correctly classified image found. Using index: $found_idx")
else
    global found_idx = correctly_classified_indices[2] # Choose the second one
    println("Multiple correctly classified images found. Choosing the second one at index: $found_idx")
end

global found_idx = correctly_classified_indices[11]
# --- End Selection ---

original_image = full_test_X[:, :, :, found_idx:found_idx] # Keep 4 dims (W, H, C, N=1)

# --- Display Selected Original Image --- 
println("Displaying the selected original image for inspection...")
original_image_squeezed_display = original_image[:,:,:,1]
dennorm_original_display = denormalize_image(original_image_squeezed_display)
img_original_rgb_display = colorview(RGB, permutedims(dennorm_original_display, (3, 2, 1)))
display(plot(img_original_rgb_display, title="Selected Original Image (Index: $found_idx)\nTrue Label: $(reverse_label_mapping[original_label_idx])", aspect_ratio=:equal, axis=([], false)))
# --- End Display --- 

# --- 2. Set Up Counterfactual ---
counterfactual_image = Float32.(copy(original_image)) # Mutable copy as Float32

# Target label as one-hot vector (num_classes x 1)
target_label_onehot = Flux.onehot(target_label_idx, 0:num_classes-1)
target_label_onehot = reshape(target_label_onehot, num_classes, 1) # Ensure 2D for logitcrossentropy

println("Target label set to '$(reverse_label_mapping[target_label_idx])'")

# --- 3. Define Optimization Parameters ---
num_epochs = 2_000      # Number of iterations to modify the image
learning_rate = 0.01  # Step size for changing image pixels (tune this)

println("Starting counterfactual optimization for $num_epochs epochs with learning rate $learning_rate...")

# --- 4. Optimization Loop ---
history = Dict(:loss => Float64[], :probs => Vector{Float64}[], :pred_label => Int[])
intermediate_images = [] # Store images at intervals

for epoch in 1:num_epochs
    # Calculate gradient of loss w.r.t the input image
    loss_val, grads = Flux.withgradient(counterfactual_image) do img
        logits = model(img)
        Flux.logitcrossentropy(logits, target_label_onehot)
    end

    # Update the image using gradient descent
    # grads[1] contains the gradient w.r.t. the first argument (the image)
    counterfactual_image .-= learning_rate .* grads[1]

    # Get current prediction details for the *modified* image
    current_logits = model(counterfactual_image)
    current_probs = softmax(current_logits[:, 1]) # Probabilities for this single image
    current_pred_label = Flux.onecold(current_logits)[1] - 1

    # Store history
    push!(history[:loss], loss_val)
    push!(history[:probs], current_probs)
    push!(history[:pred_label], current_pred_label)

    # Store intermediate image (optional, less frequent)
    if epoch % (num_epochs ÷ 10) == 0 || epoch == 1 || epoch == num_epochs
         push!(intermediate_images, copy(counterfactual_image[:,:,:,1])) # Store 3D slice
    end

    # Print progress
    if epoch % 100 == 0 || epoch == 1 || epoch == num_epochs
        println("epoch $(epoch)/$(num_epochs): Loss=$(round(loss_val, digits=4)), 
                probs=$(round.(history[:probs][end], digits=3)), 
                pred_label=$(reverse_label_mapping[current_pred_label])")
    end
end

# Extract base data
epochs_vec = collect(1:num_epochs) # num_epochs should match the length of history items
losses_vec = history[:loss]
pred_labels_idx_vec = history[:pred_label]
all_probs_vec = history[:probs] # Vector of Vectors

# Process probabilities into separate columns
# Assumes 4 classes, order corresponding to labels 0, 1, 2, 3
num_history_epochs = length(losses_vec)
prob_no_anomaly = zeros(Float64, num_history_epochs)
prob_type1 = zeros(Float64, num_history_epochs)
prob_type2 = zeros(Float64, num_history_epochs)
prob_type3 = zeros(Float64, num_history_epochs)

for i in 1:num_history_epochs
    prob_vector = all_probs_vec[i]
    if length(prob_vector) == num_classes # Basic check
        prob_no_anomaly[i] = prob_vector[1] # Index 1 -> Label 0 ("no anomaly")
        prob_type1[i] = prob_vector[2]      # Index 2 -> Label 1 ("anomaly type 1")
        prob_type2[i] = prob_vector[3]      # Index 3 -> Label 2 ("anomaly type 2")
        prob_type3[i] = prob_vector[4]      # Index 4 -> Label 3 ("anomaly type 3")
    else
        @warn "Probability vector at epoch $i has unexpected length: $(length(prob_vector))"
    end
end

# Get Predicted Label Names (requires reverse_label_mapping to be in scope)
pred_labels_name_vec = [reverse_label_mapping[idx] for idx in pred_labels_idx_vec]

# Create DataFrame
history_df = DataFrame(
    Epoch = epochs_vec,
    Loss = losses_vec,
    PredLabelIndex = pred_labels_idx_vec,
    PredLabelName = pred_labels_name_vec,
    Prob_NoAnomaly = prob_no_anomaly,
    Prob_AnomalyType1 = prob_type1,
    Prob_AnomalyType2 = prob_type2,
    Prob_AnomalyType3 = prob_type3
)

# Save DataFrame
history_filename = "counterfactual_history_idx$(found_idx).csv"
CSV.write(history_filename, history_df)

# Helper function to reverse normalization
function denormalize_image(img_tensor_whc) # Expects Width x Height x Channels
    mean_vals = Float32[0.485, 0.456, 0.406] # Must match training
    std_vals = Float32[0.229, 0.224, 0.225]  # Must match training
    denorm_img = similar(img_tensor_whc)
    for c in 1:3
        denorm_img[:, :, c] = (img_tensor_whc[:, :, c] .* std_vals[c]) .+ mean_vals[c]
    end
    # Clamp values to valid range [0, 1] for RGB
    clamp!(denorm_img, 0.0f0, 1.0f0)
    return denorm_img
end

# Prepare images for display
# Squeeze the batch dimension (last one)
original_image_squeezed = original_image[:,:,:,1]
final_counterfactual_image_squeezed = counterfactual_image[:,:,:,1]

# Denormalize
denorm_original = denormalize_image(original_image_squeezed)
denorm_final = denormalize_image(final_counterfactual_image_squeezed)

# Convert to RGB images for plotting (needs Channels x Height x Width)
img_original_rgb = colorview(RGB, permutedims(denorm_original, (3, 2, 1)))
img_final_rgb = colorview(RGB, permutedims(denorm_final, (3, 2, 1)))

using FileIO

original_filename = "original_image_idx$(found_idx).png"; save(original_filename, img_original_rgb)
final_filename = "counterfactual_image_idx$(found_idx).png"; save(final_filename, img_final_rgb)

history