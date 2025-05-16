cd(@__DIR__)
using Pkg, OhMyREPL; Pkg.activate("."); Pkg.instantiate(); Pkg.build()

using Flux, Enzyme, Metal, LinearAlgebra
using Random, Images, ImageDraw, FileIO, Statistics, Dates
using BSON: @save, @load
using CSV, DataFrames, DataFramesMeta
using ProgressMeter, BeepBeep
using Flux: Chain, Conv, BatchNorm, MaxPool, MeanPool, Dense, Dropout, relu

#########################
#
# training weld classifier
#
#########################

label_mapping = Dict(
    "no anomaly" => 0,     # formerly NoDifetto
    "anomaly type 1" => 1, # formerly Difetto1
    "anomaly type 2" => 2, # formerly Difetto2
    "anomaly type 3" => 3  # formerly Difetto4
)

# reverse mapping required for confusion matrix labels
reverse_label_mapping = Dict(v => k for (k, v) in label_mapping)

# Mapping from actual folder names to the canonical names used in label_mapping
folder_to_canonical_name_map = Dict(
    "NoDifetto" => "no anomaly",
    "Difetto1" => "anomaly type 1",
    "Difetto2" => "anomaly type 2",
    "Difetto4" => "anomaly type 3" # Ensure "Difetto4" indeed maps to "anomaly type 3"
)
# And its inverse for convenience
canonical_name_to_folder_map = Dict(v => k for (k,v) in folder_to_canonical_name_map)


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
        Conv((3, 3), 1 => 32, pad=(1, 1)),
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
function load_and_preprocess_image(image_path, target_size=(128, 128); mean_val=nothing, std_val=nothing)
    # Load image and resize
    img = load(image_path) 
    img = imresize(img, target_size)
    
    # Convert to Float32 array, should be (H, W) for Gray
    img_float_hw = Float32.(channelview(img)) 
    
    # Reshape to (H, W, C=1)
    img_array_hwc = reshape(img_float_hw, size(img_float_hw, 1), size(img_float_hw, 2), 1)
    
    # Rearrange dimensions to be (width, height, channels) for Flux Conv layers
    img_array_whc = permutedims(img_array_hwc, (2, 1, 3)) # Now (W, H, 1)
    
    # Normalize if mean_val and std_val are provided
    if !isnothing(mean_val) && !isnothing(std_val)
        img_array_whc[:, :, 1] = (img_array_whc[:, :, 1] .- mean_val) ./ std_val
    else
        # Ensure pixels are in [0,1] if not normalizing yet (though `channelview` of N0f8 usually is)
        # This step might be redundant if images are already [0,1] Float32
        clamp!(img_array_whc, 0.0f0, 1.0f0) 
    end
    
    return img_array_whc
end

function get_all_image_paths_for_stats(base_dir, p_canonical_name_to_folder_map=canonical_name_to_folder_map, dataset_type="training", n_sampling=1)
    data_dir = joinpath(base_dir, dataset_type)
    all_paths = String[]
    
    # Iterate based on the folders present that map to canonical names
    # Use the map where keys are canonical names and values are actual folder names
    for (canonical_name, folder_name) in p_canonical_name_to_folder_map # Corrected map
        class_folder_path = joinpath(data_dir, folder_name) # folder_name is now "NoDifetto", "Difetto1", etc.
        if isdir(class_folder_path)
            img_files = filter(f -> any(endswith.(lowercase(f), [".png", ".jpg", ".jpeg"])), readdir(class_folder_path))
            sort!(img_files) # Consistent order
            sampled_files = img_files[1:n_sampling:end] # n_sampling=1 for all files
            for img_name in sampled_files
                push!(all_paths, joinpath(class_folder_path, img_name))
            end
        else
            # It's okay if a folder for a canonical class doesn't exist in, e.g., a small test split
            @warn "Sub-folder $class_folder_path for canonical class '$canonical_name' (actual folder '$folder_name') not found in $dataset_type set."
        end
    end
    if isempty(all_paths)
        @warn "No image paths found for statistics in $data_dir with n_sampling=$n_sampling. Check folder names and map."
    end
    return all_paths
end


function calculate_dataset_stats(image_paths, target_size=(128,128))
    if isempty(image_paths)
        @error "No image paths provided for statistics calculation."
        return Float32(0.5), Float32(0.5) # Default fallback
    end
    println("Calculating dataset statistics from $(length(image_paths)) images...")
    all_pixels = Float32[]
    p_bar = Progress(length(image_paths), desc="Processing images for stats: ")
    for img_path in image_paths
        # Load image without normalization for stats calculation
        img_array_whc = load_and_preprocess_image(img_path, target_size) 
        append!(all_pixels, vec(img_array_whc[:,:,1])) 
        next!(p_bar)
    end

    if isempty(all_pixels)
        @error "No pixels collected for statistics calculation. Check image loading."
        return Float32(0.5), Float32(0.5) # Default fallback
    end

    mean_stat = mean(all_pixels)
    std_stat = std(all_pixels)
    # Ensure std_stat is not zero to avoid division by zero during normalization
    if std_stat < eps(Float32)
        @warn "Calculated standard deviation is very close to zero ($std_stat). Using 1.0f0 to avoid division by zero."
        std_stat = 1.0f0
    end
    println("Calculated Mean: $mean_stat, Std: $std_stat")
    return Float32(mean_stat), Float32(std_stat)
end

"""
load images with a limit on samples per class and option to take every nth image.
Uses global label mappings for consistency and applies normalization if stats are provided.
"""
function load_dataset_with_limit(directory, samples_per_class=nothing, target_size=(128, 128), n=1; 
                                 mean_val=nothing, std_val=nothing, 
                                 # Use global mappings by default, allow override if necessary
                                 p_label_mapping=label_mapping, 
                                 p_reverse_label_mapping=reverse_label_mapping,
                                 p_canonical_name_to_folder_map=canonical_name_to_folder_map)
    images = []
    labels = []
    
    # Process classes in the order of their numeric labels (0, 1, 2, 3)
    # Sort reverse_label_mapping by key (numeric label) to ensure consistent order
    sorted_numeric_labels = sort(collect(keys(p_reverse_label_mapping)))
    
    output_class_names = [p_reverse_label_mapping[lbl_idx] for lbl_idx in sorted_numeric_labels]
    # @info "Target class order for loading: $output_class_names" # Can be verbose

    for numeric_label in sorted_numeric_labels
        canonical_class_name = p_reverse_label_mapping[numeric_label]
        
        folder_name = ""
        if haskey(p_canonical_name_to_folder_map, canonical_class_name)
            folder_name = p_canonical_name_to_folder_map[canonical_class_name]
        else
            @warn "No folder mapping found for canonical class name: '$canonical_class_name'. Skipping."
            continue
        end

        class_dir = joinpath(directory, folder_name)
        if !isdir(class_dir)
            # This warning is fine if a class is intentionally missing from a split (e.g. test)
            # @warn "Directory '$class_dir' for class '$canonical_class_name' (folder '$folder_name') not found. Skipping."
            continue
        end
        
        class_idx = numeric_label # This is the numeric label consistent with global_label_mapping
        
        img_files = filter(f -> any(endswith.(lowercase(f), [".png", ".jpg", ".jpeg"])), 
                           readdir(class_dir))
        sort!(img_files) # Sort for consistent sampling
        
        sampled_files = img_files[1:n:end]
        
        if !isnothing(samples_per_class) && length(sampled_files) > samples_per_class
            sampled_files = sampled_files[1:samples_per_class]
        end
        
        # p_desc = "Loading $canonical_class_name (folder $folder_name): " # Can be verbose
        # p_bar = Progress(length(sampled_files), desc=p_desc)
        for img_name in sampled_files
            img_path = joinpath(class_dir, img_name)
            try
                # Pass mean_val and std_val for normalization
                img_array = load_and_preprocess_image(img_path, target_size; mean_val=mean_val, std_val=std_val)
                push!(images, img_array)
                push!(labels, class_idx)
                # next!(p_bar)
            catch e
                @warn "Error loading $img_path: $e"
            end
        end
    end
    
    if !isempty(images)
        h, w, c = size(images[1])
        n_images = length(images)
        X = Array{Float32, 4}(undef, h, w, c, n_images)
        for i in 1:n_images
            X[:, :, :, i] = images[i]
        end
        y = Int.(labels)
        
        # @info "Loaded $(size(X, 4)) images from $directory (sampling every $(n)th image if n > 1)"
        return X, y, output_class_names # Return canonical class names in numeric label order
    else
        @warn "No images loaded from $directory. Check paths and mappings."
        return Array{Float32, 4}(undef, target_size[1], target_size[2], 1, 0), Int[], output_class_names 
    end
end

"""
create batches of data.
"""
function batch_generator(X, y, img_batch_size=32; shuffle_data=true)
    n_samples = size(X, 4)
    indices = collect(1:n_samples)
    
    if shuffle_data
        indices = shuffle(indices)
    end
    
    batches = []
    for i in 1:img_batch_size:n_samples
        end_idx = min(i + img_batch_size - 1, n_samples)
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
function calculate_metrics_batched(model, X, y, dataset_name="Dataset", img_batch_size=32)
    n_samples = size(X, 4)
    all_predictions = []
    
    println("Processing $(n_samples) images from $(dataset_name) in batches of $(img_batch_size)...")
    
    # Get predictions batch by batch
    @showprogress for i in 1:img_batch_size:n_samples
        end_idx = min(i + img_batch_size - 1, n_samples)
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
function train_model(base_dir=nothing; samples_per_class=nothing, num_epochs=50, img_batch_size=32, n=1,
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
      
    if use_gpu
        # check for local (🍎) GPU(s)
        if Metal.functional()
            println("Using GPU for training")
            train_X = train_X |> gpu
            val_X = val_X |> gpu
            test_X = test_X |> gpu # Metal.synchronize()

            model = model |> gpu # send model to gpu too
        else
            println("Metal GPU not available, falling back to CPU")
        end
    end

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
        batches = batch_generator(train_X, train_y, img_batch_size)
        
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
        train_metrics, _ = calculate_metrics_batched(model, train_X, train_y, "Training", img_batch_size)
        val_metrics, _ = calculate_metrics_batched(model, val_X, val_y, "Validation", img_batch_size)
        test_metrics, _ = calculate_metrics_batched(model, test_X, test_y, "Test", img_batch_size)
        
        # Combine all metrics
        all_metrics = vcat(train_metrics, val_metrics, test_metrics)
        
        # Save to CSV
        metrics_df = save_metrics_csv(all_metrics, "weld_classification_metrics.csv")
        println("Comprehensive metrics calculated and saved successfully")
    catch e
        println("Warning: Could not save full metrics due to error: $e")
        println("This is non-critical as model and basic metrics are already saved.")
    end

    beep("mario")
    
    return model, metrics
end

# Set the path to your dataset
# Set the path to your dataset
base_dir = "DB - Copy/"

if !isdir(base_dir)
    println("Warning: Dataset directory not found. Please update the path.")
    # return # Consider if you want to exit here or let it fail later
end

println("\nDataset contents:", readdir(base_dir))

# Calculate normalization statistics from the full training dataset (n_sampling=1)
println("\nCalculating normalization statistics from training data...")
# Pass the map that goes from canonical names TO folder names
training_image_paths_for_stats = get_all_image_paths_for_stats(base_dir, canonical_name_to_folder_map, "training", 1)
mean_global, std_global = calculate_dataset_stats(training_image_paths_for_stats, (128, 128))

# Load full datasets without sampling or limits, applying calculated normalization
println("\nLoading full datasets with normalization...")
# The n=10 is from your existing code for sampling images for the actual dataset
# Pass the global mappings and calculated stats
full_train_X, full_train_y, class_names = load_dataset_with_limit(
    joinpath(base_dir, "training"), 
    nothing,  # No limit per class
    (128, 128),
    1; # Take every 10th image as per your original setup
    mean_val=mean_global, std_val=std_global,
    p_label_mapping=label_mapping, 
    p_reverse_label_mapping=reverse_label_mapping,
    p_canonical_name_to_folder_map=canonical_name_to_folder_map
)

full_val_X, full_val_y, _ = load_dataset_with_limit(
    joinpath(base_dir, "validation"), 
    nothing,
    (128, 128),
    1; # Take every 10th image
    mean_val=mean_global, std_val=std_global,
    p_label_mapping=label_mapping,
    p_reverse_label_mapping=reverse_label_mapping,
    p_canonical_name_to_folder_map=canonical_name_to_folder_map
)

full_test_X, full_test_y, _ = load_dataset_with_limit(
    joinpath(base_dir, "testing"), 
    nothing,
    (128, 128),
    1; # Take every 10th image
    mean_val=mean_global, std_val=std_global,
    p_label_mapping=label_mapping,
    p_reverse_label_mapping=reverse_label_mapping,
    p_canonical_name_to_folder_map=canonical_name_to_folder_map
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
    img_batch_size=32,
    num_epochs=20,  # More epochs for better results
    use_gpu=false   # Change to true to use GPU if available
)

using BSON: @load

model_save_directory = "model_save_20250509_152731_39JA3kM4" 
model_save_directory = "model_save_fulldata_20epochs"

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

original_label_idx = label_mapping["anomaly type 3"]
target_label_idx = label_mapping["no anomaly"]     # Should be 0

# --- Find ALL images CORRECTLY classified as 'anomaly type 2' ---
println("Searching for all test images correctly classified as 'anomaly type 3'...")
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
    error("Could not find any test image correctly classified as 'anomaly type 3', despite confusion matrix indicating they exist. Check data consistency.")
elseif length(correctly_classified_indices) == 1
    global found_idx = correctly_classified_indices[1]
    println("Only one correctly classified image found. Using index: $found_idx")
else
    global found_idx = correctly_classified_indices[2] # Choose the second one
    println("Multiple correctly classified images found. Choosing the second one at index: $found_idx")
end

global found_idx = correctly_classified_indices[29]
# --- End Selection ---

original_image = full_test_X[:, :, :, found_idx:found_idx] # Keep 4 dims (W, H, C, N=1)

# --- Display Selected Original Image --- 
# Helper function to reverse normalization

function denormalize_image(img_tensor_whc, mean_val, std_val) # Expects Width x Height x Channels (W, H, 1)
    denorm_img = similar(img_tensor_whc)
    
    # Operate on the single channel
    denorm_img[:, :, 1] = (img_tensor_whc[:, :, 1] .* std_val) .+ mean_val
    
    # Clamp values to valid range [0, 1]
    clamp!(denorm_img, 0.0f0, 1.0f0)
    return denorm_img
end

println("Displaying the selected original image for inspection...")
original_image_squeezed_display = original_image[:,:,:,1]
denorm_original_for_display_whc = denormalize_image(original_image_squeezed_display, mean_global, std_global)
original_gray = colorview(Gray, permutedims(denorm_original_for_display_whc[:,:,1], (2,1)))
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

    # grads[1] now has dimensions (Width, Height, Channels=1, Batch=1)
    grayscale_gradient_update = grads[1] 
    
    # Update the image using gradient descent
    counterfactual_image .-= learning_rate .* grayscale_gradient_update

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
history_filename = "cf_full_hist_idx$(found_idx).csv"
CSV.write(history_filename, history_df)

# Prepare images for display
# Squeeze the batch dimension (last one)
original_image_squeezed = original_image[:,:,:,1]
final_counterfactual_image_squeezed = counterfactual_image[:,:,:,1]

# Denormalize using global statistics
denorm_original = denormalize_image(original_image_squeezed, mean_global, std_global)
denorm_final = denormalize_image(final_counterfactual_image_squeezed, mean_global, std_global)

# Convert to Gray images for saving (needs Height x Width for colorview(Gray, ...))
# denorm_original and denorm_final are (W, H, 1)
# We need to extract the 2D slice and permute it to (H, W)
img_original_gray = colorview(Gray, permutedims(denorm_original[:,:,1], (2, 1)))
img_final_gray = colorview(Gray, permutedims(denorm_final[:,:,1], (2, 1)))

using FileIO

original_filename = "original_full_idx$(found_idx).png"; save(original_filename, img_original_gray)
final_filename = "cf_full_image_idx$(found_idx).png"; save(final_filename, img_final_gray)

counterfactual_image[:, :, 1, 1] .- original_image[:, :, 1, 1] |>
    δ_p -> CSV.write("full_delta_pixels_$(found_idx).csv",  DataFrame(δ_p[:, :, 1], :auto))


#########################
#
# saliency maps
#
#########################

using Images: imresize # Ensure imresize is explicitly available if not already

"""
Generates a Grad-CAM heatmap for a given image, model, and target class.
Highlights regions in the image important for the target class prediction,
based on activations of a target convolutional layer.

Args:
    model: The trained Flux model.
    image_tensor_whcn: The input image tensor (Width, Height, Channels, Batch=1).
    predicted_class_idx: The 0-based index of the target class.
    target_layer_index: The index of the target convolutional layer's output 
                        (e.g., the BatchNorm after the last Conv). For your model, this is 11.
Return:
    A 2D (Width x Height) Grad-CAM heatmap, normalized to [0, 1].
"""
function generate_grad_cam(model, image_tensor_whcn, predicted_class_idx, target_layer_index::Int)
    Flux.testmode!(model)

    # 1. Define sub-models to get feature maps and gradients
    # Model up to the target layer (inclusive)
    model_upto_target = Chain(model.layers[1:target_layer_index]...)
    
    # Model from after the target layer to the end
    # Ensure that if target_layer_index is the last layer, model_after_target is an identity or handled
    if target_layer_index >= length(model.layers)
        # This case should ideally not happen for typical Grad-CAM target layers
        # If it does, model_after_target might be an identity function or just the final output processing
        model_after_target = identity
    else
        model_after_target = Chain(model.layers[target_layer_index+1:end]...)
    end

    # 2. Forward pass to get feature map activations from the target layer
    feature_maps = model_upto_target(image_tensor_whcn) # Shape: (fm_w, fm_h, fm_c, 1)

    # 3. Define a function to get the score of the target class from the feature maps
    #    This function will be differentiated w.r.t. feature_maps
    function get_class_score_from_feature_maps(fmaps)
        logits = model_after_target(fmaps)
        # predicted_class_idx is 0-based, Julia indexing is 1-based
        return logits[predicted_class_idx + 1, 1] # Score for the target class, for the single batch item
    end

    # 4. Calculate gradients of the class score w.r.t. the feature map activations
    _, grads_feature_maps_tuple = Flux.withgradient(get_class_score_from_feature_maps, feature_maps)
    grads_feature_maps = grads_feature_maps_tuple[1] # Shape: (fm_w, fm_h, fm_c, 1)

    # 5. Global Average Pooling of the gradients to get neuron importance weights (alpha_k)
    #    Pool over spatial dimensions (width and height of feature maps)
    #    grads_feature_maps shape: (fm_w, fm_h, num_channels, batch_size=1)
    #    We want weights of shape (num_channels,)
    weights = dropdims(mean(grads_feature_maps, dims=(1, 2)), dims=(1, 2, 4)) # Shape: (fm_c,)
                                                                          # dims=(1,2,4) because input is (W,H,C,N)

    # 6. Compute the weighted combination of feature maps
    #    feature_maps shape: (fm_w, fm_h, fm_c, 1)
    fm_w, fm_h, fm_c, _ = size(feature_maps)
    grad_cam_map = zeros(Float32, fm_w, fm_h)

    for k in 1:fm_c
        grad_cam_map .+= weights[k] .* feature_maps[:, :, k, 1]
    end

    # 7. Apply ReLU (often done in Grad-CAM to keep only positive influences)
    grad_cam_map = max.(grad_cam_map, 0.0f0)

    # 8. Resize the heatmap to the original image dimensions
    #    Input image_tensor_whcn has shape (W, H, C, N)
    original_w, original_h = size(image_tensor_whcn, 1), size(image_tensor_whcn, 2)
    if fm_w > 0 && fm_h > 0 # Ensure map is not empty
        # imresize expects (H, W) or (H, W, C)
        # grad_cam_map is (fm_w, fm_h), so permute for imresize, then permute back
        grad_cam_map_hw = permutedims(grad_cam_map, (2,1)) # (fm_h, fm_w)
        resized_map_hw = imresize(grad_cam_map_hw, (original_h, original_w)) # (original_h, original_w)
        resized_map_wh = permutedims(resized_map_hw, (2,1)) # (original_w, original_h)
    else
        resized_map_wh = zeros(Float32, original_w, original_h)
    end


    # 9. Normalize the heatmap to [0, 1]
    min_val, max_val = extrema(resized_map_wh)
    if max_val > min_val
        grad_cam_normalized = (resized_map_wh .- min_val) ./ (max_val - min_val)
    else
        # Handle cases where the map is all zeros or constant
        grad_cam_normalized = zeros(Float32, size(resized_map_wh))
    end
    
    return grad_cam_normalized # Returns a 2D (Width x Height) map
end

println("\nGenerating Grad-CAM Map...")

# Ensure model is in test mode (might be redundant, but safe)
Flux.testmode!(model)

# Get the original prediction for the selected image
original_logits = model(original_image) # original_image is (W, H, C, N=1)
original_pred_idx = Flux.onecold(original_logits[:, 1])[1] - 1 # Get 0-based index for the single image

println("Original Predicted Class for Grad-CAM: $(reverse_label_mapping[original_pred_idx]) (Index: $original_pred_idx)")

# --- Generate the Grad-CAM map ---
# For your model structure:
# Conv((3, 3), 128 => 256, pad=(1, 1)),  // model.layers[10]
# BatchNorm(256, relu),                 // model.layers[11] <- Target layer output
# MeanPool((2, 2), stride=(2, 2)),      // model.layers[12]
target_conv_layer_output_idx = 11 
grad_cam_map_wh = generate_grad_cam(model, original_image, original_pred_idx, target_conv_layer_output_idx)

CSV.write("grad_cam_map_full_idx$(found_idx).csv", DataFrame(grad_cam_map_wh, :auto))

"""
Generates a vanilla gradient saliency map for a given image and model.
Highlights pixels influencing the score of the predicted class.
"""
function generate_saliency_map(model, image_tensor_whcn, predicted_class_idx)
    # Ensure model is in test mode
    Flux.testmode!(model)

    # Ensure image tensor requires gradients (Flux usually handles this implicitly)
    # image_tensor_whcn = Flux.param(image_tensor_whcn) # Usually not needed for input grads

    # Calculate gradient of the predicted class score w.r.t. the input image
    loss_val, grads = Flux.withgradient(image_tensor_whcn) do img
        logits = model(img)
        # Target the score of the *predicted* class for this image
        # Logits are typically (num_classes, batch_size)
        # predicted_class_idx is 0-based, so add 1 for Julia indexing
        logits[predicted_class_idx + 1, 1]
    end

    # Get the gradient w.r.t the image (grads[1])
    saliency_grad = grads[1]

    # Process the gradient to get a single saliency value per pixel:
    # 1. Take absolute value
    saliency_abs = abs.(saliency_grad)
    # 2. Find max across color channels (dimension 3)
    # Input is (W, H, C, N=1), so reduce along dim 3
    saliency_map_wh = maximum(saliency_abs, dims=3)[:,:,1,1] # Result is (W, H)

    # Normalize the map to [0, 1] for visualization
    min_val, max_val = extrema(saliency_map_wh)
    if max_val > min_val
        saliency_map_normalized = (saliency_map_wh .- min_val) ./ (max_val - min_val)
    else
        saliency_map_normalized = zeros(Float32, size(saliency_map_wh)) # Avoid division by zero
    end

    return saliency_map_normalized # Returns a 2D (Width x Height) map
end

println("\nGenerating Saliency Map...")

# Ensure model is in test mode (might be redundant, but safe)
Flux.testmode!(model)

# Get the original prediction for the selected image
original_logits = model(original_image) # original_image is (W, H, C, N=1)
original_pred_idx = Flux.onecold(original_logits[:, 1])[1] - 1 # Get 0-based index for the single image

println("Original Predicted Class for Saliency: $(reverse_label_mapping[original_pred_idx]) (Index: $original_pred_idx)")

# Generate the saliency map
saliency_map_wh = generate_saliency_map(model, original_image, original_pred_idx)

CSV.write("saliency_map_full_idx$(found_idx).csv", DataFrame(saliency_map_wh, :auto))

