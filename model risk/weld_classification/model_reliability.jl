using Pkg
cd(@__DIR__); Pkg.activate("."); Pkg.instantiate()

using Random, Distributions, Turing
using LinearAlgebra
using DataFrames, CSV

"""
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
"""

# Load the confusion matrix data
confusion_matrix = [72 1 4 0;
                   2 62 0 0;
                   7 0 37 1;
                   0 0 0 60]

# Define the class names
class_names = ["no anomaly", "anomaly type 1", "anomaly type 2", "anomaly type 3"]

# Define the number of classes
num_classes = length(class_names)

# Define the Turing model
@model function classification_reliability(confusion_matrix, num_classes)
    # For each true class
    θ = Vector{Vector{Float64}}(undef, num_classes)
    
    for i in 1:num_classes
        # Dirichlet prior on classification probabilities for true class i
        θ[i] ~ Dirichlet(ones(num_classes))
        
        # Likelihood: multinomial distribution of classifications
        confusion_matrix[i,:] ~ Multinomial(sum(confusion_matrix[i,:]), θ[i])
    end
    
    return θ
end

n_draws = 2_500; n_warmup = 5_000; n_chains = 4
sampler = NUTS(n_warmup, 0.65)

model_reliability = classification_reliability(confusion_matrix, num_classes) |>
    model -> sample(MersenneTwister(231123), model, sampler, MCMCThreads(), n_draws, n_chains)

# Convert MCMC samples to DataFrame and filter for relevant columns
model_reliability_samples = DataFrame(model_reliability) |>
    df -> select(df,
    ["iteration", "chain", 
     names(df, r"θ")...])

# Create a new names vector
class_names = ["no anomaly", "cracking", "porosity", "lack of penetration"]
current_names = names(model_reliability_samples)
new_names = copy(current_names)

# Replace the theta column names with descriptive probability names
for i in 1:length(class_names)
    for j in 1:length(class_names)
        old_name = "θ[$i][$j]"
        if old_name in current_names
            idx = findfirst(==(old_name), current_names)
            new_names[idx] = "Pr($(class_names[j]) | $(class_names[i]))"
        end
    end
end

# Rename the columns
rename!(model_reliability_samples, Dict(zip(current_names, new_names)))

CSV.write("model_reliability_samples.csv", model_reliability_samples)

model_reliability_samples = CSV.read("model_reliability_samples.csv", DataFrame)

# Define costs for each action
costs = Dict(
    "no_action" => 0,                        # No cost if no action taken
    "secondary_inspection" => 500,           # Cost of additional inspection for porosity
    "replacement" => 7500,                   # Unified replacement cost regardless of anomaly type
    "missed_anomaly_penalty" => 50000        # Regulatory/safety penalty for missing critical anomalies
)

# Probability that porosity exceeds volumetric limit requiring replacement
p_porosity_requires_replacement = 0.01

# Probability of imperfect replacement causing future issues
p_imperfect_replacement = 0.01

# Future cost multiplier for imperfect replacement
future_cost_multiplier = 1.5

function calculate_expected_costs_rt(
    pr_no_no, pr_crack_no, pr_porosity_no, pr_LOF_no,
    pr_no_crack, pr_crack_crack, pr_porosity_crack, pr_LOF_crack,
    pr_no_porosity, pr_crack_porosity, pr_porosity_porosity, pr_LOF_porosity,
    pr_no_LOF, pr_crack_LOF, pr_porosity_LOF, pr_LOF_LOF,
    costs)
    
    # Build the probability structure from individual arguments
    p_true_given_pred = Dict(
        "no_anomaly" => [pr_no_no, pr_crack_no, pr_porosity_no, pr_LOF_no],
        "cracking" => [pr_no_crack, pr_crack_crack, pr_porosity_crack, pr_LOF_crack],
        "porosity" => [pr_no_porosity, pr_crack_porosity, pr_porosity_porosity, pr_LOF_porosity],
        "lack_of_penetration" => [pr_no_LOF, pr_crack_LOF, pr_porosity_LOF, pr_LOF_LOF]
    )
    
    # Expected costs for each predicted class
    expected_costs = Dict()
    
    # If predicted "no anomaly"
    expected_costs["predict_no_anomaly"] = 
        p_true_given_pred["no_anomaly"][2] * costs["missed_anomaly_penalty"] +
        p_true_given_pred["no_anomaly"][3] * costs["missed_anomaly_penalty"] * p_porosity_requires_replacement +
        p_true_given_pred["no_anomaly"][4] * costs["missed_anomaly_penalty"]
    
    # If predicted "cracking"
    expected_costs["predict_cracking"] = 
        costs["replacement"] +
        (p_true_given_pred["cracking"][1] + 
         p_true_given_pred["cracking"][3] * (1 - p_porosity_requires_replacement)) * costs["replacement"] +
        p_imperfect_replacement * future_cost_multiplier * costs["replacement"]
    
    # If predicted "porosity"
    expected_costs["predict_porosity"] = 
        costs["secondary_inspection"] +
        p_porosity_requires_replacement * (
            costs["replacement"] +
            p_imperfect_replacement * future_cost_multiplier * costs["replacement"]
        ) +
        p_true_given_pred["porosity"][2] * costs["missed_anomaly_penalty"] +
        p_true_given_pred["porosity"][4] * costs["missed_anomaly_penalty"]
    
    # If predicted "lack_of_penetration"
    expected_costs["predict_penetration"] = 
        costs["replacement"] +
        (p_true_given_pred["lack_of_penetration"][1] + 
         p_true_given_pred["lack_of_penetration"][3] * (1 - p_porosity_requires_replacement)) * costs["replacement"] +
        p_imperfect_replacement * future_cost_multiplier * costs["replacement"]
    
    return expected_costs
end

using DataFramesMeta

decision_analysis = @rtransform(model_reliability_samples,
    :expected_costs = calculate_expected_costs_rt(
        :"Pr(no anomaly | no anomaly)", :"Pr(cracking | no anomaly)", 
        :"Pr(porosity | no anomaly)", :"Pr(lack of penetration | no anomaly)",
        :"Pr(no anomaly | cracking)", :"Pr(cracking | cracking)",
        :"Pr(porosity | cracking)", :"Pr(lack of penetration | cracking)",
        :"Pr(no anomaly | porosity)", :"Pr(cracking | porosity)",
        :"Pr(porosity | porosity)", :"Pr(lack of penetration | porosity)",
        :"Pr(no anomaly | lack of penetration)", :"Pr(cracking | lack of penetration)",
        :"Pr(porosity | lack of penetration)", :"Pr(lack of penetration | lack of penetration)",
        costs)
    ) |> 
    df -> @rtransform(df,
        :cost_predict_no_anomaly = :expected_costs["predict_no_anomaly"],
        :cost_predict_cracking = :expected_costs["predict_cracking"],
        :cost_predict_porosity = :expected_costs["predict_porosity"],
        :cost_predict_penetration = :expected_costs["predict_penetration"],
        :min_cost = minimum(values(:expected_costs)),
        :best_decision = first(findmin(collect(pairs(:expected_costs))))[2]
    )

