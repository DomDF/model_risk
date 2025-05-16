using Pkg, OhMyREPL; cd(@__DIR__); Pkg.activate("."); Pkg.instantiate(); Pkg.build()

using Random, Distributions, Turing, LinearAlgebra
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

model_reliability_samples

#############################################################
#
# Decision analysis
#
###############################################################

using Printf
# using CSV # Assuming you have already loaded your data, e.g.:
# model_reliability_samples = CSV.read("your_file.csv", DataFrame)

# --- Define States ---
states = [:no_anomaly, :lack_of_penetration, :cracking, :porosity]
n_states = length(states); predictions = states

C_inspect = 1_000; C_per_rg = 100; C_repair_crack = 700; C_repair_lop = 500; C_repair_por = 200; C_fail = 10_000
# --- Cost Definitions (Reflecting latest description) ---

# --- COST DEFINITIONS (Reflecting latest description) ---
# Cost of following the model's prediction
C_model = Dict(
    :no_anomaly_na => 0,
    :no_anomaly_lop => C_fail,
    :no_anomaly_crack => C_fail/2,
    :no_anomaly_por => C_fail / 10,
    :lack_of_penetration_na => C_repair_lop,
    :lack_of_penetration_lop => C_repair_lop,
    :lack_of_penetration_crack => C_repair_lop + C_repair_crack,
    :lack_of_penetration_por => C_repair_lop + C_repair_por,
    :cracking_na => C_repair_crack,
    :cracking_lop => C_repair_crack + C_repair_lop,
    :cracking_crack => C_repair_crack,
    :cracking_por => C_repair_crack + C_repair_por,
    :porosity_na => C_repair_por,
    :porosity_lop => C_repair_por + C_repair_lop,
    :porosity_crack => C_repair_por + C_repair_crack,
    :porosity_por => C_repair_por
)

# Cost of manual inspection + follow up based on true state
C_manual = Dict(
    :no_anomaly => C_per_rg,
    :lack_of_penetration => C_per_rg + C_repair_lop,
    :cracking => C_per_rg + C_repair_crack,
    :porosity => C_per_rg + C_repair_por
)

# a hybrid approach, where if significant damage (cracks or lop) is predicted by the model, it triggers a perfect manual inspection
C_hybrid = Dict(
    :no_anomaly_na => 0, 
    :no_anomaly_lop => C_fail,
    :no_anomaly_crack => C_fail/2,
    :no_anomaly_por => C_fail/10,
    :lack_of_penetration_na => C_per_rg,
    :lack_of_penetration_lop => C_per_rg + C_repair_lop,
    :lack_of_penetration_crack => C_per_rg + C_repair_crack,
    :lack_of_penetration_por => C_per_rg + C_repair_por,
    :cracking_na => C_per_rg,
    :cracking_lop => C_per_rg + C_repair_lop,
    :cracking_crack => C_per_rg + C_repair_crack,
    :cracking_por => C_per_rg + C_repair_por,
    :porosity_na => C_repair_por,
    :porosity_lop => C_repair_por + C_repair_lop,
    :porosity_crack => C_repair_por + C_repair_crack,
    :porosity_por => C_repair_por
)
# ----------------------------------------------------------

# --- Function to generate probability column names ---
function prob_col_sym(predicted_state::Symbol, true_state::Symbol)
    predicted_str = replace(string(predicted_state), '_' => ' ')
    true_str = replace(string(true_state), '_' => ' ')
    return Symbol("Pr($(predicted_str) | $(true_str))")
end

# --- Helper for cost dictionary keys ---
# Provides the abbreviation for the true state part of the key
true_state_abbrevs = Dict(
    :no_anomaly => "na",
    :lack_of_penetration => "lop",
    :cracking => "crack",
    :porosity => "por"
)

function cost_dict_key(predicted_state::Symbol, true_state_actual::Symbol)
    # For C_model and C_hybrid, the key is like :prediction_truestateabbrev
    return Symbol(string(predicted_state) * "_" * true_state_abbrevs[true_state_actual])
end


# --- Calculate Costs for the Three Strategies for each Sample k ---
comparison_costs_list = []

# Iterate through each row (posterior sample k) of your DataFrame
for row in eachrow(model_reliability_samples)
    iteration_k = row.iteration
    chain_k = row.chain

    # --- Calculate costs conditional on each possible true state s_t ---
    for true_state_s_t in states

        E_cost_Model_k_st = 0.0
        E_cost_Hybrid_k_st = 0.0

        # --- Strategy: Model-Based ---
        # Sum over all possible predictions the model could make
        for predicted_state_s_p in predictions
            # Get P_k(predicted_state_s_p | true_state_s_t)
            prob_col = prob_col_sym(predicted_state_s_p, true_state_s_t)
            if !hasproperty(row, prob_col) # Make sure to check `row` not the DataFrame here
                @error "Missing probability column: $(prob_col) for iteration $(iteration_k), chain $(chain_k). Skipping this true_state."
                E_cost_Model_k_st = NaN # Mark as invalid
                break # out of inner loop (predictions)
            end
            prob_prediction_given_true = row[prob_col]

            if isnan(prob_prediction_given_true) || !isfinite(prob_prediction_given_true) || prob_prediction_given_true < 0.0 || prob_prediction_given_true > 1.0
                @warn "Sample $(iteration_k), True State $(true_state_s_t), Predicted $(predicted_state_s_p): Invalid probability $(prob_prediction_given_true). Skipping."
                E_cost_Model_k_st = NaN
                break
            end

            # Get cost for this (prediction, true_state) pair from C_model
            model_key = cost_dict_key(predicted_state_s_p, true_state_s_t)
            if !haskey(C_model, model_key)
                @error "Missing key in C_model: $(model_key). Cannot calculate model cost."
                E_cost_Model_k_st = NaN
                break
            end
            cost_for_this_prediction = C_model[model_key]
            E_cost_Model_k_st += prob_prediction_given_true * cost_for_this_prediction
        end # End loop over predicted states for Model strategy

        # --- Strategy: Hybrid ---
        # Sum over all possible predictions the model could make for the hybrid strategy
        if !isnan(E_cost_Model_k_st) # Proceed only if probs were valid for model cost calc
            for predicted_state_s_p in predictions
                prob_col = prob_col_sym(predicted_state_s_p, true_state_s_t)
                # We already checked for prob_col existence and validity above,
                # but good practice to ensure it's safe if logic changes.
                # We assume if E_cost_Model_k_st is not NaN, then these probs were okay.
                prob_prediction_given_true = row[prob_col]

                hybrid_key = cost_dict_key(predicted_state_s_p, true_state_s_t)
                if !haskey(C_hybrid, hybrid_key)
                    @error "Missing key in C_hybrid: $(hybrid_key). Cannot calculate hybrid cost."
                    E_cost_Hybrid_k_st = NaN
                    break
                end
                cost_for_this_prediction_hybrid = C_hybrid[hybrid_key]
                E_cost_Hybrid_k_st += prob_prediction_given_true * cost_for_this_prediction_hybrid
            end # End loop over predicted states for Hybrid strategy
        else
             E_cost_Hybrid_k_st = NaN # If model cost was NaN, hybrid should also be
        end


        # --- Strategy: Manual Inspection ---
        # Cost is deterministic for this strategy, given the true state
        if !haskey(C_manual, true_state_s_t)
            @error "Missing key in C_manual: $(true_state_s_t). Cannot calculate manual cost."
            E_cost_Manual_k_st = NaN
        else
            E_cost_Manual_k_st = C_manual[true_state_s_t]
        end

        # Store results for this sample k and scenario s_t
        push!(comparison_costs_list, (
            iteration = iteration_k,
            chain = chain_k,
            true_state_scenario = true_state_s_t,
            cost_model = E_cost_Model_k_st,
            cost_hybrid = E_cost_Hybrid_k_st,
            cost_manual = E_cost_Manual_k_st
        ))
    end # End loop over true states s_t
end # End loop over samples k

# --- Convert list of NamedTuples to DataFrame for easier analysis ---
comparison_costs_df = DataFrame(comparison_costs_list)

# You can now analyze comparison_costs_df
# For example, to see the average costs for each true_state_scenario:
# using Statistics
# combined_df = combine(groupby(comparison_costs_df, :true_state_scenario),
#                       :cost_model => mean => :mean_cost_model,
#                       :cost_hybrid => mean => :mean_cost_hybrid,
#                       :cost_manual => mean => :mean_cost_manual)
# print(combined_df)

# --- Analyze Costs and Calculate EVPI ---
comparison_costs_df = DataFrame(comparison_costs_list)
CSV.write("comparison_costs.csv", comparison_costs_df)

using DataFramesMeta, UnicodePlots

comparison_costs_df |>
    df -> @rsubset(df, :true_state_scenario == "no_anomaly" |> Symbol) |>
    df -> df.lowest_cost |>
    UnicodePlots.histogram

mean_costs = combine(groupby(comparison_costs_df, :true_state_scenario),
        :cost_model => mean => :mean_cost_model,
        :cost_manual => mean => :mean_cost_manual
    )

if nrow(comparison_costs_df) > 0
    println("\n--- Analysis Results ---")

    # Calculate mean costs for the two strategies
    mean_costs = combine(groupby(comparison_costs_df, :true_state_scenario),
        :cost_model => mean => :mean_cost_model,
        :cost_manual => mean => :mean_cost_manual
    )

    println("\nMean Expected Cost per Strategy | True State Scenario:")
    show(mean_costs, allrows=true, allcols=true)

    # Calculate C_best_avg = lowest mean cost across actions (strategies)
    mean_costs.C_best_avg = min.(mean_costs.mean_cost_model, mean_costs.mean_cost_manual)

    # Calculate C_avg_min = mean(lowest cost per row)
    avg_min_costs = combine(groupby(comparison_costs_df, :true_state_scenario),
        :lowest_cost => mean => :C_avg_min
    )

    # Combine for EVPI calculation
    evpi_df = innerjoin(mean_costs, avg_min_costs, on=:true_state_scenario)

    # Calculate EVPI = C_best_avg - C_avg_min
    evpi_df.EVPI = evpi_df.C_best_avg .- evpi_df.C_avg_min

    # Determine which strategy had the lowest mean cost
    evpi_df.best_avg_strategy = ifelse.(evpi_df.mean_cost_model .< evpi_df.mean_cost_manual, "Model", "Manual")
    # Handle ties - arbitrarily picks Manual if equal
     evpi_df.best_avg_strategy[isapprox.(evpi_df.mean_cost_model, evpi_df.mean_cost_manual)] .= "Model/Manual"


    println("\nEVPI Analysis (Model Parameters) Conditioned on True State:")
    println("EVPI = [Lowest Mean Cost (Model vs Manual)] - [Mean Lowest Cost per Sample (Model vs Manual)]")
    @printf "%-25s %20s %15s %15s %15s\n" "True State Scenario" "Best Avg Strategy" "Cost(Best Avg)" "Cost(Avg Min)" "EVPI"
    println("-"^90)
    for row in eachrow(evpi_df)
        c_best = isfinite(row.C_best_avg) ? @sprintf("%.2f", row.C_best_avg) : "N/A"
        c_avg_min = isfinite(row.C_avg_min) ? @sprintf("%.2f", row.C_avg_min) : "N/A"
        evpi_val = isfinite(row.EVPI) ? @sprintf("%.2f", row.EVPI) : "N/A"
        @printf "%-25s %20s %15s %15s %15s\n" row.true_state_scenario row.best_avg_strategy c_best c_avg_min evpi_val
    end
else
    println("No valid strategy costs were calculated to perform analysis.")
end