# Change working directory to the current file's directory
cd(@__DIR__)

# Set up a new environment with suitable libraries loaded
using Pkg; Pkg.activate("."); Pkg.build(); Pkg.instantiate()

# Import necessary packages
using POMDPs, POMDPModelTools, POMDPPolicies, POMDPSimulators  # Core POMDP functionality
using DiscreteValueIteration   # For solving MDPs using value iteration
using Distributions, Random, Statistics    # For probability distributions and random number generation
using Flux

using Parameters, ProgressMeter

const n_assets = 3  # You can change this to the desired number of assets

# Define the state space for the fleet management problem
struct FleetState
    statuses::NTuple{n_assets, Bool}  # Operational status of each asset (true = operational, false = failed)
    ages::NTuple{n_assets, Int}       # Ages of each asset
    time::Int                         # Current time step
end

# Define possible actions
abstract type FleetAction end

struct DoNothing <: FleetAction end           # Take no action
struct RepairAsset <: FleetAction             # Repair a specific asset
    asset_id::Int
end
struct ReplaceFleet <: FleetAction end        # Replace all assets

# Define the MDP (Markov Decision Process) for fleet management
struct FleetMDP <: MDP{FleetState, FleetAction}
    N::Int                      # Total time steps (horizon)
    operational_reward::Float64 # Reward for each operational asset per time step
    failure_cost::Float64       # Cost incurred when an asset fails
    repair_cost::Float64        # Cost to repair an asset
    replacement_cost::Float64   # Cost to replace the entire fleet
end

# Constructor for FleetMDP
function FleetMDP(; N::Int, operational_reward::Float64, failure_cost::Float64, repair_cost::Float64, replacement_cost::Float64)
    return FleetMDP(N, operational_reward, failure_cost, repair_cost, replacement_cost)
end

# Failure probability function: P(failure) increases with age
function failure_probability(age::Int)
    return min(0.1 + 0.05 * age, 1.0)
end

# Additional imports for POMDPs functionality
using POMDPs: discount, states, actions, stateindex, actionindex

# Discount factor for future rewards (0.95 means future rewards are valued 95% as much as immediate rewards)
POMDPs.discount(mdp::FleetMDP) = 0.95

# Generate all possible states for the MDP
function POMDPs.states(mdp::FleetMDP)
    # Initialize an empty array to store all possible states
    state_space = [
        # We use list comprehension to generate all possible states
        FleetState(statuses, ages, t) 
        
        # Iterate over all possible time steps from 0 to N
        # This represents the progression of time in our system
        for t in 0:mdp.N
        
        # Generate all possible combinations of asset statuses
        # Each asset can be either operational (true) or failed (false)
        # This represents the current condition of each asset in the fleet
        for statuses in Iterators.product(fill([false, true], n_assets)...)
        
        # Generate all possible age combinations for the assets
        # The age of each asset can range from 0 to the current time step
        # This ensures that an asset's age never exceeds the current time
        # and represents the aging process of the assets over time
        for ages in Iterators.product(fill(0:t, n_assets)...)
    ]
    
    # Add a special terminal state to the state space
    # This state represents the end of the simulation where:
    # - All assets have failed (statuses are all false)
    # - All assets have reached maximum age (N+1)
    # - Time has exceeded the simulation horizon (N+1)
    push!(state_space, FleetState((false, false, false), (mdp.N+1, mdp.N+1, mdp.N+1), mdp.N + 1))
    
    # Check if the state space is empty (which shouldn't happen under normal circumstances)
    # This is a safeguard against potential errors in MDP parameter setting
    if isempty(state_space)
        error("State space is empty. Check the MDP parameters, especially N.")
    end
    
    # Return the complete state space
    return state_space
end

# Define all possible actions for the MDP
POMDPs.actions(mdp::FleetMDP) = [DoNothing(), ReplaceFleet(), [RepairAsset(i) for i in 1:n_assets]...]

# Assign a unique index to each state
function POMDPs.stateindex(mdp::FleetMDP, s::FleetState)
    state_space = states(mdp)
    index = findfirst(x -> x == s, state_space)
    if isnothing(index)
        # If the state is not found, check if it's a terminal state
        if s.time > mdp.N
            # Return the last index of the state space for terminal states
            return length(state_space)
        else
            error("State $s not found in state space")
        end
    end
    return index
end

# Get the total number of states
function n_states(mdp::FleetMDP)
    return length(states(mdp))
end

# Get the total number of actions
function n_actions(mdp::FleetMDP)
    return length(actions(mdp))
end

# Define actions available in a specific state
function POMDPs.actions(mdp::FleetMDP, s::FleetState)
    acts = FleetAction[DoNothing(), ReplaceFleet()]
    for i in 1:n_assets
        if !s.statuses[i]
            push!(acts, RepairAsset(i))
        end
    end
    return acts
end

# Define the transition function: P(s' | s, a)
function POMDPs.transition(mdp::FleetMDP, s::FleetState, a::FleetAction)
    statuses = s.statuses
    ages = s.ages

    # Apply action effects
    if isa(a, RepairAsset)
        idx = a.asset_id
        statuses = Base.setindex(statuses, true, idx)  # Set repaired asset to operational
        ages = Base.setindex(ages, 0, idx)             # Reset age of repaired asset
    elseif isa(a, ReplaceFleet)
        statuses = ntuple(i -> true, n_assets)  # All assets become operational
        ages = ntuple(i -> 0, n_assets)         # All ages reset to 0
    end

    # Update ages, ensuring they don't exceed N+1
    ages = ntuple(i -> statuses[i] ? min(ages[i] + 1, mdp.N + 1) : ages[i], n_assets)

    # Calculate failure probabilities for each asset
    p_fail = ntuple(i -> failure_probability(ages[i]), n_assets)

    # Compute next state probabilities
    next_states = FleetState[]
    probs = Float64[]

    # Iterate over all possible combinations of failures
    for fail_pattern in Iterators.product(fill([false, true], n_assets)...)
        prob = 1.0
        new_statuses = ntuple(i -> statuses[i] && !fail_pattern[i], n_assets)
        for i in 1:n_assets
            if statuses[i]
                if fail_pattern[i]
                    prob *= p_fail[i]
                else
                    prob *= 1 - p_fail[i]
                end
            end
        end
        s′ = FleetState(new_statuses, ages, s.time + 1)
        push!(next_states, s′)
        push!(probs, prob)
    end

    return SparseCat(next_states, probs)  # Return a sparse categorical distribution over next states
end

# Define the reward function: R(s, a, s')
# This function calculates the reward for transitioning from state s to s' after taking action a
function POMDPs.reward(mdp::FleetMDP, s::FleetState, a::FleetAction, s′::FleetState)
    # Initialize the reward
    r = 0.0

    # Calculate operational reward
    # We sum up rewards for each operational asset in the next state (s′)
    # This incentivizes keeping assets operational
    r += sum(status ? mdp.operational_reward : 0.0 for status in s′.statuses)
    
    # Add action costs
    # Different actions incur different costs to the fleet manager
    if isa(a, RepairAsset)
        # If the action is to repair an asset, we add the repair cost (negative reward)
        # This represents the financial cost of repairing an asset
        r += mdp.repair_cost  # Negative value
    elseif isa(a, ReplaceFleet)
        # If the action is to replace the entire fleet, we add the replacement cost (negative reward)
        # This represents the significant financial investment of replacing all assets
        r += mdp.replacement_cost  # Negative value
    end
    # Note: DoNothing action incurs no direct cost
    
    # Add failure costs
    # We penalize for any assets that were operational in s but failed in s′
    for i in 1:n_assets
        if s.statuses[i] && !s′.statuses[i]
            # If an asset was operational (true) in s but is now failed (false) in s′,
            # we add the failure cost (negative reward)
            # This represents the cost of unexpected failures (e.g., lost productivity, emergency repairs)
            r += mdp.failure_cost  # Negative value
        end
    end
    
    # Return the total reward for this state-action-next_state transition
    return r
end

# Define terminal states
function POMDPs.isterminal(mdp::FleetMDP, s::FleetState)
    return s.time > mdp.N
end

# Initialize MDP with specific parameters
mdp = FleetMDP(
    N = n_assets,
    operational_reward = 10.0,
    failure_cost = -50.0,
    repair_cost = -20.0,
    replacement_cost = -100.0
)

# Assign a unique index to each action
function POMDPs.actionindex(mdp::FleetMDP, a::FleetAction)
    action_space = actions(mdp)
    index = findfirst(x -> isequal(x, a), action_space)
    if isnothing(index)
        @warn "Action $a not found in action space. Returning default index 1."
        return 1
    end
    return index
end

# Solve the MDP using value iteration
dvi_solver = ValueIterationSolver(max_iterations=100, belres=1e-3, verbose=true)
dvi_policy = solve(dvi_solver, mdp)

using BSON: @save, @load
@save "dvi_policy.bson" dvi_policy
@load "dvi_policy.bson" dvi_policy

using Plots, DataFrames

# Define action labels
action_labels = ["DoNothing", "ReplaceFleet", "RepairAsset1", "RepairAsset2", "RepairAsset3"]

# Create a function to generate a meaningful state description
function state_description(index)
    total_states = size(dvi_policy.qmat, 1)
    time_step = (index - 1) ÷ (total_states ÷ (mdp.N + 1))
    return "t=$time_step"
end

# Create a heatmap with more interpretable labels
heatmap(dvi_policy.qmat, 
    ylabel="States (Time step)", 
    xlabel="Actions", 
    color=:viridis,
    xticks=(1:5, action_labels),  # Add action labels
    yticks=(1:200:n_states(mdp), [state_description(i) for i in 1:200:n_states(mdp)]),  # Add time step labels
    clim=(minimum(dvi_policy.qmat), maximum(dvi_policy.qmat)),  # Set color limits to min and max of Q-values
    fontfamily = "/Users/ddifrancesco/Library/Fonts/Atkinson-Hyperlegible-Regular-102"
    )


# Simulate the policy if it was successfully created
initial_state = FleetState((true, true, true), (0, 0, 0), 0)
for (s, a, r, sp) in stepthrough(mdp, policy, initial_state, "s,a,r,sp", max_steps=10)
    println("State: $s")
    println("Action: $a")
    println("Reward: $r")
    println("Next State: $sp")
    println("---")
end

# Feature extraction function
# This function converts a FleetState into a vector of numerical features
# that can be used as input to our neural network
function extract_features(s::FleetState)
    return Float32[
        s.statuses...,  # Operational status of each asset (1.0 for true, 0.0 for false)
        s.ages ./ mdp.N...,  # Normalized ages of each asset
        s.time / mdp.N  # Normalized current time step
    ]
end

# DQN model
# This neural network approximates the Q-function: Q(s, a)
# Input: state features
# Output: Q-values for each action
model = Chain(
    Dense(2*n_assets + 1, 128, relu),  # Input layer: 2*n_assets + 1 features
    Dense(128, 64, relu),              # Hidden layer
    Dense(64, length(actions(mdp)))   # Output layer: Q-value for each action
)

# Experience replay buffer
# This struct represents a single experience (state transition)
struct Experience
    state::Vector{Float32}      # Current state features
    action::Int                 # Action taken (as an index)
    reward::Float32             # Reward received
    next_state::Vector{Float32} # Next state features
    done::Bool                  # Whether this transition led to a terminal state
end

buffer = Experience[]  # Buffer to store experiences
buffer_size = 10000    # Maximum number of experiences to store
batch_size = 32        # Number of experiences to sample for each training step

# Epsilon-greedy policy
# This function implements an epsilon-greedy policy for exploration
ε = 0.1  # Probability of choosing a random action
# Update the epsilon_greedy function
function epsilon_greedy(s::FleetState, model::Chain, mdp::FleetMDP, ε::Float64)
    if rand() < ε
        return rand(actions(mdp))  # Explore: choose a random action
    else
        q_values = model(extract_features(s))  # Exploit: use the model to get Q-values
        return actions(mdp)[argmax(q_values)]  # Choose the action with the highest Q-value
    end
end

# Training loop
optimizer = Flux.Optimise.Adam(0.001)  # Adam optimizer with learning rate 0.001
opt_state = Flux.setup(optimizer, model)  # Setup the optimizer state
γ = 0.99  # Discount factor for future rewards
n_episodes = 1000  # Number of episodes to train

# Learning rate scheduler
function lr_schedule(initial_lr, min_lr, decay_factor, episode)
    return max(initial_lr * decay_factor^episode, min_lr)
end

# Epsilon scheduler
function epsilon_schedule(initial_ε, min_ε, decay_factor, episode)
    return max(initial_ε * decay_factor^episode, min_ε)
end

function train_dqn(mdp, model, n_episodes, buffer_size, batch_size, γ, initial_ε, initial_lr)
    buffer = Experience[]
    
    # Initialize schedulers
    min_lr = 1e-4
    min_ε = 0.01
    lr_decay = 0.995
    ε_decay = 0.995

    # Initialize optimizer with initial learning rate
    opt = Flux.setup(Flux.Adam(initial_lr), model)
    
    losses = Float64[]

    for episode in 1:n_episodes
        # Update learning rate and epsilon for this episode
        current_lr = lr_schedule(initial_lr, min_lr, lr_decay, episode)
        current_ε = epsilon_schedule(initial_ε, min_ε, ε_decay, episode)

        # Update optimizer with new learning rate
        opt = Flux.setup(Flux.Adam(current_lr), model)

        s = rand(states(mdp))
        total_reward = 0.0
        episode_loss = 0.0
        step_count = 0
        
        while !POMDPs.isterminal(mdp, s)
            features = extract_features(s)
            
            # Use current epsilon for exploration
            a = epsilon_greedy(s, model, mdp, current_ε)
            
            sp = rand(POMDPs.transition(mdp, s, a))
            r = POMDPs.reward(mdp, s, a, sp)
            done = POMDPs.isterminal(mdp, sp)
            
            push!(buffer, Experience(features, actionindex(mdp, a), r, extract_features(sp), done))
            
            if length(buffer) > buffer_size
                popfirst!(buffer)
            end
            
            total_reward += r
            s = sp
            step_count += 1
            
            if length(buffer) >= batch_size
                batch = sample(buffer, batch_size, replace = false)
                
                # Compute target Q-values
                next_states = hcat([exp.next_state for exp in batch]...)
                next_q_values = model(next_states)
                max_next_q = maximum(next_q_values, dims=1)
                targets = [exp.reward + γ * (1 - exp.done) * max_next_q[i] for (i, exp) in enumerate(batch)]
                
                # Compute current Q-values
                states = hcat([exp.state for exp in batch]...)
                current_q = model(states)
                
                # Compute loss
                loss, grads = Flux.withgradient(model) do m
                    q_values = m(states)
                    sum((q_values[CartesianIndex.([exp.action for exp in batch], 1:batch_size)] .- targets).^2) / batch_size
                end
                
                # Update model
                Flux.update!(opt, model, grads[1])
                
                episode_loss += loss
            end
        end
        
        push!(losses, episode_loss / step_count)
        
        if episode % 100 == 0
            println("Episode $episode: Total reward = $total_reward, Avg Loss = $(losses[end]), ε = $current_ε, lr = $current_lr")
        end
    end
    
    return model, losses
end

# Usage example:
initial_lr = 0.001
initial_ε = 0.1
trained_model, training_losses = train_dqn(mdp, model, n_episodes, buffer_size, batch_size, γ, initial_ε, initial_lr)

# DQN policy
# This function defines the learned policy: it chooses the action with the highest Q-value
dqn_policy(s::FleetState) = actions(mdp)[argmax(model(extract_features(s)))]

plot(training_losses)

using POMDPPolicies

# Wrap the dqn_policy function in a FunctionPolicy object
dqn_policy_object = FunctionPolicy(dqn_policy)

# Simulate the DQN policy
initial_state = FleetState((true, true, true), (0, 0, 0), 0)
for (s, a, r, sp) in stepthrough(mdp, dqn_policy_object, initial_state, "s,a,r,sp", max_steps=10)
    println("State: $s")
    println("Action: $a")
    println("Reward: $r")
    println("Next State: $sp")
    println("---")
end

# Function to evaluate a policy
function evaluate_policy(mdp, policy, n_episodes=1000, max_steps=100)
    total_rewards = []
    for _ in 1:n_episodes
        s = rand(states(mdp))  # Start from a random state
        episode_reward = 0.0
        for _ in 1:max_steps
            a = action(policy, s)
            sp = rand(POMDPs.transition(mdp, s, a))
            r = POMDPs.reward(mdp, s, a, sp)
            episode_reward += r
            if POMDPs.isterminal(mdp, sp)
                break
            end
            s = sp
        end
        push!(total_rewards, episode_reward)
    end
    return mean(total_rewards), std(total_rewards)
end

# Evaluate the DQN policy
dqn_mean_reward, dqn_std_reward = evaluate_policy(mdp, dqn_policy_object)
println("DQN Policy - Mean reward: $dqn_mean_reward, Std dev: $dqn_std_reward")

# Evaluate the exact solution policy
exact_mean_reward, exact_std_reward = evaluate_policy(mdp, policy)
println("Exact Solution Policy - Mean reward: $exact_mean_reward, Std dev: $exact_std_reward")

# Compare policies on specific states
function compare_policies_on_states(mdp, dqn_policy, exact_policy, states_to_check)
    for s in states_to_check
        dqn_action = action(dqn_policy, s)
        exact_action = action(exact_policy, s)
        println("State: $s")
        println("  DQN action: $dqn_action")
        println("  Exact action: $exact_action")
        println("---")
    end
end

# Choose some states to compare
states_to_check = [
    FleetState((true, true, true), (0, 0, 0), 0),
    FleetState((false, true, true), (2, 1, 1), 1),
    FleetState((true, false, false), (3, 2, 2), 2),
    FleetState((false, false, false), (3, 3, 3), 3)
]

compare_policies_on_states(mdp, dqn_policy_object, policy, states_to_check)

# Calculate the percentage of matching actions
function calculate_matching_percentage(mdp, dqn_policy, exact_policy)
    matching_count = 0
    total_count = 0
    for s in states(mdp)
        dqn_action = action(dqn_policy, s)
        exact_action = action(exact_policy, s)
        if dqn_action == exact_action
            matching_count += 1
        end
        total_count += 1
    end
    return (matching_count / total_count) * 100
end

matching_percentage = calculate_matching_percentage(mdp, dqn_policy_object, policy)
println("Percentage of matching actions: $matching_percentage%")

# Saliency Map Function
function compute_saliency_map(model, state::FleetState)
    features = extract_features(state)
    
    # Compute gradients of the maximum Q-value with respect to the input features
    grads = Zygote.gradient(features) do x
        q_values = model(x)
        maximum(q_values)
    end
    
    # The saliency map is the absolute value of the gradients
    saliency_map = abs.(grads[1])
    
    return saliency_map
end

# Function to interpret saliency map
function interpret_saliency_map(saliency_map, state::FleetState)
    n_assets = length(state.statuses)
    feature_names = vcat(
        ["Asset $(i) Status" for i in 1:n_assets],
        ["Asset $(i) Age" for i in 1:n_assets],
        ["Time"]
    )
    
    # Sort features by saliency
    sorted_indices = sortperm(saliency_map, rev=true)
    
    println("Saliency Map Interpretation:")
    for (i, idx) in enumerate(sorted_indices)
        println("$i. $(feature_names[idx]): $(round(saliency_map[idx], digits=4))")
    end
end

# Counterfactual Analysis Function
# Counterfactual Analysis Function
function counterfactual_analysis(model, state::FleetState, new_action::FleetAction)
    original_features = extract_features(state)
    original_q_values = model(original_features)
    original_action = argmax(original_q_values)
    
    # Initialize the modified features
    modified_features = copy(original_features)
    
    # Record of intermediate steps
    steps = []
    
    # Iterate over features
    for i in 1:length(modified_features)
        # Try changing this feature
        for new_value in [0.0f0, 1.0f0]  # You can expand this range if needed
            temp_features = copy(modified_features)
            temp_features[i] = new_value
            
            temp_q_values = model(temp_features)
            temp_action = argmax(temp_q_values)
            
            # Record this step
            push!(steps, (i, new_value, temp_action, temp_q_values))
            
            # If this change results in the desired action, keep it
            if temp_action == actionindex(mdp, new_action)
                modified_features = temp_features
                break
            end
        end
        
        # If we've achieved the desired action, stop iterating
        if argmax(model(modified_features)) == actionindex(mdp, new_action)
            break
        end
    end
    
    final_q_values = model(modified_features)
    final_action = argmax(final_q_values)
    
    return original_action, final_action, original_q_values, final_q_values, steps, modified_features
end

# Function to interpret counterfactual analysis results
function interpret_counterfactual(state::FleetState, new_action::FleetAction, original_action, final_action, original_q_values, final_q_values, steps, modified_features)
    n_assets = length(state.statuses)
    feature_names = vcat(
        ["Asset $(i) Status" for i in 1:n_assets],
        ["Asset $(i) Age" for i in 1:n_assets],
        ["Time"]
    )
    action_names = ["DoNothing", "ReplaceFleet", "RepairAsset1", "RepairAsset2", "RepairAsset3"]
    
    println("Counterfactual Analysis Results:")
    println("Desired action: $new_action")
    println("Original action: $(action_names[original_action])")
    println("Final action: $(action_names[final_action])")
    
    println("\nOriginal Q-values:")
    for (i, q) in enumerate(original_q_values)
        println("$(action_names[i]): $(round(q, digits=4))")
    end
    
    println("\nFinal Q-values:")
    for (i, q) in enumerate(final_q_values)
        println("$(action_names[i]): $(round(q, digits=4))")
    end
    
    println("\nIntermediate steps:")
    for (i, (feature, value, action, q_values)) in enumerate(steps)
        println("Step $i:")
        println("  Changed feature: $(feature_names[feature])")
        println("  New value: $value")
        println("  Resulting action: $(action_names[action])")
        println("  Q-values: ", join(["$(action_names[j]): $(round(q, digits=4))" for (j, q) in enumerate(q_values)], ", "))
    end
    
    println("\nFinal modified features:")
    for (i, (orig, mod)) in enumerate(zip(extract_features(state), modified_features))
        if orig != mod
            println("$(feature_names[i]): $orig -> $mod")
        end
    end
end

# Example usage
example_state = FleetState((true, false, true), (2, 3, 1), 5)
new_action = RepairAsset(2)  # We want to change the action to repairing the second asset

# Perform and interpret counterfactual analysis
original_action, final_action, original_q_values, final_q_values, steps, modified_features = 
    counterfactual_analysis(trained_model, example_state, new_action)
interpret_counterfactual(example_state, new_action, original_action, final_action, original_q_values, final_q_values, steps, modified_features)

# Example usage
example_state = FleetState((true, false, true), (2, 3, 1), 5)

# Compute and interpret saliency map
saliency_map = compute_saliency_map(trained_model, example_state)
interpret_saliency_map(saliency_map, example_state)

# Function to compute saliency matrix for all states
function compute_saliency_matrix(model, mdp)
    all_states = states(mdp)
    n_features = length(extract_features(first(all_states)))
    saliency_matrix = zeros(Float32, length(all_states), n_features)
    
    for (i, state) in enumerate(all_states)
        saliency_map = compute_saliency_map(model, state)
        saliency_matrix[i, :] = saliency_map
    end
    
    return saliency_matrix
end

# Compute the saliency matrix
saliency_matrix = compute_saliency_matrix(trained_model, mdp)

# Function to analyze and visualize the saliency matrix
function analyze_saliency_matrix(saliency_matrix, mdp)
    n_features = size(saliency_matrix, 2)
    feature_names = vcat(
        ["Asset $(i) Status" for i in 1:n_assets],
        ["Asset $(i) Age" for i in 1:n_assets],
        ["Time"]
    )
    
    # Compute average saliency for each feature
    avg_saliency = mean(saliency_matrix, dims=1)[:]
    
    # Sort features by average saliency
    sorted_indices = sortperm(avg_saliency, rev=true)
    
    # Plot average saliency
    p1 = bar(feature_names[sorted_indices], avg_saliency[sorted_indices],
             title="Average Feature Saliency",
             xlabel="Features", ylabel="Average Saliency",
             rotation=45, legend=false)
    
    # Heatmap of saliency across states
    p2 = heatmap(saliency_matrix,
             xlabel="Features", ylabel="States",
             color=:viridis,
             xticks=(1:n_features, feature_names)
             ) 
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800,1000))
    
    # Print top 5 most salient features
    println("Top 5 most salient features:")
    for i in 1:5
        idx = sorted_indices[i]
        println("$i. $(feature_names[idx]): $(round(avg_saliency[idx], digits=4))")
    end
end

# Analyze and visualize the saliency matrix
analyze_saliency_matrix(saliency_matrix, mdp)

# Perform and interpret counterfactual analysis
feature_to_change = 2  # Change the status of the second asset
new_value = 1.0f0  # Set it to operational (true)
original_action, modified_action, original_q_values, modified_q_values = counterfactual_analysis(trained_model, example_state, feature_to_change, new_value)
interpret_counterfactual(example_state, feature_to_change, new_value, original_action, modified_action, original_q_values, modified_q_values)