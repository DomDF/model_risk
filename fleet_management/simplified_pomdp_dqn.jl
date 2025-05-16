# Change working directory to the current file's directory
cd(@__DIR__)

# Set up environment with necessary packages
using Pkg; Pkg.activate("."); Pkg.build(); Pkg.instantiate()

# Import core packages
# Pkg.add(["POMDPs", "POMDPModelTools", "POMDPPolicies", "POMDPSimulators"])
using POMDPs, POMDPModelTools, POMDPPolicies, POMDPSimulators
# Pkg.add(["Distributions", "Random", "Statistics", "Parameters", "ProgressMeter"])
using Distributions, Random, Statistics, Parameters, ProgressMeter
# Pkg.add(["Flux", "BSON"])
using Flux, BSON
# Pkg.add(["Plots", "DataFrames", "ProgressMeter"])
using Plots, DataFrames, ProgressMeter

# Configuration parameters
const n_assets = 5  # Number of assets
const n_asset_types = 3  # Different types of assets
const n_condition_levels = 4  # More granular condition levels

# Define asset types with different characteristics
struct AssetType
    name::String
    base_failure_rate::Float64
    repair_difficulty::Float64
    replacement_cost::Float64
    operational_value::Float64
end

# Pre-define our asset types
const ASSET_TYPES = [
    AssetType("Standard", 0.05, 1.0, 80.0, 8.0),
    AssetType("Heavy-Duty", 0.03, 1.5, 150.0, 15.0),
    AssetType("Precision", 0.08, 2.0, 120.0, 12.0)
]

# Define the true state for our fleet management POMDP
struct FleetState
    # True condition of each asset (1=perfect, n_condition_levels=critical)
    conditions::NTuple{n_assets, Int}
    
    # Types of each asset
    asset_types::NTuple{n_assets, Int}
    
    # How many hours each asset has been used
    usage_hours::NTuple{n_assets, Float64}
    
    # Time since last maintenance for each asset
    time_since_maintenance::NTuple{n_assets, Float64}
    
    # Available maintenance resources (0-1 scale)
    maintenance_capacity::Float64
    
    # Current time step
    time::Int
end

# Define the observation model (what the agent can observe)
struct FleetObservation
    # Observable condition indicators (sensor readings)
    condition_indicators::NTuple{n_assets, Float64}
    
    # Observable features that indicate possible issues
    warning_indicators::NTuple{n_assets, Bool}
    
    # Results of last inspection (if performed)
    last_inspection_results::NTuple{n_assets, Int}
    
    # Time since last inspection
    time_since_inspection::NTuple{n_assets, Float64}
    
    # Types of each asset (known)
    asset_types::NTuple{n_assets, Int}
    
    # Available maintenance capacity (known)
    maintenance_capacity::Float64
    
    # Current time step
    time::Int
end

# Define the possible actions in our POMDP
abstract type FleetAction end

struct DoNothing <: FleetAction end  # Take no action

struct InspectAsset <: FleetAction   # Inspect an asset to get better information
    asset_id::Int
end

struct RepairAsset <: FleetAction    # Repair a specific asset
    asset_id::Int
    repair_level::Int  # Different repair levels (1=minor, 3=major)
end

struct ReplaceAsset <: FleetAction   # Replace a specific asset
    asset_id::Int
    new_type::Int      # Choose the type of replacement
end

struct ReplaceFleet <: FleetAction end  # Replace all assets

# Define the POMDP model
struct FleetPOMDP <: POMDP{FleetState, FleetAction, FleetObservation}
    horizon::Int                   # Time horizon
    discount_factor::Float64       # Discount factor for future rewards
    inspection_cost::Float64       # Cost of inspecting an asset
    repair_costs::Vector{Float64}  # Cost of repairs at different levels
    capacity_growth_rate::Float64  # Rate at which maintenance capacity replenishes
end

# Non-linear degradation function based on Weibull-like deterioration
function non_linear_degradation_probability(condition::Int, asset_type::Int, 
                                           usage_hours::Float64, time_since_maintenance::Float64)
    # Base factor determined by asset type
    base_factor = ASSET_TYPES[asset_type].base_failure_rate
    
    # Weibull shape parameter (>1 means increasing failure rate with age)
    shape = 1.5
    
    # Scale parameter affected by current condition
    scale = 1000.0 / (condition^2)
    
    # Calculate Weibull probability: 1 - exp(-(t/scale)^shape)
    equivalent_age = usage_hours + 20 * time_since_maintenance
    
    # Non-linear degradation probability that increases more rapidly with usage/age
    prob = 1.0 - exp(-(equivalent_age/scale)^shape)
    
    # Adjust by base factor and ensure within [0,1]
    return min(base_factor * prob * condition / 2.0, 1.0)
end

# Function to generate observation from true state
# This is where partial observability is defined!
function generate_observation(state::FleetState)
    # Generate condition indicators (noisy readings of true condition)
    condition_indicators = ntuple(i -> begin
        true_condition = state.conditions[i]
        
        # Add more noise for assets that haven't been inspected recently
        noise_level = 0.2 + 0.1 * min(state.time_since_maintenance[i] / 50.0, 1.0)
        noise = rand(Normal(0, noise_level))
        
        # Normalize to 0-1 range and add noise
        indicator = (true_condition / n_condition_levels) + noise
        return clamp(indicator, 0.0, 1.0)
    end, n_assets)
    
    # Generate warning indicators (more likely if condition is poor)
    # These are binary indicators that become more accurate with worse condition
    warning_indicators = ntuple(i -> begin
        # Probability of correct warning increases with worse condition
        warning_accuracy = 0.5 + 0.4 * (state.conditions[i] / n_condition_levels)
        true_warning = state.conditions[i] >= n_condition_levels / 2
        
        # Random chance to get the warning right based on accuracy
        return rand() < warning_accuracy ? true_warning : !true_warning
    end, n_assets)
    
    # Default inspection results (no recent inspection)
    last_inspection_results = ntuple(i -> 0, n_assets)
    time_since_inspection = ntuple(i -> state.time_since_maintenance[i], n_assets)
    
    return FleetObservation(
        condition_indicators,
        warning_indicators,
        last_inspection_results,
        time_since_inspection,
        state.asset_types,
        state.maintenance_capacity,
        state.time
    )
end

# POMDP function implementations
function POMDPs.observation(pomdp::FleetPOMDP, s::FleetState, a::FleetAction, sp::FleetState)
    # Start with the standard observation
    obs = generate_observation(sp)
    
    # If the action was an inspection, update the observation accordingly
    if isa(a, InspectAsset)
        asset_id = a.asset_id
        
        # Create updated inspection results tuple
        new_inspection_results = ntuple(i -> 
            i == asset_id ? sp.conditions[i] : obs.last_inspection_results[i], 
            n_assets)
        
        # Create updated time since inspection tuple
        new_inspection_times = ntuple(i -> 
            i == asset_id ? 0.0 : obs.time_since_inspection[i], 
            n_assets)
        
        # Return updated observation
        return FleetObservation(
            obs.condition_indicators,
            obs.warning_indicators,
            new_inspection_results,
            new_inspection_times,
            obs.asset_types,
            obs.maintenance_capacity,
            obs.time
        )
    else
        return obs
    end
end

# Generate initial state
function POMDPs.initialstate(pomdp::FleetPOMDP)
    # Randomly assign asset types
    asset_types = ntuple(i -> rand(1:n_asset_types), n_assets)
    
    # Initialize with random conditions biased toward good condition
    conditions = ntuple(i -> rand(DiscreteNonParametric(1:n_condition_levels, 
                                 [0.4, 0.3, 0.2, 0.1])), n_assets)
    
    # Initialize with random usage hours
    usage_hours = ntuple(i -> rand() * 100.0, n_assets)
    
    # Start with random maintenance times
    time_since_maintenance = ntuple(i -> rand() * 50.0, n_assets)
    
    # Start with full maintenance capacity
    maintenance_capacity = 1.0
    
    # Start at time 0
    time = 0
    
    return FleetState(
        conditions,
        asset_types,
        usage_hours,
        time_since_maintenance,
        maintenance_capacity,
        time
    )
end

# Generate all possible actions for a given state
function POMDPs.actions(pomdp::FleetPOMDP, s::FleetState)
    # Start with the basic actions
    act_list = FleetAction[DoNothing(), ReplaceFleet()]
    
    # Add inspection actions if we have maintenance capacity
    if s.maintenance_capacity >= 0.1
        for i in 1:n_assets
            push!(act_list, InspectAsset(i))
        end
    end
    
    # Add repair actions if we have sufficient maintenance capacity
    if s.maintenance_capacity >= 0.3
        for i in 1:n_assets
            for level in 1:3  # Three repair levels
                # Only offer repair if we have enough capacity for this level
                if s.maintenance_capacity >= 0.2 * level
                    push!(act_list, RepairAsset(i, level))
                end
            end
        end
    end
    
    # Add replace actions if we have sufficient maintenance capacity
    if s.maintenance_capacity >= 0.5
        for i in 1:n_assets
            for type in 1:n_asset_types
                push!(act_list, ReplaceAsset(i, type))
            end
        end
    end
    
    return act_list
end

# Calculate reward function
function POMDPs.reward(pomdp::FleetPOMDP, s::FleetState, a::FleetAction, sp::FleetState)
    reward = 0.0
    
    # Reward for operational assets
    for i in 1:n_assets
        # Assets in better condition provide more value
        condition_factor = (n_condition_levels - s.conditions[i] + 1) / n_condition_levels
        asset_value = ASSET_TYPES[s.asset_types[i]].operational_value
        reward += condition_factor * asset_value
    end
    
    # Costs for actions
    if isa(a, InspectAsset)
        reward -= pomdp.inspection_cost
    elseif isa(a, RepairAsset)
        repair_cost = pomdp.repair_costs[a.repair_level] * 
                     ASSET_TYPES[s.asset_types[a.asset_id]].repair_difficulty
        reward -= repair_cost
    elseif isa(a, ReplaceAsset)
        replacement_cost = ASSET_TYPES[a.new_type].replacement_cost
        reward -= replacement_cost
    elseif isa(a, ReplaceFleet)
        # Cost to replace all assets
        for i in 1:n_assets
            reward -= ASSET_TYPES[s.asset_types[i]].replacement_cost * 0.8  # Bulk discount
        end
    end
    
    # Penalty for failures (condition worsening to critical)
    for i in 1:n_assets
        if s.conditions[i] < n_condition_levels && sp.conditions[i] == n_condition_levels
            # Failure occurred - apply penalty
            failure_cost = ASSET_TYPES[s.asset_types[i]].operational_value * 5
            reward -= failure_cost
        end
    end
    
    return reward
end

# Transition function with non-linear degradation
function POMDPs.transition(pomdp::FleetPOMDP, s::FleetState, a::FleetAction)
    # Process action effects first
    updated_conditions = s.conditions
    updated_time_since_maintenance = s.time_since_maintenance
    updated_asset_types = s.asset_types
    
    if isa(a, RepairAsset)
        # Apply repair effects
        asset_id = a.asset_id
        repair_level = a.repair_level
        
        # Calculate repair effectiveness based on level and asset type
        effectiveness = repair_level / 3.0  # Scale by repair level
        repaired_condition = max(1, s.conditions[asset_id] - 
                                Int(floor(effectiveness * (n_condition_levels - 1))))
        
        # Update condition for the repaired asset
        updated_conditions = ntuple(i -> 
            i == asset_id ? repaired_condition : s.conditions[i], 
            n_assets)
        
        # Reset maintenance time for the repaired asset
        updated_time_since_maintenance = ntuple(i -> 
            i == asset_id ? 0.0 : s.time_since_maintenance[i], 
            n_assets)
    elseif isa(a, ReplaceAsset)
        # Replace a single asset
        asset_id = a.asset_id
        new_type = a.new_type
        
        # Update conditions for the replaced asset (set to perfect)
        updated_conditions = ntuple(i -> 
            i == asset_id ? 1 : s.conditions[i], 
            n_assets)
        
        # Update asset type
        updated_asset_types = ntuple(i -> 
            i == asset_id ? new_type : s.asset_types[i], 
            n_assets)
        
        # Reset maintenance time
        updated_time_since_maintenance = ntuple(i -> 
            i == asset_id ? 0.0 : s.time_since_maintenance[i], 
            n_assets)
    elseif isa(a, ReplaceFleet)
        # Replace all assets (set all to perfect condition)
        updated_conditions = ntuple(i -> 1, n_assets)
        
        # Keep same asset types (unless we want to randomize here)
        updated_asset_types = s.asset_types
        
        # Reset all maintenance times
        updated_time_since_maintenance = ntuple(i -> 0.0, n_assets)
    end
    
    # Now process natural state transitions
    
    # Update maintenance capacity
    next_capacity = min(1.0, s.maintenance_capacity + pomdp.capacity_growth_rate)
    
    # If an action used maintenance capacity, reduce it
    if isa(a, InspectAsset)
        next_capacity -= 0.1
    elseif isa(a, RepairAsset)
        next_capacity -= 0.2 * a.repair_level
    elseif isa(a, ReplaceAsset)
        next_capacity -= 0.5
    elseif isa(a, ReplaceFleet)
        next_capacity = 0.0  # Depletes all maintenance capacity
    end
    
    next_capacity = max(0.0, next_capacity)  # Ensure non-negative
    
    # Update usage hours (different assets might be used more/less)
    usage_increment = ntuple(i -> rand(10.0:30.0), n_assets)
    next_usage_hours = ntuple(i -> s.usage_hours[i] + usage_increment[i], n_assets)
    
    # Update time since maintenance
    next_time_since_maintenance = ntuple(i -> 
        updated_time_since_maintenance[i] + 1.0, n_assets)
    
    # Apply non-linear degradation based on usage and time
    final_conditions = ntuple(i -> begin
        # Calculate degradation probability
        deg_prob = non_linear_degradation_probability(
            updated_conditions[i], 
            updated_asset_types[i],
            usage_increment[i],  # Use the increment rather than total
            updated_time_since_maintenance[i]
        )
        
        # Apply probabilistic degradation
        if rand() < deg_prob
            # Degradation amount can vary
            degradation = rand(1:2)  # Can degrade by 1 or 2 levels
            return min(updated_conditions[i] + degradation, n_condition_levels)
        else
            return updated_conditions[i]
        end
    end, n_assets)
    
    # Increment time
    next_time = s.time + 1
    
    # Create and return the next state
    next_state = FleetState(
        final_conditions,
        updated_asset_types,
        next_usage_hours,
        next_time_since_maintenance,
        next_capacity,
        next_time
    )
    
    return Deterministic(next_state)  # Return as a deterministic distribution
end

# Check if a state is terminal
function POMDPs.isterminal(pomdp::FleetPOMDP, s::FleetState)
    return s.time >= pomdp.horizon
end

# Discount factor
POMDPs.discount(pomdp::FleetPOMDP) = pomdp.discount_factor

# Initialize our POMDP
function create_fleet_pomdp()
    return FleetPOMDP(
        100,                     # horizon
        0.95,                    # discount_factor
        5.0,                     # inspection_cost
        [10.0, 25.0, 50.0],      # repair_costs for different levels
        0.1                      # capacity_growth_rate
    )
end

# Deep Q-Network Agent for POMDP
struct DQNAgent
    qnetwork::Chain               # Q-network
    replay_buffer::Vector{Any}    # Experience replay buffer
    batch_size::Int               # Batch size for training
    γ::Float64                    # Discount factor
    ε_start::Float64              # Starting exploration rate
    ε_end::Float64                # Final exploration rate
    ε_decay::Float64              # Decay rate for exploration
    learning_rate::Float64        # Learning rate
    buffer_size::Int              # Maximum buffer size
    optimizer                     # Optimizer
    current_step::Int             # Current training step
    action_space::Vector{FleetAction}  # All possible actions
end

# Feature extraction function
function extract_features(obs::FleetObservation)
    # Extract and normalize features
    features = Float32[]
    
    # Condition indicators (already normalized)
    append!(features, collect(obs.condition_indicators))
    
    # Warning indicators (convert to Float32)
    append!(features, Float32.(collect(obs.warning_indicators)))
    
    # Last inspection results (normalize)
    append!(features, collect(obs.last_inspection_results) ./ n_condition_levels)
    
    # Time since inspection (normalize)
    max_time = 100.0  # Maximum expected time
    append!(features, collect(obs.time_since_inspection) ./ max_time)
    
    # Asset types (normalize)
    append!(features, collect(obs.asset_types) ./ n_asset_types)
    
    # Maintenance capacity
    push!(features, Float32(obs.maintenance_capacity))
    
    # Time (normalize)
    push!(features, obs.time / 100.0)  # Assuming max time is 100
    
    return features
end

# Create DQN agent
function create_dqn_agent(pomdp::FleetPOMDP)
    # Get full action space
    s0 = initialstate(pomdp)
    action_space = vcat([DoNothing(), ReplaceFleet()], 
                      [InspectAsset(i) for i in 1:n_assets],
                      [RepairAsset(i, l) for i in 1:n_assets for l in 1:3],
                      [ReplaceAsset(i, t) for i in 1:n_assets for t in 1:n_asset_types])
    
    # Count number of possible actions
    n_actions = length(action_space)
    
    # Define feature size
    feature_size = 4 * n_assets + 2  # Number of features extracted from observation
    
    # Create Q-network with larger hidden layers to compensate for lack of recurrence
    qnetwork = Chain(
        Dense(feature_size, 256, relu),    # Larger first layer
        Dense(256, 256, relu),             # Additional hidden layer
        Dense(256, 128, relu),             # Another hidden layer
        Dense(128, n_actions)              # Output layer (Q-values)
    )
    
    # Use Adam optimizer
    learning_rate = 0.0005
    optimizer = Flux.setup(Flux.Adam(learning_rate), qnetwork)
    
    # Initialize agent
    return DQNAgent(
        qnetwork,              # Q-network
        [],                    # Empty replay buffer
        32,                    # Batch size
        0.99,                  # Discount factor
        1.0,                   # Start with 100% exploration
        0.01,                  # End with 1% exploration
        0.995,                 # Decay rate
        learning_rate,         # Learning rate
        10000,                 # Buffer size
        optimizer,             # Optimizer
        0,                     # Current step
        action_space           # Action space
    )
end

# Action selection function with epsilon-greedy exploration
function select_action(agent::DQNAgent, obs::FleetObservation, evaluate::Bool=false)
    # Extract features
    features = extract_features(obs)
    
    # Calculate current epsilon for exploration
    if evaluate
        ε = 0.0  # No exploration during evaluation
    else
        # Decay epsilon over time
        decay = agent.ε_decay^agent.current_step
        ε = agent.ε_end + (agent.ε_start - agent.ε_end) * decay
    end
    
    # With probability ε, select random action
    if rand() < ε
        return rand(agent.action_space)
    end
    
    # Otherwise, select action based on Q-values
    q_values = agent.qnetwork(features)
    
    # Select action with highest Q-value
    action_idx = argmax(q_values)
    
    return agent.action_space[action_idx]
end

# Store experience in replay buffer
function store_experience(agent::DQNAgent, obs, action_idx, reward, next_obs, done)
    # Create experience tuple
    experience = (obs, action_idx, reward, next_obs, done)
    
    # Add to buffer
    push!(agent.replay_buffer, experience)
    
    # Trim buffer if too large
    if length(agent.replay_buffer) > agent.buffer_size
        popfirst!(agent.replay_buffer)
    end
end

# Training function for DQN agent
function train(agent::DQNAgent)
    # Need enough samples in buffer
    if length(agent.replay_buffer) < agent.batch_size
        return 0.0
    end
    
    # Sample batch of experiences
    batch = sample(agent.replay_buffer, agent.batch_size)
    
    # Unpack batch elements
    states = hcat([exp[1] for exp in batch]...)
    actions = [exp[2] for exp in batch]
    rewards = [exp[3] for exp in batch]
    next_states = hcat([exp[4] for exp in batch]...)
    dones = [exp[5] for exp in batch]
    
    # Calculate loss and gradient
    loss, grads = Flux.withgradient(agent.qnetwork) do q
        # Current Q-values for the taken actions
        current_q = q(states)
        q_values = current_q[CartesianIndex.([actions],[1:agent.batch_size]...)]
        
        # Target Q-values
        next_q = q(next_states)
        max_next_q = [dones[i] ? 0.0f0 : maximum(next_q[:, i]) for i in 1:agent.batch_size]
        targets = Float32.(rewards + agent.γ .* max_next_q)
        
        # Calculate mean squared error
        return mean((q_values - targets).^2)
    end
    
    # Update weights
    Flux.update!(agent.optimizer, agent.qnetwork, grads[1])
    
    return loss
end

# Full training loop
function train_dqn(pomdp::FleetPOMDP, agent::DQNAgent, n_episodes::Int)
    # Initialize metrics
    rewards_history = []
    loss_history = []
    
    # Progress meter
    progress = Progress(n_episodes, dt=1.0, desc="Training DQN... ", barglyphs=BarGlyphs("[=> ]"))
    
    for episode in 1:n_episodes
        # Reset environment
        s = initialstate(pomdp)
        o = generate_observation(s)
        
        # Initialize episode metrics
        total_reward = 0.0
        episode_losses = []
        
        # Run episode
        step = 0
        done = false
        
        while !done && step < 100  # Cap at 100 steps
            # Extract features from observation
            features = extract_features(o)
            
            # Select action
            a = select_action(agent, o, false)  # Not in evaluation mode
            
            # Convert action to index
            action_idx = findfirst(x -> x == a, agent.action_space)
            
            # Take action
            sp = rand(transition(pomdp, s, a))
            r = reward(pomdp, s, a, sp)
            op = rand(observation(pomdp, s, a, sp))
            done = isterminal(pomdp, sp)
            
            # Store experience
            store_experience(agent, features, action_idx, r, extract_features(op), done)
            
            # Update agent
            agent.current_step += 1
            loss = train(agent)
            if !isnothing(loss)
                push!(episode_losses, loss)
            end
            
            # Update state and observation
            s = sp
            o = op
            
            # Update reward
            total_reward += r
            
            step += 1
        end
        
        # Record metrics
        push!(rewards_history, total_reward)
        if !isempty(episode_losses)
            push!(loss_history, mean(episode_losses))
        else
            push!(loss_history, 0.0)
        end
        
        # Update progress
        avg_reward = mean(rewards_history[max(1, episode-99):episode])
        next!(progress, showvalues=[
            (:episode, episode),
            (:reward, total_reward),
            (:avg_reward, avg_reward),
            (:loss, isempty(episode_losses) ? 0.0 : mean(episode_losses))
        ])
    end
    
    return rewards_history, loss_history
end

# Evaluation function
function evaluate_dqn(pomdp::FleetPOMDP, agent::DQNAgent, n_episodes::Int=20)
    rewards = []
    
    for _ in 1:n_episodes
        # Reset environment
        s = initialstate(pomdp)
        o = generate_observation(s)
        
        # Initialize episode metrics
        total_reward = 0.0
        
        # Run episode
        step = 0
        done = false
        
        while !done && step < 100  # Cap at 100 steps
            # Select action - evaluation mode (no exploration)
            a = select_action(agent, o, true)
            
            # Take action
            sp = rand(transition(pomdp, s, a))
            r = reward(pomdp, s, a, sp)
            op = rand(observation(pomdp, s, a, sp))
            done = isterminal(pomdp, sp)
            
            # Update state and observation
            s = sp
            o = op
            
            # Update reward
            total_reward += r
            
            step += 1
        end
        
        push!(rewards, total_reward)
    end
    
    return mean(rewards), std(rewards)
end

# Visualization and analysis functions
function visualize_training(rewards, losses)
    p1 = plot(rewards, label="Reward", xlabel="Episode", ylabel="Total Reward",
             title="Training Progress")
    
    p2 = plot(losses, label="Loss", xlabel="Episode", ylabel="Loss",
             title="Training Loss")
    
    plot(p1, p2, layout=(2,1), size=(800,600))
end

function analyze_policy(pomdp::FleetPOMDP, agent::DQNAgent)
    # Create a set of test states
    test_states = [
        # Perfect condition fleet
        FleetState(ntuple(i -> 1, n_assets), ntuple(i -> rand(1:n_asset_types), n_assets),
                 ntuple(i -> 0.0, n_assets), ntuple(i -> 0.0, n_assets),
                 1.0, 0),
        
        # Mixed condition fleet
        FleetState(ntuple(i -> rand(1:n_condition_levels), n_assets), ntuple(i -> rand(1:n_asset_types), n_assets),
                 ntuple(i -> rand() * 100.0, n_assets), ntuple(i -> rand() * 50.0, n_assets),
                 0.8, 10),
        
        # Poor condition fleet
        FleetState(ntuple(i -> rand(3:n_condition_levels), n_assets), ntuple(i -> rand(1:n_asset_types), n_assets),
                 ntuple(i -> rand() * 200.0, n_assets), ntuple(i -> rand() * 100.0, n_assets),
                 0.5, 30),
        
        # Very poor condition fleet
        FleetState(ntuple(i -> n_condition_levels, n_assets), ntuple(i -> rand(1:n_asset_types), n_assets),
                 ntuple(i -> rand() * 300.0, n_assets), ntuple(i -> rand() * 150.0, n_assets),
                 1.0, 40)
    ]
    
    # Analyze policy for each test state
    for (i, s) in enumerate(test_states)
        o = generate_observation(s)
        
        # Select action with trained policy
        a = select_action(agent, o, true)
        
        println("Test State $i:")
        println("  Conditions: $(s.conditions)")
        println("  Asset Types: $(s.asset_types)")
        println("  Maintenance Capacity: $(s.maintenance_capacity)")
        println("  Selected Action: $a")
        println("  True vs. Observed Conditions:")
        for j in 1:n_assets
            println("    Asset $j: True=$(s.conditions[j]), Observed=$(round(o.condition_indicators[j]*n_condition_levels; digits=1)), Warning=$(o.warning_indicators[j])")
        end
        println()
    end
end

# Main execution
function main()
    # Create POMDP
    pomdp = create_fleet_pomdp()
    
    # Create DQN agent
    agent = create_dqn_agent(pomdp)
    
    # Train agent
    println("Starting training...")
    rewards, losses = train_dqn(pomdp, agent, 1000)
    
    # Evaluate agent
    mean_reward, std_reward = evaluate_dqn(pomdp, agent, 50)
    println("Evaluation - Mean reward: $mean_reward, Std dev: $std_reward")
    
    # Visualize training progress
    visualize_training(rewards, losses)
    
    # Analyze policy
    analyze_policy(pomdp, agent)
    
    # Save model
    model_state = Flux.state(agent.qnetwork)
    @save "fleet_dqn_model.bson" model_state
    
    println("Training complete! Model saved.")
end

# Run main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 