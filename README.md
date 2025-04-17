## Multi-Armed Bandit Algorithm Implementation

## Project Overview
This project implements and compares three popular multi-armed bandit algorithms for sequential decision-making under uncertainty. The multi-armed bandit problem represents a classic exploration vs. exploitation dilemma where an agent must balance trying different options to discover the best one (exploration) while also leveraging knowledge of the best-known option (exploitation).

## Problem Statement
In the multi-armed bandit problem, we face multiple slot machines (arms), each with an unknown probability distribution of rewards. Our goal is to maximize the total reward by deciding which arm to pull in each round. The challenge is that we don't know in advance which arm has the highest expected reward, so we must learn through experience.

## Algorithms Implemented

### 1. ε-Greedy Algorithm
- **Approach**: Select the best-known arm with probability 1-ε, and explore a random arm with probability ε
- **Configuration**: Tested with ε values of 0.1 and 0.01
- **Trade-off**: Higher ε values lead to more exploration, lower values focus on exploitation

### 2. Upper Confidence Bound (UCB)
- **Approach**: Select arms based on their potential upside by adding an exploration bonus to the estimated value
- **Configuration**: Tested with confidence parameters c=1.0 and c=2.0
- **Advantage**: Balances exploration and exploitation with theoretical guarantees

### 3. Thompson Sampling
- **Approach**: Model each arm's reward distribution using Bayesian statistics (Beta distributions)
- **Mechanism**: Samples from probability distributions and selects the arm with the highest sampled value
- **Advantage**: Adapts exploration based on uncertainty in a principled way

## Features
- Flexible bandit environment with configurable arms and reward distributions
- Comprehensive simulation framework for running and comparing algorithms
- Performance tracking metrics (rewards, regret, optimal arm selection rate)
- Visualization tools for analyzing algorithm behavior
- Exploration-exploitation analysis

## Installation and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-armed-bandit-algorithms.git
cd multi-armed-bandit-algorithms

# Install requirements
pip install numpy matplotlib pandas scipy

# Run the simulation
python simulation_example.py
```

## Results and Analysis
The simulation was run with 10 arms and 10,000 iterations. Key findings:

![cumulative_rewards](https://github.com/user-attachments/assets/359105df-e0b9-48de-be14-bf0f09a89fb9)

*Cumulative rewards over time for each algorithm*

![regret](https://github.com/user-attachments/assets/c63b545f-9d56-4422-be57-f7deea667859)

*Cumulative regret showing the cost of suboptimal decisions*

![optimal_arm_selection](https://github.com/user-attachments/assets/4651a180-8cb6-43a5-be0d-08c631c035d7)

*Rate at which each algorithm selects the optimal arm*

### Performance Summary
- ε-Greedy (ε=0.01) achieved a 98.89% optimal arm selection rate
- UCB (c=1.0) achieved a 98.64% optimal arm selection rate
- Thompson Sampling demonstrated strong performance with adaptive exploration
- Lower exploration rates generally led to higher cumulative rewards once the optimal arm was identified

## Usage Examples

### Creating a Bandit Environment
```python
from multi_armed_bandit import BanditEnvironment

# Create a 10-arm bandit with custom reward distributions
means = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.1, 0.3, 0.4, 0.5]
stds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
env = BanditEnvironment(10, means, stds)
```

### Running Multiple Algorithms
```python
from multi_armed_bandit import EpsilonGreedy, UCB, ThompsonSampling, BanditSimulation

# Create algorithms
algorithms = [
    EpsilonGreedy(env.n_arms, epsilon=0.1),
    UCB(env.n_arms, c=2.0),
    ThompsonSampling(env.n_arms)
]

# Create and run simulation
sim = BanditSimulation(env, algorithms, n_iterations=10000)
results = sim.run()

# Get performance summary
summary = sim.performance_summary()
print(summary)

# Plot results
figures = sim.plot_results()
```

## Technical Details
- Environment implementation using Gaussian reward distributions
- Accurate tracking of key performance indicators (KPIs)
- Dynamic visualization of exploration-exploitation balance
- Efficient simulation framework supporting multiple algorithm configurations

## Future Improvements
- Implement contextual bandit algorithms
- Add support for non-stationary environments where optimal arms change over time
- Implement policy gradient methods for comparison
- Create interactive visualizations to better understand algorithm behavior

## References
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.
- Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. Advances in neural information processing systems, 24.

## Project Assignment Context
This project was originally completed as part of a university assignment at Bharati Vidyapeeth's Engineering College from January 2022 to March 2022. The implementation demonstrates the application of reinforcement learning concepts to decision optimization problems.
