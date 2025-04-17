import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import BanditEnvironment, EpsilonGreedy, UCB, ThompsonSampling, BanditSimulation

def run_specific_experiment():
    """
    Run a specific experiment to demonstrate the performance of multi-armed bandit algorithms
    on a 10-arm problem with 10,000 iterations.
    """
    # Set seed for reproducibility
    np.random.seed(12345)
    
    # Create environment with 10 arms
    # Define arm mean rewards explicitly for reproducibility
    means = np.array([-0.25, 0.5, 1.2, -0.8, 0.3, 0.8, -0.5, 0.6, -0.1, -0.3])
    stds = np.ones(10) * 0.5
    env = BanditEnvironment(10, means, stds)
    
    print(f"Environment created with {env.n_arms} arms")
    print(f"True mean rewards: {env.means}")
    print(f"Optimal arm: {env.best_arm} (expected reward: {env.optimal_reward:.4f})")
    
    # Create algorithms with different parameters
    algorithms = [
        EpsilonGreedy(env.n_arms, epsilon=0.1),
        EpsilonGreedy(env.n_arms, epsilon=0.01),
        UCB(env.n_arms, c=2.0),
        UCB(env.n_arms, c=1.0),
        ThompsonSampling(env.n_arms)
    ]
    
    # Create simulation
    n_iterations = 10000
    print(f"\nRunning simulation for {n_iterations} iterations...")
    
    sim = BanditSimulation(env, algorithms, n_iterations)
    results = sim.run(verbose=True)
    
    # Get performance summary
    summary_df = sim.performance_summary()
    print("\nPerformance Summary:")
    print(summary_df)
    
    # Save results to output directory
    output_dir = "simulation_results"
    
    # Plot results and save figures
    metrics = ['cumulative_rewards', 'regret', 'optimal_arm_selection', 'arm_counts']
    figures = sim.plot_results(metrics, save_dir=output_dir)
    
    # Create a comparison of exploration-exploitation balance
    exploration_analysis(results, env, n_iterations, output_dir)
    
    return sim, results, summary_df, figures

def exploration_analysis(results, env, n_iterations, output_dir):
    """
    Analyze the exploration-exploitation balance for each algorithm.
    
    Parameters:
    -----------
    results : dict
        Results from the simulation
    env : BanditEnvironment
        The bandit environment
    n_iterations : int
        Number of iterations
    output_dir : str
        Directory to save the plots
    """
    # Calculate exploration rate over time (% of non-optimal arm pulls)
    plt.figure(figsize=(12, 8))
    
    window_size = min(1000, n_iterations // 10)
    
    for algo_name, result in results.items():
        # 1 for exploration (non-optimal arm), 0 for exploitation (optimal arm)
        exploration = [1 if a != env.best_arm else 0 for a in result['actions']]
        # Calculate moving average of exploration rate
        cumsum = np.cumsum(exploration)
        moving_avg = [cumsum[i] / (i + 1) for i in range(len(cumsum))]
        plt.plot(moving_avg, label=algo_name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Exploration Rate')
    plt.title('Exploration Rate Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(f"{output_dir}/exploration_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table of exploration vs exploitation
    exploration_summary = []
    
    for algo_name, result in results.items():
        # Count explorations (non-optimal arm pulls)
        explorations = sum(1 for a in result['actions'] if a != env.best_arm)
        exploration_rate = explorations / n_iterations
        
        exploration_summary.append({
            'Algorithm': algo_name,
            'Explorations': explorations,
            'Exploitations': n_iterations - explorations,
            'Exploration Rate': exploration_rate,
            'Exploitation Rate': 1 - exploration_rate
        })
    
    # Convert to DataFrame and print
    import pandas as pd
    df = pd.DataFrame(exploration_summary)
    print("\nExploration-Exploitation Summary:")
    print(df)
    
    # Save to CSV
    df.to_csv(f"{output_dir}/exploration_exploitation_summary.csv", index=False)

if __name__ == "__main__":
    # Run the specific experiment
    sim, results, summary_df, figures = run_specific_experiment()
    
    # Display the figures (in interactive environments)
    for fig_name, fig in figures.items():
        plt.figure(fig.number)
        plt.show()