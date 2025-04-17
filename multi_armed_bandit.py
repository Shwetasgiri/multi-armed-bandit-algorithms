import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import time
import os

class BanditEnvironment:
    """
    Simulates a multi-armed bandit environment with specified number of arms
    and reward distributions.
    """
    def __init__(self, n_arms, means=None, stds=None):
        """
        Initialize the bandit environment.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit environment
        means : list or np.ndarray, optional
            Mean rewards for each arm. If None, randomly generated.
        stds : list or np.ndarray, optional
            Standard deviations for rewards. If None, set to 1.0 for all arms.
        """
        self.n_arms = n_arms
        
        # If means not provided, randomly generate them
        if means is None:
            self.means = np.random.normal(0, 1, n_arms)
        else:
            self.means = np.array(means)
            
        # If stds not provided, set them all to 1
        if stds is None:
            self.stds = np.ones(n_arms)
        else:
            self.stds = np.array(stds)
            
        # Store the best arm and its expected reward
        self.best_arm = np.argmax(self.means)
        self.optimal_reward = self.means[self.best_arm]
        
    def pull(self, arm):
        """
        Pull an arm and get a reward.
        
        Parameters:
        -----------
        arm : int
            The arm to pull
            
        Returns:
        --------
        float
            The reward obtained from pulling the arm
        """
        return np.random.normal(self.means[arm], self.stds[arm])


class BanditAlgorithm:
    """Base class for bandit algorithms"""
    
    def __init__(self, n_arms, name="Base Algorithm"):
        """
        Initialize the bandit algorithm.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit environment
        name : str
            Name of the algorithm
        """
        self.n_arms = n_arms
        self.name = name
        
        # Initialize performance tracking metrics
        self.reset()
        
    def reset(self):
        """Reset all tracking metrics"""
        self.counts = np.zeros(self.n_arms)  # Number of times each arm was pulled
        self.rewards = np.zeros(self.n_arms)  # Cumulative reward for each arm
        self.total_reward = 0  # Total cumulative reward
        self.action_history = []  # History of actions taken
        self.reward_history = []  # History of rewards received
        self.cumulative_reward_history = []  # History of cumulative rewards
        self.regret_history = []  # History of regret
        
    def select_arm(self):
        """
        Select an arm to pull. Must be implemented by subclasses.
        
        Returns:
        --------
        int
            The selected arm
        """
        raise NotImplementedError("Subclasses must implement select_arm method")
        
    def update(self, arm, reward, optimal_reward):
        """
        Update the algorithm's state based on the arm pulled and reward received.
        
        Parameters:
        -----------
        arm : int
            The arm that was pulled
        reward : float
            The reward received
        optimal_reward : float
            The optimal reward that could have been received
        """
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.total_reward += reward
        
        # Update histories
        self.action_history.append(arm)
        self.reward_history.append(reward)
        self.cumulative_reward_history.append(self.total_reward)
        
        # Calculate regret (difference between optimal and received reward)
        regret = optimal_reward - reward
        if len(self.regret_history) == 0:
            self.regret_history.append(regret)
        else:
            self.regret_history.append(self.regret_history[-1] + regret)
            
    def get_average_rewards(self):
        """
        Get the average reward for each arm.
        
        Returns:
        --------
        np.ndarray
            Average rewards for each arm
        """
        avg_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                avg_rewards[i] = self.rewards[i] / self.counts[i]
        return avg_rewards


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm for multi-armed bandit problems.
    
    This algorithm selects the best arm with probability 1-epsilon,
    and explores randomly with probability epsilon.
    """
    
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize the Epsilon-Greedy algorithm.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit environment
        epsilon : float
            Exploration parameter, between 0 and 1
        """
        super().__init__(n_arms, name=f"ε-Greedy (ε={epsilon})")
        self.epsilon = epsilon
        
    def select_arm(self):
        """
        Select an arm using the epsilon-greedy strategy.
        
        Returns:
        --------
        int
            The selected arm
        """
        # Explore: with probability epsilon, select a random arm
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        
        # Exploit: with probability 1-epsilon, select the arm with highest average reward
        avg_rewards = self.get_average_rewards()
        
        # If some arms haven't been tried yet, try them first
        untried_arms = np.where(self.counts == 0)[0]
        if len(untried_arms) > 0:
            return np.random.choice(untried_arms)
            
        # Otherwise, select the arm with the highest average reward
        return np.argmax(avg_rewards)


class UCB(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm for multi-armed bandit problems.
    
    This algorithm balances exploration and exploitation by selecting the arm
    with the highest upper confidence bound.
    """
    
    def __init__(self, n_arms, c=2.0):
        """
        Initialize the UCB algorithm.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit environment
        c : float
            Exploration parameter controlling the confidence bound
        """
        super().__init__(n_arms, name=f"UCB (c={c})")
        self.c = c
        
    def select_arm(self):
        """
        Select an arm using the UCB strategy.
        
        Returns:
        --------
        int
            The selected arm
        """
        # If some arms haven't been tried yet, try them first
        untried_arms = np.where(self.counts == 0)[0]
        if len(untried_arms) > 0:
            return np.random.choice(untried_arms)
            
        # Calculate total number of pulls
        total_counts = np.sum(self.counts)
        
        # Calculate UCB values for each arm
        avg_rewards = self.get_average_rewards()
        exploration_bonus = self.c * np.sqrt(np.log(total_counts) / self.counts)
        ucb_values = avg_rewards + exploration_bonus
        
        # Select the arm with the highest UCB value
        return np.argmax(ucb_values)


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling algorithm for multi-armed bandit problems.
    
    This algorithm models each arm's reward as a Beta distribution and
    samples from these distributions to decide which arm to pull.
    """
    
    def __init__(self, n_arms):
        """
        Initialize the Thompson Sampling algorithm.
        
        Parameters:
        -----------
        n_arms : int
            Number of arms in the bandit environment
        """
        super().__init__(n_arms, name="Thompson Sampling")
        # For each arm, track alpha and beta parameters of the Beta distribution
        self.alphas = np.ones(n_arms)  # Prior successes + 1
        self.betas = np.ones(n_arms)   # Prior failures + 1
        
    def reset(self):
        """Reset all tracking metrics"""
        super().reset()
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        
    def select_arm(self):
        """
        Select an arm using the Thompson Sampling strategy.
        
        Returns:
        --------
        int
            The selected arm
        """
        # Sample from Beta distributions for each arm
        samples = np.array([np.random.beta(self.alphas[i], self.betas[i]) 
                          for i in range(self.n_arms)])
        
        # Select the arm with the highest sampled value
        return np.argmax(samples)
        
    def update(self, arm, reward, optimal_reward):
        """
        Update the algorithm's state based on the arm pulled and reward received.
        
        Parameters:
        -----------
        arm : int
            The arm that was pulled
        reward : float
            The reward received
        optimal_reward : float
            The optimal reward that could have been received
        """
        # Call the parent class's update method
        super().update(arm, reward, optimal_reward)
        
        # Update Beta distribution parameters
        # We need to properly normalize rewards to [0, 1] range
        # and ensure our beta parameters stay positive
        
        # Find min and max rewards across all arms to normalize
        min_possible = -2.0  # Assuming lowest possible mean - 2*std ~ -2
        max_possible = 2.0   # Assuming highest possible mean + 2*std ~ 2
        
        # Scale reward to [0, 1] range
        normalized_reward = max(0.01, min(0.99, (reward - min_possible) / (max_possible - min_possible)))
        
        # Update success and failure counts with minimum values to avoid non-positive parameters
        self.alphas[arm] += normalized_reward
        self.betas[arm] += (1 - normalized_reward)


class BanditSimulation:
    """
    Simulation class for running and comparing different multi-armed bandit algorithms.
    """
    
    def __init__(self, env, algorithms, n_iterations=1000):
        """
        Initialize the simulation.
        
        Parameters:
        -----------
        env : BanditEnvironment
            The bandit environment to run simulations on
        algorithms : list of BanditAlgorithm
            The algorithms to compare
        n_iterations : int
            Number of iterations (arm pulls) to simulate
        """
        self.env = env
        self.algorithms = algorithms
        self.n_iterations = n_iterations
        self.results = {}
        
    def run(self, verbose=True):
        """
        Run the simulation for all algorithms.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress updates
        
        Returns:
        --------
        dict
            Results of the simulation
        """
        start_time = time.time()
        
        # Reset all algorithms
        for algo in self.algorithms:
            algo.reset()
            
        # Run each algorithm for n_iterations
        for algo in self.algorithms:
            if verbose:
                print(f"Running simulation for {algo.name}...")
                
            for i in range(self.n_iterations):
                # Select an arm using the algorithm
                arm = algo.select_arm()
                
                # Pull the arm and get reward
                reward = self.env.pull(arm)
                
                # Update the algorithm's state
                algo.update(arm, reward, self.env.optimal_reward)
                
                # Print progress every 10% of iterations
                if verbose and i % (self.n_iterations // 10) == 0 and i > 0:
                    print(f"  {i}/{self.n_iterations} iterations completed...")
            
            # Store results
            self.results[algo.name] = {
                'actions': algo.action_history,
                'rewards': algo.reward_history,
                'cumulative_rewards': algo.cumulative_reward_history,
                'regret': algo.regret_history,
                'arm_counts': algo.counts,
                'optimal_arm_pull_rate': np.sum(np.array(algo.action_history) == self.env.best_arm) / self.n_iterations
            }
            
            if verbose:
                print(f"  Completed. Optimal arm selection rate: {self.results[algo.name]['optimal_arm_pull_rate']:.2%}")
                
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Simulation completed in {elapsed_time:.2f} seconds.")
            
        return self.results
    
    def plot_results(self, metrics=None, save_dir=None, figsize=(12, 8)):
        """
        Plot the results of the simulation.
        
        Parameters:
        -----------
        metrics : list of str, optional
            The metrics to plot. If None, plot all metrics.
        save_dir : str, optional
            Directory to save the plots. If None, don't save.
        figsize : tuple, optional
            Figure size for plots
            
        Returns:
        --------
        dict
            Dictionary of matplotlib figures
        """
        if not self.results:
            raise ValueError("No results to plot. Run the simulation first.")
            
        if metrics is None:
            metrics = ['cumulative_rewards', 'regret', 'optimal_arm_selection']
            
        figures = {}
            
        # Create save directory if it doesn't exist
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Plotting cumulative rewards
        if 'cumulative_rewards' in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            for algo_name, result in self.results.items():
                ax.plot(result['cumulative_rewards'], label=algo_name)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title('Cumulative Rewards Over Time')
            ax.legend()
            ax.grid(True)
            
            figures['cumulative_rewards'] = fig
            
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'cumulative_rewards.png'), dpi=300, bbox_inches='tight')
                
        # Plotting regret
        if 'regret' in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            for algo_name, result in self.results.items():
                ax.plot(result['regret'], label=algo_name)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Cumulative Regret')
            ax.set_title('Cumulative Regret Over Time')
            ax.legend()
            ax.grid(True)
            
            figures['regret'] = fig
            
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'regret.png'), dpi=300, bbox_inches='tight')
                
        # Plotting optimal arm selection rate
        if 'optimal_arm_selection' in metrics:
            # Calculate moving average of optimal arm selection
            window_size = min(1000, self.n_iterations // 10)
            
            fig, ax = plt.subplots(figsize=figsize)
            for algo_name, result in self.results.items():
                optimal_selections = [1 if a == self.env.best_arm else 0 for a in result['actions']]
                cumsum = np.cumsum(optimal_selections)
                # Calculate moving average
                moving_avg = [cumsum[i] / (i + 1) for i in range(len(cumsum))]
                ax.plot(moving_avg, label=algo_name)
                
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Optimal Arm Selection Rate')
            ax.set_title('Optimal Arm Selection Rate Over Time')
            ax.legend()
            ax.grid(True)
            
            figures['optimal_arm_selection'] = fig
            
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'optimal_arm_selection.png'), dpi=300, bbox_inches='tight')
                
        # Plotting arm selection counts
        if 'arm_counts' in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set up bar positions
            n_algos = len(self.algorithms)
            n_arms = self.env.n_arms
            width = 0.8 / n_algos
            
            for i, (algo_name, result) in enumerate(self.results.items()):
                # Calculate positions for this algorithm's bars
                positions = np.arange(n_arms) + (i - n_algos/2 + 0.5) * width
                # Plot the bars
                bars = ax.bar(positions, result['arm_counts'], width, label=algo_name)
                
            # Highlight the optimal arm
            ax.axvline(x=self.env.best_arm, color='r', linestyle='--', alpha=0.5, 
                      label='Optimal Arm')
            
            ax.set_xlabel('Arm')
            ax.set_ylabel('Pull Count')
            ax.set_title('Arm Selection Counts')
            ax.set_xticks(np.arange(n_arms))
            ax.legend()
            ax.grid(True, axis='y')
            
            figures['arm_counts'] = fig
            
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'arm_counts.png'), dpi=300, bbox_inches='tight')
                
        plt.close('all')  # Close all figures to free memory
        return figures
        
    def performance_summary(self):
        """
        Generate a performance summary for all algorithms.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with performance metrics
        """
        if not self.results:
            raise ValueError("No results to summarize. Run the simulation first.")
            
        # Create summary data
        summary_data = []
        
        for algo_name, result in self.results.items():
            final_reward = result['cumulative_rewards'][-1]
            final_regret = result['regret'][-1]
            optimal_arm_rate = result['optimal_arm_pull_rate']
            
            summary_data.append({
                'Algorithm': algo_name,
                'Total Reward': final_reward,
                'Average Reward': final_reward / self.n_iterations,
                'Total Regret': final_regret,
                'Average Regret': final_regret / self.n_iterations,
                'Optimal Arm Selection Rate': optimal_arm_rate
            })
            
        return pd.DataFrame(summary_data)


def run_demo(n_arms=10, n_iterations=10000, seed=42):
    """
    Run a demonstration of the multi-armed bandit algorithms.
    
    Parameters:
    -----------
    n_arms : int
        Number of arms in the bandit environment
    n_iterations : int
        Number of iterations to run
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (simulation, results, summary_df, figures)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create bandit environment
    # Using random means between 0 and 1
    means = np.random.normal(0, 1, n_arms)
    stds = np.ones(n_arms) * 0.5
    env = BanditEnvironment(n_arms, means, stds)
    
    # Print the true mean rewards and optimal arm
    print(f"True mean rewards: {means}")
    print(f"Optimal arm: {env.best_arm} (expected reward: {env.optimal_reward:.4f})")
    
    # Create algorithms
    algorithms = [
        EpsilonGreedy(n_arms, epsilon=0.1),
        UCB(n_arms, c=2.0),
        ThompsonSampling(n_arms)
    ]
    
    # Create and run simulation
    sim = BanditSimulation(env, algorithms, n_iterations)
    results = sim.run()
    
    # Get performance summary
    summary_df = sim.performance_summary()
    print("\nPerformance Summary:")
    print(summary_df)
    
    # Plot results
    metrics = ['cumulative_rewards', 'regret', 'optimal_arm_selection', 'arm_counts']
    figures = sim.plot_results(metrics)
    
    return sim, results, summary_df, figures


if __name__ == "__main__":
    # Run demonstration with 10 arms and 10,000 iterations
    sim, results, summary, figures = run_demo(n_arms=10, n_iterations=10000)
    
    # Display a few of the plots
    plt.figure(figures['cumulative_rewards'].number)
    plt.show()
    
    plt.figure(figures['regret'].number)
    plt.show()
    
    plt.figure(figures['optimal_arm_selection'].number)
    plt.show()