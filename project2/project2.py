"""
Project 2 for aa228/CS238, Winter 2023.
"""

import numpy as np
import argparse
import pandas as pd
import os
import timeit
import tqdm
import plotly.express as px

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_size", type=str, required=True, help="Size of dataset. Either small, medium, or large.")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to output directory.")
    args = parser.parse_args()
    return args

class QLearning():
    """Q-learning algorithm as in the textbook."""
    def __init__(self, n_states=10, n_actions=4, discount=0.95, learning_rate=0.1):
        """
        Contains the Q function along with useful meta-data.

        Args:
          n_states: (int) number of states in the environment
          n_actions: number of actions in the environment
          discount: discount factor
          learning_rate: learning rate   
        """
        self.state_space = np.arange(n_states)
        self.action_space = np.arange(n_actions)
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((n_states, n_actions))

    def update(self, state, action, reward, new_state):
        """
        Update the Q function.

        Args:
          state: (int) current state
          action: (int) current action
          reward: (float) reward given by the environment for this state action pair.
          new_state: (int) next state as given by a simulator.
        """
        self.Q[state, action] += self.learning_rate * (reward + self.discount * np.max(self.Q[new_state, :]) - self.Q[state, action])

def main():
    """
    Runs the program
    """
    print(f"Args are {args}.")
    # load data from args.data_file
    data_path = os.path.join("data", args.data_size) + ".csv"
    df = pd.read_csv(data_path)
    print(df)
    print(f"Columnsa re {df.columns}")

    # run Q-learning
    small_config = {"n_states": 100, "n_actions": 4, "discount": 0.95}
    medium_config = {"n_states": 500 * 100, "n_actions": 7, "discount": 0.0}
    large_config = {"n_states": 312020, "n_actions": 9, "discount": 0.95}
    config = None
    if args.data_size == "small":
        config = small_config
    elif args.data_size == "medium":
        config = medium_config
    elif args.data_size == "large":
        config = large_config
    q = QLearning(**config, learning_rate=0.01)
    n_samples = df.shape[0]

    for _ in tqdm.tqdm(range(100)):
        for j in range(n_samples):
            q.update(df.iloc[j, 0] - 1, df.iloc[j, 1] - 1, df.iloc[j, 2], df.iloc[j, 3] - 1)

    # save results to args.output_file
    print(q.Q.shape)
    print(q.Q)
    policy = np.argmax(q.Q, axis=1)
    # add one to the policy to make it 1-indexed
    policy += 1
    print(policy)
    
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
    output_path = os.path.join(args.output_file, args.data_size) + ".policy"
    np.savetxt(output_path, policy, fmt="%d")

    # Plot a histogram of the policy
    fig = px.histogram(policy)
    fig.show()


if __name__ == "__main__":
    args = parse_args()
    runtime = timeit.timeit(main, number=1)
    print(f"The time to run the code on {args.data_size} was {runtime}s.")
