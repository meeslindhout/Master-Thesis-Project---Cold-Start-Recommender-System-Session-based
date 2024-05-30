# Libraries for loading and processing data
import pandas as pd 
import torch

# Libraries for custom gym environment
import numpy as np
import gym
from gym import spaces

# Libraries for DQN model
import torch.nn as nn

# Libraries for DQN RL agent
import torch.optim as optim
import random
import os
from collections import defaultdict, deque
from datetime import datetime

# Prepare data for offline evaluation
class LogToTrajectoryConverter:
    '''
    This class is used to map a log file to a trajectory.
    A trajectory is a list of tuples, where each tuple contains the state, action, reward, and next state to be taken in the environment.
    When the trajectory is finished, the next state is set to None.
    
    Moreover, the reward function is defined in this class to calculate the reward for a given state and action (e.g., click = 1, add to cart = 2, purchase = 3).
    '''
    def __init__(self) -> None:
        self.reward_dict = {}
        self.tensor_trajectories = None
        self.data = None

    def load_dataset(self, data: pd.DataFrame):
        '''
        Load the dataset into the class. 
        '''
        self.data = data
        print('Data loaded successfully.')

    def set_rewards(self, reward_dict):
        '''
        Input is a dictionary with the action as the key and the reward as the value.
        '''
        self.reward_dict = reward_dict
        print('Rewards set successfully.')
    
    def create_ssar_tensor_trajectories(self, n_history):
        '''
        Create the trajectories for the SSAR model. Trajectories or sometimes called episodes are created from each session of the passed dataset.
        The trajectories are stored in tensors for faster processing in the RL agent.
        '''        
        trajectories_list = []
        grouped = self.data.groupby('session_id')

        for session_id, group in grouped:
            current_state = [0] * n_history
            for i in range(len(group)):
                action = group.iloc[i]['itemid']
                reward = self.reward_dict[group.iloc[i]['event']]
                next_state = current_state[1:] + [action]

                trajectories_list.append({
                    'session_id': session_id,
                    'state': current_state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state
                })

                current_state = next_state

        trajectories_df = pd.DataFrame(trajectories_list)

        trajectories_tensor = defaultdict(list)
        for _, row in trajectories_df.iterrows():
            state_tensor = torch.tensor(row['state'], dtype=torch.float32).cuda()
            action_tensor = torch.tensor(row['action'], dtype=torch.int64).cuda()
            reward_tensor = torch.tensor(row['reward'], dtype=torch.float32).cuda()
            next_state_tensor = torch.tensor(row['next_state'], dtype=torch.float32).cuda()

            trajectories_tensor[row['session_id']].append((state_tensor, action_tensor, reward_tensor, next_state_tensor))
        
        self.tensor_trajectories = [trajectory for trajectory in trajectories_tensor.values()]
        print('Trajectories created successfully.')
            
# Custom gym environment for offline evaluation
class OfflineEnv(gym.Env):
    def __init__(self, trajectories, n_history):
        super(OfflineEnv, self).__init__()
        self.trajectories = trajectories
        self.n_history = n_history
        self.current_step = 0
        self.current_trajectory = 0

        self.state_size = len(trajectories[0][0][0])
        self.action_size = max([t[1].item() for traj in trajectories for t in traj]) + 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

    def reset(self):
        self.current_step = 0
        self.current_trajectory = (self.current_trajectory + 1) % len(self.trajectories)
        state = self.trajectories[self.current_trajectory][self.current_step][0].cpu().numpy()
        return state

    def step(self, action):
        trajectory = self.trajectories[self.current_trajectory]
        self.current_step += 1

        if self.current_step >= len(trajectory):
            done = True
            next_state = np.zeros(self.state_size)
            reward = 0
        else:
            next_state = trajectory[self.current_step][0].cpu().numpy()
            reward = trajectory[self.current_step][2].item()
            done = self.current_step == len(trajectory) - 1

        return next_state, reward, done, {}

# DQN model that will be used in the RL agent
class DQN(nn.Module):
    def __init__(self, state_size, action_size, n_hiddenlayer_neurons=512):
        """
        Initializes the DQN (Deep Q-Network) class.

        Args:
            state_size (int): The size of the input state.
            action_size (int): The number of possible actions.
            n_hiddenlayer_neurons (int, optional): The number of neurons in the hidden layers. Defaults to 512.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, n_hiddenlayer_neurons)
        self.fc2 = nn.Linear(n_hiddenlayer_neurons, n_hiddenlayer_neurons)
        self.fc3 = nn.Linear(n_hiddenlayer_neurons, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN RL agent
class OfflineDQNAgent:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 learning_rate=3e-4, 
                 gamma=0.99,
                 n_history=1):
        self.n_history = n_history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.memory = deque(maxlen=10000)
        self.kpi_tracker = {
            'episode_rewards': [],
            'losses': []
        }

    def select_action(self, state, epsilon=0.1):
        """
        Selects an action based on the given state.

        Parameters:
            state (list): The current state of the environment.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.

        Returns:
            int: The selected action.

        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.model.fc3.out_features)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            return self.model(state_tensor).argmax().item()

    def step(self, state, action, reward, next_state, done):
        """
        Takes a step in the RL environment.

        Args:
            state (object): The current state of the environment.
            action (object): The action taken in the environment.
            reward (float): The reward received for taking the action.
            next_state (object): The next state of the environment after taking the action.
            done (bool): Whether the episode is done or not.

        Returns:
            None
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=512):
        """
        Trains the reinforcement learning model using a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The size of the batch to sample from the memory buffer. Defaults to 512.

        Returns:
            None
        """
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.kpi_tracker['losses'].append(loss.item())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, dataset_name):
        """
        Saves the agent so that it can be loaded without retraining it.

        Args:
            dataset_name (str): The name of the dataset.

        Raises:
            ValueError: If the dataset name is not provided.

        Returns:
            None
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # create directory if it does not exist
        if not os.path.exists('trained agents'):
            os.makedirs('trained agents')
            
        # raise error if dataset name is not provided
        if dataset_name is None:
            raise ValueError('Please provide a dataset name.')        
        # create folder for dataset if it does not exist
        if not os.path.exists(f'trained agents/{dataset_name}'):
            os.makedirs(f'trained agents/{dataset_name}')
        # save model        
        torch.save(self.model.state_dict(), f'trained agents/{dataset_name}/DQN trained agent {timestamp} n_hist{self.n_history}.pth')
      
    def load_model(self, filepath):        
        '''
        Loads a pre-trained model from the given file path.

        Args:
            filepath (str): Path to the file containing the saved model state_dict.

        Returns:
            None
        '''
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()

    def predict(self, states, n_predictions=1):
        '''
        Predict the top n actions for a given state. input is a list of states
        '''        
        if not isinstance(states, list):
            states = [states]
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)
        top_actions = torch.topk(q_values, n_predictions, dim=1).indices
        return top_actions.tolist()
