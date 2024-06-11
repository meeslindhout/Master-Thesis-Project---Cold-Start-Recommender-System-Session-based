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
import wandb


class OfflineEnv(gym.Env):
    '''
    Custom offline gym environment that 'simulates' the actions of a recommender system. 
    '''
    def __init__(self, episodes, n_history):
        super(OfflineEnv, self).__init__()
        self.episodes = episodes
        self.n_history = n_history
        self.current_step = 0
        self.current_episode = 0

        self.state_size = len(episodes[0][0][0])
        self.action_size = max([t[1].item() for traj in episodes for t in traj]) + 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

    def reset(self):
        self.current_step = 0
        self.current_episode = (self.current_episode + 1) % len(self.episodes)
        state = self.episodes[self.current_episode][self.current_step][0].cpu().numpy()
        return state

    def step(self, action):
        episode = self.episodes[self.current_episode]
        self.current_step += 1

        if self.current_step >= len(episode):
            done = True
            next_state = np.zeros(self.state_size)
            reward = 0
        else:
            next_state = episode[self.current_step][0].cpu().numpy()
            reward = episode[self.current_step][2].item()
            done = self.current_step == len(episode) - 1

        return next_state, reward, done, {}

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
    def __init__(self, state_size=None, action_size=None, learning_rate=3e-4, gamma=0.99, n_history=1, memory = 10_000, mode='training'):
        '''
        TODO: write offline DQN agent class documentation
        TODO: add more kpi performance metrics and connect to wandb
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.n_history = n_history
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() and mode == 'training' else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.memory = deque(maxlen=memory)
        self.kpi_tracker = {
            'episode_rewards': [],
            'losses': []
            }
        self.session = -1
        self.session_items = []

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
        
        # Log step-level metrics to wandb
        wandb.log({"reward": reward, "done": done})

    def train(self, batch_size=64):
        """
        Trains the reinforcement learning model using a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The size of the batch to train defaults to 64.

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
        
        # log loss to wandb
        wandb.log({'loss': loss.item()})

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
        print('Model specific parameters')
        print(f"n_history: {self.n_history}")
        print(f"state_size: {self.state_size}")
        print(f"action_size: {self.action_size}")
        file_name = f'{timestamp} - n_hist {self.n_history} state_size {self.state_size} action_size {self.action_size}.pth'
        torch.save(self.model.state_dict(), f'trained agents/{dataset_name}/{file_name}')
        print(f'Model successfully saved!')
        print(f"Saved path: \ntrained agents/{dataset_name}/{file_name}")
             
    def load_model(self, filepath):        
        '''
        Loads a pre-trained model from the given file path.

        Args:
            filepath (str): Path to the file containing the saved model state_dict.

        Returns:
            None
        '''
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f'Pretrained model successfully loaded')

    def predict(self, states, n_predictions=1):
        '''
        Predict the top n actions for a given state. input is a list of states.
        For example, if n_history = 3, the state should be a list of 3 integers that indicate the last 3 actions taken in the environment. aka the item ids.
        '''        
        if not isinstance(states, list):
            states = [states]
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)
        top_actions = torch.topk(q_values, n_predictions, dim=1).indices
        return top_actions.tolist()
  
    def predict_scores(self, states, predict_for_item_ids):
        # deze functie wordt de predict_next() in de recsys code
        '''
        Predict the scores for a given state. Input is a list of states.
        For example, if n_history = 3, the state should be a list of 3 integers that indicate the last 3 actions taken in the environment, aka the item ids.
        
        Parameters
        --------
        states : list
            List of states, where each state is a list of item IDs.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            
        Returns
        --------
        out : list
            List of prediction scores for the selected items.
        '''        
        if not isinstance(states, list):
            states = [states]
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)
        
        # Extract the scores for the items in predict_for_item_ids
        scores = []
        for q_value in q_values:
            item_scores = [q_value[item_id].item() for item_id in predict_for_item_ids]
            scores.extend(item_scores)
        
        return scores

class rl_recommender():
    '''
    Parameters - preprocessing
    -----------   
    n_history : int
       Number of previous items that are taken into account to predict the next item.
    reward_dict : dict
       The reward for a given state and action (e.g., click = 1, add to cart = 2, purchase = 3)
       For example: {0: 5, 1: 8, 2: 10}
    event_key : int
        header of the event column (default: 'event')
    -----------------------------------------------------------------------------

    Parameters - dqn
    -----------
    mode : str 
        Set to training or predicting (default: 'training')
    trained_model_path : str
        If mode='predicting', then give the path of the pth file
    num_episodes: int
        Number of episodes that the DQN takes to learn the environment (default: 1000) small number as default for debugging
    batch_size: int
        Batch size of neural net (default: 64) more info see: https://ai.stackexchange.com/questions/23254/is-there-a-logical-method-of-deducing-an-optimal-batch-size-when-training-a-dee
        TDLR: for best performance choose 512
    target_update_freq: int
        Target update frequency of DQN, must be found by trial and error / hyper parameter tuning.
    memory: int
        Memory of DQN agent (default: 10.000) more info see: https://ai.stackexchange.com/questions/42462/what-is-the-purpose-of-a-replay-memory-buffer-in-deep-q-learning-networks
        TODO: check if 10.000 makes sense in recommender systems
    learning_rate : float
        learning rate (default: 3e-4)
    gamma: float
        exploration rate (default: 0.99)
    dataset_name: str
        Enter the name of the dataset.
        Trained agent Will be saved in 'trained agents/{dataset_name}'.
    -----------------------------------------------------------------------------
    
    Parameters - for pretrained model
    ----------- 
    file_path: str
        Name of pth file with trained weights of DQN
        For example: 'trained agents\DQN trained agent 20240530_085334 n_hist 1.pth'
    state_size: int
        Number of states
    action_size: int
        Number of possible states
    
    Parameters - standard (in all recommender algorithms of session-rec github repo) 
    ----------- 
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    -----------------------------------------------------------------------------
    '''
    def __init__(self, 
                 n_history=None, reward_dict = {}, event_key='event',
                 
                 mode='training', num_episodes=1000, batch_size=64, target_update_freq=None, memory=10_000, learning_rate=3e-4, gamma=0.99, dataset_name='dataset_not_undefined', log_to_wandb=True, custom_wandb_note=None,
                 
                 file_path=None, state_size=None, action_size=None,
                 
                 session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time'):
        
        # RS parameters for preprocessing
        self.n_history = n_history
        self.reward_dict = reward_dict
        self.event_key = event_key
        
        # DQN parameters
        self.agent = None
        self.mode = mode
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = memory
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.dataset_name = dataset_name
        self.log_to_wandb = log_to_wandb
        self.custom_wandb_note = custom_wandb_note
        
        # when model is loaded
        self.file_path = file_path
        self.state_size = state_size
        self.action_size = action_size
        

        # RS framework parameters
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.session = -1
        self.session_items = []
               
    def convert_to_episodes(self, data: pd.DataFrame):
        '''
        Convert the dataset to episodes for the SSAR model. 
        An episode is a list of tuples, where each tuple contains the state, action, reward, and next state to be taken in the environment.
        The episodes are stored in tensors for faster processing in the RL agent.
        '''
        device = torch.device("cuda" if torch.cuda.is_available() and self.mode == 'training' else "cpu")
        
        episodes_list = []
        grouped = data.groupby(self.session_key)

        for session_id, group in grouped:
            current_state = [0] * self.n_history
            for i in range(len(group)):
                action = group.iloc[i][self.item_key]
                reward = self.reward_dict[group.iloc[i][self.event_key]]
                next_state = current_state[1:] + [action]

                episodes_list.append({
                    'session_id': session_id,
                    'state': current_state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state
                })

                current_state = next_state

        episodes_df = pd.DataFrame(episodes_list)

        episodes_tensor = defaultdict(list)
        for _, row in episodes_df.iterrows():
            state_tensor = torch.tensor(row['state'], dtype=torch.int64).to(device)
            action_tensor = torch.tensor(row['action'], dtype=torch.int64).to(device)
            reward_tensor = torch.tensor(row['reward'], dtype=torch.float16).to(device)
            next_state_tensor = torch.tensor(row['next_state'], dtype=torch.int64).to(device)

            episodes_tensor[row['session_id']].append((state_tensor, action_tensor, reward_tensor, next_state_tensor))
        
        tensor_episodes = [episode for episode in episodes_tensor.values()]
        print('Episode tensors created successfully')
        return tensor_episodes
       
    def fit(self, train, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs, one for the timestamp of the events (unix timestamps) and one for the events that explains the type of event.
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''            
        if self.mode == 'training':
            # Preprocess data
            print('Started converting train data to tensors...')
            episode_tensors = self.convert_to_episodes(data = train)
            # Initialise offline gym environment
            print('Initializing gym environment...')
            env = OfflineEnv(episodes=episode_tensors, n_history=self.n_history)
            # Initialise RL agent
            print('Initializing offline training agent...')
            self.agent = OfflineDQNAgent(
                state_size = env.observation_space.shape[0], # observed ItemIds with each observed space length = n_history
                action_size = env.action_space.n, # ItemIds to recommend
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_history=self.n_history,
                memory=self.memory,
                mode=self.mode
                )
            
            # Initialise wandb to track metrics
            wandb.init(
                # set the wandb project where this run will be logged. Dataset name and n_history determine project name
                project='RecSys RL',
                notes=self.custom_wandb_note,
                config={
                    "learning_rate": self.learning_rate,
                    "gamma": self.agent.gamma,
                    "architecture": "DQN",
                    "dataset_name": self.dataset_name,
                    "version": 1.0,
                    "epochs": self.num_episodes,
                    "n_history": self.n_history,
                    "reward_dict": str(self.reward_dict),
                    "memory": self.memory,
                    "batch_size": self.batch_size,
                    "target_update_freq": self.target_update_freq,
                },
                mode="online" if self.log_to_wandb else "disabled"
            )
            # Train RL
            print('Started training loop...')
            for episode in range(self.num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.agent.step(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward

                self.agent.train(self.batch_size)
                if episode % self.target_update_freq == 0:
                    self.agent.update_target_model()

                self.agent.kpi_tracker['episode_rewards'].append(episode_reward)
                
                # Log episode reward and success rate to wandb
                wandb.log({
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "success_rate": int(episode_reward > 0)  # Assuming success is defined as positive reward
                })

                
                print(f"Episode {episode + 1}/{self.num_episodes} completed with reward: {episode_reward}")            
            print('Training completed!')
            
            # Safe trained model
            print('Saving model...')
            self.agent.save_model(dataset_name=self.dataset_name)
            
            
        if self.mode == 'predicting':
            # initialize model object
            print('Initializing offline training agent')
            self.agent = OfflineDQNAgent(state_size=self.state_size, action_size=self.action_size)
            # load weights of network
            print(f'Loading pretrained model from: {self.file_path}')
            self.agent.load_model(filepath=self.file_path)
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, timestamp=0, skip=False, mode_type='view'):
        
        # Saving the interacted item ids in the session
        # Check if the session id is the same as the previous session id
        if session_id != self.session: # if not, empty the session_items list and create a new list with n_history length list with all zeros
            self.session_items = np.zeros(self.n_history)
            self.session = session_id
        
        if mode_type == 'view': # if the event is a view, add the item to the session_items list
            # move all items one step back and add the new item to the last position
            # print(f'session items before update{self.session_items}')
            self.session_items = np.roll(self.session_items, -1)
            # print(f'session items after roll{self.session_items}')
            self.session_items[-1] = input_item_id
            # print(f'session items after update{self.session_items}')
            
        if skip: # if skipped, return the last item in the session_items list
            return
        
        # the session items list is now the state that is used to predict the next item with the DQN trained agent
        preds = self.agent.predict_scores(states = self.session_items,
                                          predict_for_item_ids = predict_for_item_ids)
    
        # print(preds)        
        # return the prediction scores for the items in the predict_for_item_ids list
        # the prediction scores are the Q-values of the DQN model

        return pd.Series(data=preds, index=predict_for_item_ids)