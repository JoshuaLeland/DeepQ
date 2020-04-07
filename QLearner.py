import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from time import sleep

from dqn_utils import ReplayBuffer
from atari_wrappers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class AtariNetwork(nn.Module):

	def __init__(self, h, w, c, num_frames, outputs):
		super(AtariNetwork, self).__init__()
		self.conv1 = nn.Conv2d(c*num_frames, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 3, stride = 1):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
		linear_input_size = convw * convh * 64
		self.head_1 = nn.Linear(linear_input_size, 512)
		self.head_2 = nn.Linear(512, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.head_1(x.view(x.size(0), -1)))
		return self.head_2(x)

class DQNAgent:

	def __init__(self, env, height, width, channels, num_frames, actions, learning_rate=1e-4, gamma=0.99, buffer_size=10000, num_timesteps=2e8, doubleQ=True):
		
		self.env = env
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.replay_buffer = ReplayBuffer(buffer_size, num_frames)
		self.num_iterations = num_timesteps / 4
		self.double = doubleQ
	
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Double Q
		self.current = AtariNetwork(height, width, channels, num_frames, actions).to(self.device)
		self.target = AtariNetwork(height, width, channels, num_frames, actions).to(self.device)

		# optimizers.
		self.optimizer1 = torch.optim.Adam(self.current.parameters(), lr=learning_rate, eps=1e-4)

		#Schedule.
		self.scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=[self.num_iterations/2], gamma=0.5)

		# Use Huber loss.
		self.loss1 = torch.nn.SmoothL1Loss()

		# start the least env.
		self.reset_env()

		# logging.
		self.running_mean_episode_reward = 0
		self.steps_episode = 0
		self.mean_episode_rewards = []
		self.best_mean_episode_reward = -float('inf')
		
	def get_action(self, state, eps=0.20):
		if(random.random() < eps):
			return self.env.action_space.sample()

		state = torch.FloatTensor(self._normalize_states(state)).transpose(0,2).float().unsqueeze(0).to(self.device)
		qvals = self.current.forward(state)
		action = np.argmax(qvals.cpu().detach().numpy())
		
		return action

	def compute_loss(self, batch):     
		states, actions, rewards, next_states, dones = batch
		states = torch.FloatTensor(self._normalize_states(states)).transpose(1,3).float().to(self.device)
		actions = torch.LongTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device)
		next_states = torch.FloatTensor(self._normalize_states(next_states)).transpose(1,3).float().to(self.device)
		dones = torch.FloatTensor(dones).to(self.device)

		# resize tensors
		actions = actions.view(actions.size(0), 1)
		dones = dones.view(dones.size(0), 1)

		# compute loss
		curr_Q1 = self.current.forward(states).gather(1, actions)

		if self.double:
			_, a_prime = self.current.forward(next_states).detach().max(1)
		else:
			_, a_prime = self.target.forward(next_states).detach().max(1)

		next_Q = self.target.forward(next_states).detach().gather(1, a_prime.unsqueeze(1))
		next_Q = next_Q.squeeze()

		next_Q = next_Q.view(next_Q.size(0), 1)
		expected_Q = rewards.view(-1,1) + (1 - dones) * self.gamma * next_Q

		# Change these to smooth loss.
		loss1 = self.loss1(curr_Q1, expected_Q.detach())

		return loss1
		
	def update(self, batch_size, update_target):
		batch = self.replay_buffer.sample(batch_size)
		loss1= self.compute_loss(batch)

		self.optimizer1.zero_grad()
		loss1.backward()
		self.optimizer1.step()
		self.scheduler1.step()

		if update_target:
			torch.save(self.current.state_dict(), "current_model_state")
			self.target.load_state_dict(self.current.state_dict())



	def step_env(self, eps=0.2):

		# store last observation.
		self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

		# encode it and get input for network
		network_obs = self.replay_buffer.encode_recent_observation()

		# Get the next action.
		action = self.get_action(network_obs, eps)

		# step the envoirment.
		next_frame, reward, done, info = self.env.step(action)

		self.last_obs = next_frame

		# update results.
		self.running_mean_episode_reward+=reward
		self.steps_episode+=1

		#store result.
		self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

		# if done reset the episode.
		if done:

			# last episode.
			self.reset_env()
			avg_episode_reward = self.running_mean_episode_reward
			self.mean_episode_rewards.append(avg_episode_reward)

			if self.best_mean_episode_reward < avg_episode_reward:
				self.best_mean_episode_reward = avg_episode_reward
				print("New highest reward: %s\n"%str(self.best_mean_episode_reward))

			self.running_mean_episode_reward = 0
			self.steps_episode = 0

		return self.mean_episode_rewards, action, reward, done

	def _normalize_states(self, states):
		return states / 255.0

	def reset_env(self):
		self.last_obs = self.env.reset()

	# Render the episode.
	def render_episode(self):

		done = False
		self.reset_env()

		# Render an episode.
		while not done:
			self.env.render()
			_,_,_, done = self.step_env(-10)
			sleep(0.016) # 1/60th of a second



