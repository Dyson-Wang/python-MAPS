from .SplitDecision import *
from .MAB.mab_constants import *
from .MAB.utils import *
import random
import numpy as np
import pickle
from os import path, mkdir
from copy import deepcopy

class MABDecider(SplitDecision):
	def __init__(self, train=False):
		super().__init__()
		self.applications = ['mnist', 'fashionmnist', 'cifar100']
		layer_intervals = [5, 8, 15] # 用于不同应用程序的层间隔配置
		self.average_layer_intervals = dict(zip(self.applications, layer_intervals))
		self.workflowids_checked = []
		self.epsilon = 0.95 # 以 epsilon 概率进行随机选择（探索）以 1 - epsilon 概率选择当前回报最高的动作（开发）
		self.r_thresh = 0.45 # r_thresh 是一个阈值参数，用于区分“低回报”和“高回报”
		self.low_rewards, self.low_counts = np.zeros(2), np.zeros(2)
		self.high_rewards, self.high_counts = np.zeros(2), np.zeros(2)
		self.train = train
		random.seed(1)
		self.load_model()

	# 加载已存在的MAB模型或者新建训练模型
	def load_model(self):
		if path.exists(SAVE_PATH):
			print(color.GREEN+"Loading pre-trained MAB model"+color.ENDC)
			with open(SAVE_PATH, 'rb') as f:
				self.model = pickle.load(f)
				# 读取
				self.average_layer_intervals, self.epsilon, self.r_thresh, all_arrays = self.model[-1]
				self.low_rewards, self.low_counts, self.high_rewards, self.high_counts = all_arrays
		else:
			print(color.GREEN+"Creating new MAB model"+color.ENDC)
			SAVE_DIR = SAVE_PATH.split('mab.pt')[0]
			if not path.exists(SAVE_DIR): mkdir(SAVE_DIR)
			self.model = []

	def save_model(self):
		print(color.GREEN+"Saving MAB model"+color.ENDC)
		all_arrays = self.low_rewards, self.low_counts, self.high_rewards, self.high_counts
		self.model.append(deepcopy((self.average_layer_intervals, self.epsilon, self.r_thresh, all_arrays)))
		with open(SAVE_PATH, 'wb') as f:
			pickle.dump(self.model, f)
		plot_graphs(self.model)

	def updateAverages(self):
		for WorkflowID in self.env.destroyedworkflows:
			if WorkflowID not in self.workflowids_checked:
				dict_ = self.env.destroyedworkflows[WorkflowID]
				workflow = dict_['application'].split('/')[1].split('_')[0]
				decision = dict_['application'].split('/')[1].split('_')[1]
				decision = 0 if decision == self.choices[0] else 1
				if decision == 0:
					intervals = dict_['destroyAt'] - dict_['createAt']
					self.average_layer_intervals[workflow] = 0.1 * intervals + 0.9 * self.average_layer_intervals[workflow]

	def updateRewards(self):
		rewards = []
		for WorkflowID in self.env.destroyedworkflows:
			if WorkflowID not in self.workflowids_checked:
				self.workflowids_checked.append(WorkflowID)
				dict_ = self.env.destroyedworkflows[WorkflowID]
				workflow = dict_['application'].split('/')[1].split('_')[0]
				decision = dict_['application'].split('/')[1].split('_')[1]
				decision = 0 if decision == self.choices[0] else 1
				intervals = dict_['destroyAt'] - dict_['createAt']
				sla = dict_['sla']
				sla_reward = 1 if intervals <= sla else 0
				acc_reward = max(0, (dict_['result'][0]/dict_['result'][1] - 0.9)) * 10
				reward = Coeff_SLA * sla_reward + Coeff_Acc * acc_reward
				rewards.append(reward)
				low = sla < self.average_layer_intervals[workflow]
				if low:
					self.low_counts[decision] += 1
					self.low_rewards[decision] = self.low_rewards[decision] + (reward - self.low_rewards[decision]) / self.low_counts[decision]
				else:
					self.high_counts[decision] += 1
					self.high_rewards[decision] = self.high_rewards[decision] + (reward - self.high_rewards[decision]) / self.high_counts[decision]
		return sum(rewards)/(len(rewards)+1e-5)

	def decision(self, workflowlist):
		# epsilon-greedy策略
		self.updateAverages()
		avg_reward = self.updateRewards()
		decisions = []
		for _, _, sla, workflow in workflowlist: # sla int, workflow "MNIST"
			if self.train and random.random() < self.epsilon:
				# 探索：随机选择一个决策
				decisions.append(random.choice(self.choices))
				if self.train: print('Random Decision:', decisions[-1])
			else:
				# 利用：选择当前已知的最优决策
				low = sla < self.average_layer_intervals[workflow.lower()]
				if low:
					decisions.append(self.choices[np.argmax(self.low_rewards)])
				else:
					decisions.append(self.choices[np.argmax(self.high_rewards)])
				if self.train: print('MAB Decision:', decisions[-1])
		# 根据奖励值调整epsilon和阈值
		# Reward based decay
		if avg_reward >= self.r_thresh:
			self.epsilon *= 0.98
			self.r_thresh = min(1, 1.05*self.r_thresh)
		if self.train: self.save_model()
		return decisions