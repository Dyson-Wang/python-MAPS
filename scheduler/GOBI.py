import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *

class GOBIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		self.model = eval(data_type+"()") # train.py - energy_latency_10()
		# 拿到模型 加载scheduler/BaGTI/checkpoints/energy_latency_10_Trained.ckpt 之后的模型
		# model.load_state_dict(checkpoint['model_state_dict'])
		self.model, _, _, _ = load_model(data_type, self.model, data_type) # 只用了模型新/旧
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1]) # 主机数 10
		dtl = data_type.split('_') # [energy, latency, 10]
  
		# src/utils.py - load_energy_latency_data()
		# self.max_container_ips = 4244.198878
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	def run_GOBI(self):
		# 计算cpu
		cpu = [host.getCPU()/100 for host in self.env.hostlist] 
		cpu = np.array([cpu]).transpose()# Hx1
		if 'latency' in self.model.name:
			# 待调度的容器IPS分数
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuC = np.array([cpuC]).transpose() # (C, 1)
			cpu = np.concatenate((cpu, cpuC), axis=1) # 
		alloc = []; prev_alloc = {}
		for c in self.env.containerlist: # c:Task
			oneHot = [0] * len(self.env.hostlist)
			# TODO
			if c: prev_alloc[c.id] = c.getHostID() # 得到之前分配的主机id
			# if c: prev_alloc[c.creationID] = c.getHostID() # 得到之前分配的主机id
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1 # 则在 oneHot 列表中将对应主机的位置标记为1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1 # 0-9取一个1 随机为容器分配一个主机
			alloc.append(oneHot)# (C, H)
		init = np.concatenate((cpu, alloc), axis=1) # (H+C, 1)与(C, H)拼接 (H+C, H)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)

		# 调用优化函数 opt，输入初始状态 init 和模型 self.model，
		# 返回优化后的结果 result、迭代次数 iteration 和适应度 fitness。
		result, iteration, fitness = opt(init, self.model, [], self.data_type)
		decision = []

		# 遍历 prev_alloc 中的每个容器ID：
		# 提取优化结果 result 中对应容器的主机分配情况（one_hot）。
		# 找到 one_hot 中最大值的索引，即新的主机ID。
		# 如果新的主机ID与之前的主机ID不同，则将 (cid, new_host) 添加到决策列表 decision 中。
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_GOBI()
		return decision