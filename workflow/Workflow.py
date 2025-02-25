from workflow.node.Node import *
from workflow.task.Task import *
from workflow.server.controller import *
from time import time, sleep
from pdb import set_trace as bp
import multiprocessing
from joblib import Parallel, delayed
import shutil
from copy import deepcopy
import torch
import torch.nn.functional as F
from pprint import pprint
from utils.ColorUtils import *
import bz2
import pickle
import _pickle as cPickle

num_cores = multiprocessing.cpu_count()

class Workflow():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, Scheduler, Decider, ContainerLimit, IntervalTime, hostinit, database, env, logger):
		self.hostlimit = len(hostinit) # 主机上限10
		self.scheduler = Scheduler # Scheduler
		self.scheduler.setEnvironment(self) # Scheduler.env=self
		self.decider = Decider # Decider
		self.decider.setEnvironment(self) # Decider.env=self
		self.containerlimit = ContainerLimit # 容器上限10
		self.hostlist = [] # 主机列表
		self.containerlist = [] # 容器列表
		self.intervaltime = IntervalTime # 时隙大小
		self.interval = 0 # 默认轮次为0
		self.db = database # conn
		self.inactiveContainers = [] # 非活跃容器
		self.destroyedccids = set() # 已经删除的容器
		self.activeworkflows = {} # 激活的工作流
		self.destroyedworkflows = {} # 完成的工作流
		self.logger = logger # logger
		self.stats = None # 
		self.environment = env # VLAN
		self.controller = RequestHandler(self.db, self) # server/controller
		self.addHostlistInit(hostinit) # hosts详细信息
		self.globalStartTime = time() # 启动时间
		self.intervalAllocTimings = [] # ?
	
	def addHostInit(self, IP, IPS, RAM, Disk, Bw, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		# 不执行？
		host = Node(len(self.hostlist),IP,IPS, RAM, Disk, Bw, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IP, IPS, RAM, Disk, Bw, Powermodel in hostList:
			self.addHostInit(IP, IPS, RAM, Disk, Bw, Powermodel)

	# callby addContainerListInit
	def addContainerInit(self, WorkflowID, CreationID, interval, split, dependentOn, SLA, application):
		
		container = Task(len(self.containerlist), WorkflowID, CreationID, interval, split, dependentOn, SLA, application, self, HostID = -1)
		self.containerlist.append(container)
		return container

	# callby addContainersInit
	def addContainerListInit(self, containerInfoList):
		# 目前最大能部署的容器数
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		deployedContainers = []
		for WorkflowID, CreationID, interval, split, dependentOn, SLA, application in containerInfoList:
			if dependentOn is None or dependentOn in self.destroyedccids: # 可以直接执行的
				dep = self.addContainerInit(WorkflowID, CreationID, interval, split, dependentOn, SLA, application)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break # 按顺序部署
		# 初始化十个none
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return [container.id for container in deployedContainers]

	def addContainer(self, WorkflowID, CreationID, interval, split, dependentOn, SLA, application):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Task(i, WorkflowID, CreationID, interval, split, dependentOn, SLA, application, self, HostID = -1)
				self.containerlist[i] = container
				return container

	def addContainerList(self, containerInfoList): # 添加真正可部署容器位置返回新可调度容器列表，之后调度
		# 还剩几个可部署容器位置
		maxdeploy = min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())
		if maxdeploy == 0: return []
		deployedContainers = []
		# 顺序访问 可优化
		for WorkflowID, CreationID, interval, split, dependentOn, SLA, application in containerInfoList:
			if dependentOn is None or dependentOn in self.destroyedccids:
				dep = self.addContainer(WorkflowID, CreationID, interval, split, dependentOn, SLA, application)
				deployedContainers.append(dep)
				if len(deployedContainers) >= maxdeploy: break
		return [container.id for container in deployedContainers]

	def getContainersOfHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container and container.hostid == hostID:
				containers.append(container.id)
		return containers

	def getContainerByID(self, containerID):
		return self.containerlist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.containerlist + self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getInactiveContainerByCID(self, creationID):
		for c in self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getCreationIDs(self, migrations, containerIDs):
		creationIDs = []
		for decision in migrations:
			if decision[0] in containerIDs: creationIDs.append(self.containerlist[decision[0]].creationID)
		return creationIDs

	def addWorkflows(self, containerInfoList): # 加入未激活工作流详细信息
		for WorkflowID, CreationID, interval, _, _, SLA, application in containerInfoList:
			if WorkflowID not in self.activeworkflows:# 未激活的工作流
				self.activeworkflows[WorkflowID] = {'ccids': [CreationID], \
					'createAt': interval, \
					'sla': SLA, \
					'startAt': -1, \
					'application': application}
			elif CreationID not in self.activeworkflows[WorkflowID]['ccids']:
				self.activeworkflows[WorkflowID]['ccids'].append(CreationID)
		print(color.YELLOW); pprint(self.activeworkflows); print(color.ENDC)

	def getPlacementPossible(self, containerID, hostID): # host部署可能性
		container = self.containerlist[containerID]
		host = self.hostlist[hostID]
		ipsreq = container.getBaseIPS()
		ramsizereq, _, _ = container.getRAM()
		disksizereq, _, _ = container.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
		return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				disksizereq <= disksizeav)

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		deployed = self.addContainerListInit(containerInfoListInit)
		return deployed

	def allocateInit(self, decision): # 调度容器
		start = time()
		migrations = []
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1 and hid != -1
			if self.getPlacementPossible(cid, hid): # 查看部署可能
				migrations.append((cid, hid))
				container.allocateAndExecute(hid)
				ram_usage, _, _ = container.getRAM()
				if self.activeworkflows[container.workflowID]['startAt'] == -1:
					self.activeworkflows[container.workflowID]['startAt'] = self.interval
				# Update RAM usages for getPlacementPossible()
				self.getHostByID(hid).ram.size += ram_usage
			# destroy pointer to this unallocated container as book-keeping is done by workload model
			else: 
				self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		self.logger.debug("First allocation: "+str(decision))
		self.logger.debug('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		# 分配到主机所花费的时间
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		# 睡眠时间
		self.visualSleep(self.intervaltime - self.intervalAllocTimings[-1])
		for host in self.hostlist:
			host.updateUtilizationMetrics()
		return migrations

	def checkWorkflowOutput(self, WorkflowID):
		wid = str(WorkflowID)
		if 'layer' in self.activeworkflows[WorkflowID]['application']:
			with bz2.BZ2File('tmp/'+wid+'/'+wid+'_3', 'rb') as f:
				output = cPickle.load(f)
		else:
			outputs = []
			for split in range(5):
				with bz2.BZ2File('tmp/'+wid+'/'+wid+'_'+str(split), 'rb') as f:
					outputs.append(cPickle.load(f))
			output = F.log_softmax(torch.cat(outputs, dim=1), dim=1)
		with bz2.BZ2File('tmp/'+str(WorkflowID)+'/target.pt', 'rb') as f:
			target = cPickle.load(f)
		pred = output.argmax(dim=1, keepdim=True)
		correct = pred.eq(target.view_as(pred)).sum().item()
		total = int(target.shape[0])
		return correct, total

	def destroyCompletedWorkflows(self):
		toDelete = []
		for WorkflowID in self.activeworkflows:
			allDestroyed = True
			for ccid in self.activeworkflows[WorkflowID]['ccids']:
				if ccid not in self.destroyedccids:
					allDestroyed = False
			if allDestroyed:
				correct, total = self.checkWorkflowOutput(WorkflowID)
				shutil.rmtree('tmp/'+str(WorkflowID)+'/')
				self.destroyedworkflows[WorkflowID] = deepcopy(self.activeworkflows[WorkflowID])
				self.destroyedworkflows[WorkflowID]['sla'] = self.activeworkflows[WorkflowID]['sla']
				self.destroyedworkflows[WorkflowID]['destroyAt'] = self.interval
				self.destroyedworkflows[WorkflowID]['result'] = (correct, total)
				print(color.GREEN); print("Workflow ID: ", WorkflowID)
				pprint(self.destroyedworkflows[WorkflowID]); print(color.ENDC)
				toDelete.append(WorkflowID)
		for WorkflowID in toDelete:
			del self.activeworkflows[WorkflowID]

	def parallelizedDestroy(self, cid):
		container = self.getInactiveContainerByCID(cid)
		container.destroy()

	def destroyCompletedContainers(self):
		destroyed, toDestroy = [], []
		for i, container in enumerate(self.containerlist):
			if container and not container.active:
				toDestroy.append(container.creationID)
				self.destroyedccids.add(container.creationID)
				self.containerlist[i] = None 
				self.inactiveContainers.append(container)
				destroyed.append(container)
		Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelizedDestroy)(i) for i in toDestroy)
		self.destroyCompletedWorkflows()
		return destroyed

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container and container.active: num += 1
		return num

	def getSelectableContainers(self):
		selectable = []
		selected = []
		containers = self.db.select("SELECT * FROM CreatedContainers;")
		for container in self.containerlist:
			if container and container.active and container.getHostID() != -1:
				selectable.append(container.id)
		print(selectable)
		return selectable

	def addContainers(self, newContainerList): # 删除已经完成的容器，添加可新部署的容器
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(newContainerList)
		return deployed, destroyed

	def getActiveContainerList(self):
		return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

	def getContainersInHosts(self):
		return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

	def parallelizedFunc(self, i):
		cid, hid = i
		container = self.getContainerByID(cid)
		if self.containerlist[cid].hostid != -1:
			# Disabled container migration
			# container.allocateAndrestore(hid)
			return container
		else:
			if self.activeworkflows[container.workflowID]['startAt'] == -1:
				self.activeworkflows[container.workflowID]['startAt'] = self.interval
			container.allocateAndExecute(hid)
		return container

	def visualSleep(self, t):
		total = str(int(t//60))+" min, "+str(t%60)+" sec"
		for i in range(int(t)):
			print("\r>> Interval timer "+str(i//60)+" min, "+str(i%60)+" sec of "+total, end=' ')
			sleep(1)
		sleep(t % 1)
		print()

	def simulationStep(self, decision): # 
		start = time()
		migrations = []
		containerIDsAllocated = []
		for (cid, hid) in decision:
			# 得到cid hid对象
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			# 部署可行性
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				containerIDsAllocated.append(cid)
				migrations.append((cid, hid))
				# Update RAM usages for getPlacementPossible()
				container = self.getContainerByID(cid)
				ram_usage, _, _ = container.getRAM()
				if container.hostid != -1: # 已有的(迁移)减去利用百分比*0.01
					self.getHostByID(container.hostid).ram.size -= ram_usage
				self.getHostByID(hid).ram.size += ram_usage
		Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelizedFunc)(i) for i in migrations)
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
		self.intervalAllocTimings.append(time() - start)
		self.logger.debug("Decision: "+str(decision))
		self.logger.debug('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		print('Interval allocation time for interval '+str(self.interval)+' is '+str(self.intervalAllocTimings[-1]))
		self.visualSleep(max(0, self.intervaltime - self.intervalAllocTimings[-1]))
		for host in self.hostlist:
			host.updateUtilizationMetrics()
		return migrations
