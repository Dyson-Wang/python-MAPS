import numpy as np

class Workload():
	def __init__(self):
		self.workflow_id = 0
		self.creation_id = 0
		self.createdContainers = [] # 已创建容器
		self.deployedContainers = [] # 已部署容器 true为已部署 false为未部署

	def getUndeployedContainers(self):
		undeployed = []
		for i,deployed in enumerate(self.deployedContainers):
			if not deployed:
				undeployed.append(self.createdContainers[i])
		return undeployed

	# 更新已部署的容器(不会迁移)
	def updateDeployedContainers(self, creationIDs):
		for cid in creationIDs:
			assert not self.deployedContainers[cid]
			self.deployedContainers[cid] = True
