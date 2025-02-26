from .Workload import *
from datetime import datetime
from workflow.database.Database import *
from random import gauss, choices
import random
import shutil  

import torch
from torchvision import datasets, transforms
import os
import bz2
import pickle
import _pickle as cPickle

class SPW(Workload):
    def __init__(self, num_workflows, std_dev, database):
        super().__init__()
        self.num_workflows = num_workflows # 1 NEW_CONTAINERS
        self.std_dev = std_dev # 标准差 TODO
        self.db = database # conn
        self.formDatasets() # 形成数据集 存到self.datasets里了
        if os.path.exists('tmp/'): shutil.rmtree('tmp/') # 删除tmp临时数据文件夹
        os.mkdir('tmp/')

    def formDatasets(self):
        self.datasets = {} # dict
        torch.manual_seed(1)
        transform=transforms.Compose([ # 未使用
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]) # 这些值很可能对应于 MNIST 数据集，因为这些是 MNIST 数据集的标准均值和标准差。
        for data_type in ['MNIST', 'FashionMNIST', 'CIFAR100']: # 三种都有
            # 加载 MNIST 数据集并对其进行预处理 存储目录
            dataset = eval("datasets."+data_type+"('workflow/workload/DockerImages/data', train=False, download=True,transform=transform)")
            # dataset = eval("datasets."+data_type+"('workflow/workload/DockerImages/data', train=True, download=True,transform=transform)")
            # train_loader = torch.utils.data.DataLoader(dataset, batch_size=20000, shuffle=True) # 批处理、是否打乱
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True) # 批处理、是否打乱
            self.datasets[data_type] = list(train_loader) # 字典里添加数据集
        
        # 建立tmp目录下的临时输入
    def createWorkflowInput(self, data_type, workflow_id):
        path = 'tmp/'+str(workflow_id)+'/'
        if not os.path.exists(path): os.mkdir(path)
        # data：表示输入数据，例如图像、文本、特征向量等。
        # target：表示与输入数据对应的标签或目标值，例如分类任务中的类别标签、回归任务中的目标值等。
        data, target = random.choice(self.datasets[data_type]) # 数据集都是相同的
        with bz2.BZ2File(path+str(workflow_id)+'_data.pt', 'wb') as f:
            cPickle.dump(data, f)
        with bz2.BZ2File(path+'target.pt', 'wb') as f:
            cPickle.dump(target, f)

    def generateNewWorkflows(self, interval):
        workflowlist = []
        # workflows = ['MNIST', 'FashionMNIST', 'CIFAR100']
        workflows = ['MNIST', 'FashionMNIST', 'MNIST']
        # min_sla, layer_intervals = 2, [7, 9, 14]# 最低服务水平2
        min_sla, layer_intervals = 2, [7, 9, 7]# 最低服务水平2 # TODO
        max_sla = [i + (i - min_sla) for i in layer_intervals]
        max_sla_dict = dict(zip(workflows, max_sla))# {'MNIST':12, 'FashionMNIST': 16, 'CIFAR100': 26}
        minimum_workflows = 1 if interval == 0 else 0
        # max(0, gauss(1, 0.5))
        # 这里决定生成新workflow的概率 均值为1，标准差为0.5---最小为1
        for i in range(max(minimum_workflows,int(gauss(self.num_workflows, self.std_dev)))):
            WorkflowID = self.workflow_id # 从零开始
            workflow = random.choices(workflows, weights=[0.5, 0.25, 0.25])[0] # 选择应用类型
            SLA = np.random.randint(2,max_sla_dict[workflow]) # 从最小2和最大之间选一个
            #                     工作流序号、时隙序号、sla、应用负载
            workflowlist.append((WorkflowID, interval, SLA, workflow))
            self.createWorkflowInput(workflow, WorkflowID) # 0
            self.workflow_id += 1
        return workflowlist

    def generateNewContainers(self, interval, workflowlist, workflowDecision):
        workloadlist = [] # 划分的容器list
        containers = []
        # 遍历需要加入的工作流
        for i, (WorkflowID, interval, SLA, workflow) in enumerate(workflowlist):
            decision = workflowDecision[i] # layer or semantic
            # critical 层划分就是4 语义就是5 TODO 关键
            for split in range(4 if 'layer' in decision else 5):
                CreationID = self.creation_id
                application = 'shreshthtuli/'+workflow.lower()+'_'+decision
                # 层划分依赖上一层
                dependentOn = CreationID - 1 if ('layer' in decision and split > 0) else None 
                workloadlist.append((WorkflowID, CreationID, interval, split, dependentOn, SLA, application))
                self.creation_id += 1
        self.createdContainers += workloadlist # 已创建的container
        self.deployedContainers += [False] * len(workloadlist) # container部署状态
        return self.getUndeployedContainers() # 返回为false的容器们 需要进行调度的