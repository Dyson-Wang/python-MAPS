import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from time import time
from subprocess import call
from os import system, rename

# Workflow imports
# 工作流、数据库、数据中心、负载
from workflow.Workflow import *
from workflow.database.Database import *
from workflow.datacenter.Datacenter_Setup import *
from workflow.datacenter.Datacenter import *
from workflow.workload.SplitPlaceWorkload import *

# Splitnet decision imports
from decider.Random import RandomDecider
from decider.Layer_Only import LayerOnlyDecider
from decider.Semantic_Only import SemanticOnlyDecider
from decider.MABDecider import MABDecider

# Scheduler imports
from scheduler.MAD_MC_Random import MADMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.POND import PONDScheduler
from scheduler.GOBI import GOBIScheduler
from scheduler.GOBI2 import GOBI2Scheduler
from scheduler.DAGOBI import DAGOBIScheduler
from scheduler.DAGOBI2 import DAGOBI2Scheduler

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp

usage = "usage: python main.py -e <environment> -m <mode> # empty environment run simulator"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-e", "--environment", action="store", dest="env", default="", 
					help="Environment is AWS, Openstack, Azure, VLAN, Vagrant")
parser.add_option("-m", "--mode", action="store", dest="mode", default="0", 
					help="Mode is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
opts, args = parser.parse_args()

# Global constants
NUM_SIM_STEPS = 100 # 100轮
HOSTS = 10 # 主机数
# CONTAINERS = HOSTS # 容器上限
CONTAINERS = 10 # 10个容器
TOTAL_POWER = 1000 # 
ROUTER_BW = 1000 # 
INTERVAL_TIME = 300 # 5分钟
NEW_CONTAINERS = 1 if HOSTS == 10 else 5
# NEW_CONTAINERS = 5 # 每轮新容器工作流
DB_NAME = ''
DB_HOST = ''
DB_PORT = 0
HOSTS_IP = []
logFile = 'SplitPlace.log'

if len(sys.argv) > 1:
	with open(logFile, 'w'): os.utime(logFile, None)

def initalizeEnvironment(environment, logger):
	# Initialize the db 初始化数据库对象
	db = Database(DB_NAME, DB_HOST, DB_PORT) # 返回的conn，数据库连接对象

	# Initialize simple fog datacenter 初始化简单边缘数据中心
	''' Can be Datacenter '''
	datacenter = Datacenter(HOSTS_IP, environment, 'Virtual')

	# Initialize workload 初始化工作负载
	''' Can be SPW '''
	workload = SPW(NEW_CONTAINERS, 0.5, db)

	# Initialize splitnet decision moduele 初始化分割决策
	''' Can be Random, SemanticOnly, LayerOnly '''
	decider = MABDecider()
	
	# Initialize scheduler 初始化调度器
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
	scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
	# scheduler = RandomScheduler()

	# Initialize Environment
	hostlist = datacenter.generateHosts()
	env = Workflow(scheduler, decider, CONTAINERS, INTERVAL_TIME, hostlist, db, environment, logger)

	# Execute first step
	# 生成一个新的工作流。包含WorkflowID,interval,sla,workflow
	newworkflowinfos = workload.generateNewWorkflows(env.interval)
	# 根据MAB模型生成decision, layer or semantic
	workflowsplits = decider.decision(newworkflowinfos)
	# 返回目前已有的未部署的容器
	newcontainerinfos = workload.generateNewContainers(env.interval, newworkflowinfos, workflowsplits) # New containers info
	env.addWorkflows(newcontainerinfos) # 添加未激活的工作流

	# 得到可以直接部署的容器
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	
	start = time()
	decision = scheduler.placement(deployed) # Decide placement using container ids
	schedulingTime = time() - start

	print("Decision:", color.BLUE+str(decision)+color.ENDC)

	# decision ※ waiting timer
	migrations = env.allocateInit(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	
	# 已经真实调度部署的容器creation id
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	# 已经真实调度的主机部署情况
	print("Containers in host:", env.getContainersInHosts())
	# 已经真实调度的容器部署情况
	print("Schedule:", env.getActiveContainerList())
	printDecisionAndMigrations(decision, migrations)

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
	return datacenter, workload, scheduler, decider, env, stats

def stepSimulation(workload, scheduler, decider, env, stats):
	# Execute first step
	# 生成一个新的工作流。包含WorkflowID,interval,sla,workflow
	newworkflowinfos = workload.generateNewWorkflows(env.interval)
	# 根据MAB模型生成decision, layer or semantic
	workflowsplits = decider.decision(newworkflowinfos)
	# 返回目前已有的未部署的容器
	newcontainerinfos = workload.generateNewContainers(env.interval, newworkflowinfos, workflowsplits) # New containers info
	# 打印新容器 格式：(WorkflowID, CreationID, interval, split, dependentOn, SLA, application)
	if opts.env != '': print(newcontainerinfos)

	# 加入未激活工作流详细信息
	env.addWorkflows(newcontainerinfos)

	# 删除已经完成的容器，添加可新部署的容器
	deployed, destroyed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	
	# timer schedulingTime
	start = time()
	############## Disabled Migration 禁止迁移容器 ############## 
	selected = [] # scheduler.selection() # Select container IDs for migration
	# 返回[(cid, hid), ...]
	decision = scheduler.filter_placement(scheduler.placement(selected+deployed)) # Decide placement for selected container ids
	schedulingTime = time() - start

	# 打印调度决策、调度容器、部署情况、分配情况
	print("Decision:", color.BLUE+str(decision)+color.ENDC)
	# 开始模拟
	migrations = env.simulationStep(decision) # Schedule containers ※
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload deployed using creation IDs
	# pr
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
	print("Containers in host:", env.getContainersInHosts())
	print("Num active containers:", env.getNumActiveContainers())
	print("Host allocation:", [(c.getHostID() if c else -1)for c in env.containerlist])
	printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)

def saveStats(stats, datacenter, workload, env, end=True):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(TOTAL_POWER)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	stats.generateDatasets(dirname)
	if 'Datacenter' in datacenter.__class__.__name__:
		saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler = stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = None, None, None, None, None
		with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
			pickle.dump(stats, handle)
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler
	if not end: return

	# this
	stats.generateGraphs(dirname)
	stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	if 'Datacenter' in datacenter.__class__.__name__:
		stats.simulated_scheduler = None
		logger.getLogger().handlers.clear(); env.logger.getLogger().handlers.clear()
		if os.path.exists(dirname+'/'+logFile): os.remove(dirname+'/'+logFile)
		rename(logFile, dirname+'/'+logFile)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':
	env, mode = opts.env, int(opts.mode)

	if env != '':
		# 将所有agent文件转换为 unix 格式
		unixify(['workflow/agent/', 'workflow/agent/scripts/'])

		# Start InfluxDB service
		print(color.HEADER+'InfluxDB service runs as a separate front-end window. Please minimize this window.'+color.ENDC)
		if 'Windows' in platform.system():
			os.startfile('C:/Program Files/InfluxDB/influxdb-1.8.3-1/influxd.exe')

		# workflow/config/VLAN_config.json
		configFile = 'workflow/config/' + opts.env + '_config.json'
	    
		logger.basicConfig(filename=logFile, level=logger.DEBUG,
	                        format='%(asctime)s - %(levelname)s - %(message)s')
		logger.debug("Creating enviornment in :{}".format(env))
		cfg = {}
		with open(configFile, "r") as f:
			cfg = json.load(f)
		DB_HOST = cfg['database']['ip']
		DB_PORT = cfg['database']['port']
		DB_NAME = 'COSCO'

		if env == 'Vagrant':
			print("Setting up VirtualBox environment using Vagrant")
			HOSTS_IP = setupVagrantEnvironment(configFile, mode)
			print(HOSTS_IP)
		elif env == 'VLAN':
			print("Setting up VLAN environment using Ansible")
			HOSTS_IP = setupVLANEnvironment(configFile, mode)
			print(HOSTS_IP)
		# exit()

	# 初始化环境传入："VLAN"、logger组件
	# 六个对象
	datacenter, workload, scheduler, decider, env, stats = initalizeEnvironment(env, logger)

	for step in range(NUM_SIM_STEPS):
		print(color.BOLD+"Simulation Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, decider, env, stats)
		if env != '' and step % 10 == 0: saveStats(stats, datacenter, workload, env, end = False)

	if opts.env != '':
		# Destroy environment if required
		eval('destroy'+opts.env+'Environment(configFile, mode)')

		# Quit InfluxDB
		if 'Windows' in platform.system():
			os.system('taskkill /f /im influxd.exe')

	saveStats(stats, datacenter, workload, env)

