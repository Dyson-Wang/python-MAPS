import numpy as np
import json
from workflow.server.restClient import *
import subprocess
import requests
import logging
import os
import platform
from metrics.powermodels.PMRaspberryPi import *
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB2ms import *
from metrics.powermodels.PME2asv4 import *
from metrics.powermodels.PME4asv4 import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *
from metrics.powermodels.PMXeon_X5570 import *
from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
from utils.Utils import *

import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

class Datacenter():
    
    def __init__(self, hosts, env, env_type):
        self.num_hosts = len(hosts) # host数量
        self.hosts = hosts # hostArray IP数组
        self.env = env # "VLAN"
        self.env_type = env_type # "Virtual"
        self.types = {'Power' : [1]} # dict

    # 调用测试性能程序
    def parallelizedFunc(self, IP):
        payload = {"opcode": "hostDetails"+self.env_type} # opcode: hostDetailsVirtual
        resp = requests.get("http://"+IP+":8081/request", data=json.dumps(payload))
        data = json.loads(resp.text)
        return data

    # 生成hosts的硬件信息，对每个边缘服务器IP连接执行命令测试性能。
    def generateHosts(self):
        print(color.HEADER+"Obtaining host information and generating hosts"+color.ENDC)
        hosts = []
        # 读取powermodels workflow/config/VLAN_config.json
        with open('workflow/config/'+self.env+'_config.json', "r") as f:
            config = json.load(f)
        powermodels = [server["powermodel"] for server in config[self.env.lower()]['servers']]
        # 
        if self.env_type == 'Virtual':
            with open('workflow/server/scripts/instructions_arch.json') as f:
                arch_dict = json.load(f)
            instructions = arch_dict[platform.machine()] # 20167518615
        outputHostsData = Parallel(n_jobs=num_cores)(delayed(self.parallelizedFunc)(i) for i in self.hosts)
        
        for i, data in enumerate(outputHostsData):
            IP = self.hosts[i]
            logging.error("Host details collected from: {}".format(IP))
            print(color.BOLD+IP+color.ENDC, data)
            IPS = (instructions * config[self.env.lower()]['servers'][i]['cpu'])/(float(data['clock']) * 1000000) if self.env_type == 'Virtual' else data['MIPS']
            Power = eval(powermodels[i]+"()")
            Ram = RAM(data['Total_Memory'], data['Ram_read'], data['Ram_write'])
            Disk_ = Disk(data['Total_Disk'], data['Disk_read'], data['Disk_write'])
            Bw = Bandwidth(data['Bandwidth'], data['Bandwidth'])
            hosts.append((IP, IPS, Ram, Disk_, Bw, Power))
        return hosts
