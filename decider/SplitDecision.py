import math
from utils.MathUtils import *
from utils.MathConstants import *
import pandas as pd
from statistics import median
import numpy as np

class SplitDecision():
    def __init__(self):
        self.env = None
        self.choices = ['layer', 'layer'] # TODO 改为每层

    # 添加env属性
    def setEnvironment(self, env):
        self.env = env

    def decision(self, workflowlist):
        pass
        