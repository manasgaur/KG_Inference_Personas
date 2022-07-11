import numpy as np
from typing import List, Dict
import json

class ExpWeights(object):
    
    def __init__(self,
                 agent_id = 0,
                 num_arms: int = 0,
                 lr: float = 0.2,
                 window: int = 5,
                 decay: float = 0.9,
                 init: float = 0.0,
                 use_std: bool = True) -> None:
        
        self.agent_id = agent_id
        self.num_arms = num_arms
        self.w1 = {i:init for i in range(2)}
        self.w2 = {i:init for i in range(self.num_arms)}
            #arm1: 0 communicate, 1 not to communicate
        self.arm1 = 0
        self.arms1 = []
        self.arm2 = 0
        self.error_buffer1 = []
        self.error_buffer2 = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std
        
        self.data = []
    
    def softmax(self, x):
        f = np.exp(x - np.max(x))  # shift values
        return f / f.sum(axis=0)
        
    def sample(self) -> float:
        p1 = self.softmax(list(self.w1.values()))
        self.arm1 = np.random.choice(range(0,len(p1)), p=p1)
        self.arms1.append(int(self.arm1 != len(p1)-1)) #communicate
        
        if self.arm1 != len(p1)-1:
            p2 = self.softmax(list(self.w2.values()))
            self.arm2 = np.random.choice(range(0,len(p2)), p=p2)
            return self.arm_to_graph(), int(self.arm1 == 1), int(self.arm2)
        return self.arm_to_graph(), int(self.arm1 == 1), -1
    
    def arm_to_graph(self):
        graph = [0 for i in range(self.num_arms+1)]
        graph[self.agent_id] = 1
        if self.arm1 == 0:
            if  self.arm2 < self.agent_id:
                graph[self.arm2] = 1
            else:
                graph[self.arm2+1] = 1
        return graph


    def update_dists(self, feedback: float, norm: float = 1.0) -> None:
        
        # Since this is non-stationary, subtract mean of previous self.window errors.
        self.error_buffer1.append(feedback)
        self.error_buffer1 = self.error_buffer1[-self.window:]
        self.arms1 = self.arms1[-self.window:]
        # normalize
        feedback1 = feedback - np.mean(self.error_buffer1)
        if self.use_std and len(self.error_buffer1) > 1: norm = np.std(self.error_buffer1);
        feedback1 /= (norm + 1e-4)
        if feedback1>=0:
            communicattion_rate = max(1,np.sum(self.arms1))
            feedback1 = feedback1/communicattion_rate
        else:
            non_communicattion_rate = max(1,np.sum(1-np.array(self.arms1)))
            feedback1 = feedback1/non_communicattion_rate
        feedback1 = 1/(1 + np.exp(-feedback1))
        feedback1 = feedback1*2-1
        # update arm weights
        #self.w1[self.arm1] *= self.decay
        self.w1[self.arm1] += self.lr * (feedback1/max(np.exp(self.w1[self.arm1]), 0.0001))
        
        
        if self.arm1 != 1:
            self.error_buffer2.append(feedback)
            self.error_buffer2 = self.error_buffer2[-self.window:]
            # normalize
            feedback2 = feedback - np.mean(self.error_buffer2)
            if self.use_std and len(self.error_buffer2) > 1: norm = np.std(self.error_buffer2);
            feedback2 /= (norm + 1e-4)

            feedback2 = 1/(1 + np.exp(-feedback2))
            feedback2 = feedback2*2-1
            #self.w2[self.arm2] *= self.decay
            self.w2[self.arm2] += self.lr * (feedback2/max(np.exp(self.w2[self.arm2]), 0.0001))
        self.data.append(feedback)


