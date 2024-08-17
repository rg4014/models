import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['bol','id','dis','dis2','dis3','rat','rat2','num','num2','bun','bol_next','id_next','dis_next',
                                                                'dis2_next','dis3_next','rat_next','rat2_next','num_next','num2_next','bun_next','action',"reward" ])
    
    def add(self, condition):
        """Add a new experience to memory."""
        tmp=(i[0] for i in condition)
        bol,id,bun,dis,dis2,dis3,rat,rat2,num,num2,bol_next,id_next,bun_next,dis_next,dis2_next,dis3_next,rat_next,rat2_next,num_next,num2_next,actions, rewards=tmp

        e = self.experience(bol,id,dis,dis2,dis3,rat,rat2,num,num2,bun,bol_next,id_next,dis_next,dis2_next,dis3_next,rat_next,rat2_next,num_next,num2_next,bun_next, actions, rewards)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size-1)
        e_tmp=self.memory[-1]
        experiences.append(e_tmp)

        bol = torch.stack([e.bol for e in experiences if e is not None]).to(self.device)
        id= torch.stack([e.id for e in experiences if e is not None]).to(self.device)
        dis= torch.stack([e.dis for e in experiences if e is not None]).to(self.device)
        dis2= torch.stack([e.dis2 for e in experiences if e is not None]).to(self.device)
        dis3= torch.stack([e.dis3 for e in experiences if e is not None]).to(self.device)
        rat= torch.stack([e.rat for e in experiences if e is not None]).to(self.device)
        rat2= torch.stack([e.rat2 for e in experiences if e is not None]).to(self.device)
        num= torch.stack([e.num for e in experiences if e is not None]).to(self.device)
        num2=torch.stack([e.num2 for e in experiences if e is not None]).to(self.device)
        bun= torch.stack([e.bun for e in experiences if e is not None]).to(self.device)
        bol_next= torch.stack([e.bol_next for e in experiences if e is not None]).to(self.device)
        id_next= torch.stack([e.id_next for e in experiences if e is not None]).to(self.device)
        dis_next= torch.stack([e.dis_next for e in experiences if e is not None]).to(self.device)
        dis2_next= torch.stack([e.dis2_next for e in experiences if e is not None]).to(self.device)
        dis3_next= torch.stack([e.dis3_next for e in experiences if e is not None]).to(self.device)
        rat_next= torch.stack([e.rat_next for e in experiences if e is not None]).to(self.device)
        rat2_next= torch.stack([e.rat2_next for e in experiences if e is not None]).to(self.device)
        num_next= torch.stack([e.num_next for e in experiences if e is not None]).to(self.device)
        num2_next=torch.stack([e.num2_next for e in experiences if e is not None]).to(self.device)
        bun_next= torch.stack([e.bun_next for e in experiences if e is not None]).to(self.device)
        actions = torch.stack([e.action for e in experiences if e is not None]).to(self.device)
        rewards = torch.stack([e.reward for e in experiences if e is not None]).to(self.device)

        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (bol,id,dis,dis2,dis3,rat,rat2,num,num2,bun,bol_next,id_next,dis_next,dis2_next,dis3_next,rat_next,rat2_next,num_next,num2_next,bun_next, actions, rewards)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
