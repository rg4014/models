import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
MAX_HASH_SIZE=10000
MAX_HASH_SIZE2=100000


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,batch_size, hidden_size=1024):
        super(Actor, self).__init__()
        self.batch_size=batch_size

        self.embed_bool = nn.Embedding(2, 2)
        self.embed_ratio1 = nn.Linear(1, 8)
        
        self.embed_dis = nn.Embedding(10, 8)
        self.embed_dis2=nn.Embedding(45,18)
        self.embed_dis3=nn.Embedding(3,2)
   
        self.embed_ratio2 = nn.Embedding(20, 8)
        self.embed_num = nn.Linear(1, 18)

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE,18 )
        self.embed_bundle=nn.Embedding(MAX_HASH_SIZE2,18)
        self.embed_num2 = nn.Embedding(40, 18)

        self.pad1= nn.ConstantPad3d((0, 16,0, 0,0,0), 0)
        self.pad2=nn.ConstantPad3d((0, 10,0, 0,0,0), 0)
        self.pad3=nn.ConstantPad3d((0, 0,0, 1,0,0), 0)

        self.conv1 = nn.Conv2d(10, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2688,hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, action_size)
     
    
    def forward(self,bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun):  

        x_bol=self.embed_bool(bol)
        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2)
        x_rat=x_rat
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clamp(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num
        x_dis1=self.embed_dis(dis1)
        x_dis2=self.embed_dis2(dis2)
      
        x_id=self.embed_id(ids)
        x_bun=self.embed_bundle(bun)
        x_dis3=self.embed_dis3(dis3)
        x_2=self.pad1(torch.cat((x_bol,x_dis3),dim=1)).unsqueeze(1)
        x_8 = self.pad2(torch.cat((x_rat,x_dis1), dim=1)).view(self.batch_size,3,13,18)
        x_18=self.pad3(torch.cat((x_dis2,x_id,x_bun,x_num),dim=1)).view(self.batch_size,6,13,18)
        
        x=torch.cat((x_2,x_8,x_18),dim=1)
      


        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
    
    def evaluate(self,bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun):
        logits = self.forward(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action, dist
        
    def get_action(self,bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun):
        logits = self.forward(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size,batch_size, hidden_size=1024, seed=1):
        super(Critic, self).__init__()
        self.batch_size=batch_size

        self.embed_bool = nn.Embedding(2, 2)
        self.embed_ratio1 = nn.Linear(1, 8)
        
        self.embed_dis = nn.Embedding(10, 8)
        self.embed_dis2=nn.Embedding(45,18)
        self.embed_dis3=nn.Embedding(3,2)
   
        self.embed_ratio2 = nn.Embedding(20, 8)
        self.embed_num = nn.Linear(1, 18)

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE,18 )
        self.embed_bundle=nn.Embedding(MAX_HASH_SIZE2,18)
        self.embed_num2 = nn.Embedding(40, 18)
        self.embed_act=nn.Embedding(9,4)

        self.pad1= nn.ConstantPad3d((0, 16,0, 0,0,0), 0)
        self.pad2=nn.ConstantPad3d((0, 10,0, 0,0,0), 0)
        self.pad3=nn.ConstantPad3d((0, 0,0, 1,0,0), 0)

        self.conv1 = nn.Conv2d(10, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(2688+4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self,bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,action):
        x_bol=self.embed_bool(bol)
        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2)
        x_rat=x_rat
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clamp(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num
        x_dis1=self.embed_dis(dis1)
        x_dis2=self.embed_dis2(dis2)
      
        x_id=self.embed_id(ids)
        x_bun=self.embed_bundle(bun)
        x_dis3=self.embed_dis3(dis3)
        x_act=self.embed_act(action).view(self.batch_size,-1)
        x_2=self.pad1(torch.cat((x_bol,x_dis3),dim=1)).unsqueeze(1)
        x_8 = self.pad2(torch.cat((x_rat,x_dis1), dim=1)).view(self.batch_size,3,13,18)
       
        x_18=self.pad3(torch.cat((x_dis2,x_id,x_bun,x_num),dim=1)).view(self.batch_size,6,13,18)
        
        x=torch.cat((x_2,x_8,x_18),dim=1)
      


        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = torch.cat((x,x_act),dim=1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, batch_size,hidden_size=1024):
        super(Value, self).__init__()
        self.batch_size=batch_size

        self.embed_bool = nn.Embedding(2, 2)
        self.embed_ratio1 = nn.Linear(1, 8)
        
        self.embed_dis = nn.Embedding(10, 8)
        self.embed_dis2=nn.Embedding(45,18)
        self.embed_dis3=nn.Embedding(3,2)
   
        self.embed_ratio2 = nn.Embedding(20, 8)
        self.embed_num = nn.Linear(1, 18)

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE,18 )
        self.embed_bundle=nn.Embedding(MAX_HASH_SIZE2,18)
        self.embed_num2 = nn.Embedding(40, 18)

        self.pad1= nn.ConstantPad3d((0, 16,0, 0,0,0), 0)
        self.pad2=nn.ConstantPad3d((0, 10,0, 0,0,0), 0)
        self.pad3=nn.ConstantPad3d((0, 0,0, 1,0,0), 0)

        self.conv1 = nn.Conv2d(10, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2688,hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, 1)

    def forward(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun):
        x_bol=self.embed_bool(bol)
        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2)
        x_rat=x_rat
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clamp(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num
        x_dis1=self.embed_dis(dis1)
        x_dis2=self.embed_dis2(dis2)
      
        x_id=self.embed_id(ids)
        x_bun=self.embed_bundle(bun)
        x_dis3=self.embed_dis3(dis3)
        x_2=self.pad1(torch.cat((x_bol,x_dis3),dim=1)).unsqueeze(1)
        x_8 = self.pad2(torch.cat((x_rat,x_dis1), dim=1)).view(self.batch_size,3,13,18) # 调整维度以适应3D convolution
       
        x_18=self.pad3(torch.cat((x_dis2,x_id,x_bun,x_num),dim=1)).view(self.batch_size,6,13,18)
        
        x=torch.cat((x_2,x_8,x_18),dim=1)
      


        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)