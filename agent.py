import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from cnn import Critic, Actor, Value


class IQL(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 batch_size
                ): 
        super(IQL, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.hard_update_every = 10
        hidden_size = 2048
        learning_rate = 3e-4
        self.batch_size=batch_size
        self.clip_grad_param = 100
        self.temperature = torch.FloatTensor([100]).to(device)
        self.expectile = torch.FloatTensor([0.8]).to(device)
           
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, batch_size,hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size,batch_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size,batch_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size,batch_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, batch_size,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        
        self.value_net = Value(state_size=state_size, batch_size=batch_size,hidden_size=hidden_size).to(device)
        
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.step = 0

    
    def get_action(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, eval=False):
        """Returns actions for given state as per current policy."""
        if eval:
            self.actor_local.eval()
   
        with torch.no_grad():
                action = self.actor_local.get_action(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
        return action.numpy()

    def calc_policy_loss(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, actions):
        with torch.no_grad():
            v = self.value_net(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
            q1 = self.critic1_target(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
            q2 = self.critic2_target(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
            min_Q = torch.min(q1,q2)

        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(bol.device)).squeeze(-1)

        _, dist = self.actor_local.evaluate(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
        
        log_probs = dist.log_prob(actions.squeeze(-1))
        log_probs=log_probs.unsqueeze(-1)
        

        
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss
    
    def calc_value_loss(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions):
        with torch.no_grad():
            q1 = self.critic1_target(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
            q2 = self.critic2_target(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
            min_Q = torch.min(q1,q2)
        
        value = self.value_net(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun)
        value_loss = loss(min_Q - value, self.expectile).mean()
        return value_loss
    
    def calc_q_loss(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, actions, rewards,bol_next,rat_next,rat2_next,num_next,num2_next,id_next,dis1_next,dis2_next,dis3_next,bun_next):
        with torch.no_grad():
            next_v = self.value_net(bol_next,rat_next,rat2_next,num_next,num2_next,id_next,dis1_next,dis2_next,dis3_next,bun_next)
            # q_target = rewards + (self.gamma * next_v) 
            q_target = rewards + next_v

        q1 = self.critic1(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
        q2 = self.critic2(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,actions.long())
        critic1_loss = ((q1 - q_target)**2).mean() 
        critic2_loss = ((q2 - q_target)**2).mean()
        return critic1_loss, critic2_loss


    def learn(self, bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun,bol_next,rat_next,rat2_next,num_next,num2_next,ids_next,dis1_next,dis2_next,dis3_next,bun_next, actions, rewards):
        self.step += 1
           
        
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, actions)
        value_loss.backward()
        self.value_optimizer.step()
     

        actor_loss = self.calc_policy_loss(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic1_loss, critic2_loss = self.calc_q_loss(bol,rat,rat2,num,num2,ids,dis1,dis2,dis3,bun, actions, rewards,bol_next,rat_next,rat2_next,num_next,num2_next,ids_next,dis1_next,dis2_next,dis3_next,bun_next )

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        if self.step % self.hard_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), value_loss.item()
    
    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)