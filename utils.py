import torch

TOTAL_DICT={'com_fea_bool':['if_accounts_first_role',
 'is_valid',
 'if_pruchase_active_in_7day',
 'role_type_fraud',
 'role_type_no_install_info',
 'role_create_os_ios',
 'is_acc_transfer_from_old',
 'if_pruchase_active_in_30day',
 'is_acc_transfer_to_new',
 'is_role_transfer_from_old',
 'if_pruchase_active_in_3day',
 'if_pruchase_active_in_1day'],
'com_fea_num':['vip_level',
 'money_flow_in_cout_last_week_adjust',
 'bundle_pop_buy_sum_last_week_adjust',
 'bundle_pop_buy_count_last_day',
 'role_total_charge_sum_usd',
 'level',
 'role_purchase_first_now',
 'role_total_refund_sum_usd',
 'bundle_pop_buy_max_price_campaign_stage_trigger',
 'money_flow_in_cout_last_week',
 'bundle_pop_buy_max_last_week_adjust',
 'bundle_pop_count_last_week',
 'bundle_pop_count_last_day',
 'purchase_price_9999_last_week',
 'role_login_secondlast_last',
 'acc_total_charge_sum_usd',
 'purchase_price_max_last_week',
 'purchase_price_sum_last_week',
 'purchase_price_999_last_week',
 'role_last_purchase_price_usd',
 'purchase_price_sum_last_3days',
 'bundle_pop_buy_count_last_week',
 'purchase_price_max',
 'purchase_price_sum_last_day',
 'bundle_pop_buy_count_last_day_campaign_stage_trigger',
 'purchase_price_999_last_week_adjust',
 'purchase_price_max_last_month',
 'bundle_pop_buy_sum_last_week',
 'bundle_pop_count_last_day_campaign_stage_trigger',
 'max_role_level',
 'role_purchase_last_now',
 'purchase_price_1999_last_week',
 'bundle_pop_count_campaign_stage_trigger_9999',
 'bundle_pop_buy_max_last_week',
 'bundle_pop_buy_max_price_campaign_stage_trigger_commax',
 'money_flow_out_cout_last_week_adjust',
 'bundle_pop_count_last_week_adjust',
 'purchase_price_499_last_week_adjust',
 'bundle_pop_buy_sum_last_day',
 'role_first_purchase_price_usd',
 'role_logout_last_now',
 'purchase_price_max_last_day',
 'role_purchase_first_last',
 'money_flow_out_cout_last_day',
 'purchase_price_4999_last_week',
 'money_flow_in_cout_last_day',
 'bundle_pop_buy_count_last_day_tech_unlock_trigger',
 'bundle_pop_count_tech_unlock_trigger_9999',
 'role_total_charge_cnt',
 'bundle_pop_buy_max_price_tech_unlock_trigger_commax',
 'purchase_price_499_last_week',
 'acc_total_refund_sum_usd',
 'role_total_refund_cnt',
 'bundle_pop_buy_count_last_week_adjust',
 'purchase_price_2999_last_week',
 'acc_total_refund_cnt',
 'money_flow_out_cout_last_week',
 'bundle_pop_count_last_day_tech_unlock_trigger',
 'bundle_pop_buy_max_last_day',
 'purchase_price_sum_last_month',
 'purchase_price_max_last_week_adjust','recharge','his30totalpay','base_level','research_building_level','city_level','lifespan','chapter_task'],
'com_fea_ratio':['resource_rarity',
 'speedup_rarity',
 'top_key_rarity',
 'officer_bar_ticket_rarity',
 'common_equipment_exp_consume_levelup_1_diff_num_relatively',
 'common_equipment_exp_consume_blueprint_1_diff_num_relatively',
 'common_equipment_exp_rarity',
 'airforce_equipment_exp_consume_levelup_1_diff_num_relatively',
 'airforce_equipment_exp_consume_blueprint_1_diff_num_relatively',
 'airforce_equipment_exp_rarity',
 'ammunition_consume_levelup_1_diff_num_relatively',
 'ammunition_consume_levelup_10_diff_num_relatively',
 'ammunition_consume_blueprint_1_diff_num_relatively',
 'ammunition_consume_blueprint_10_diff_num_relatively',
 'ammunition_rarity',
 'airforce_ammunition_consume_levelup_1_diff_num_relatively',
 'airforce_ammunition_consume_levelup_10_diff_num_relatively',
 'airforce_ammunition_consume_blueprint_1_diff_num_relatively',
 'airforce_ammunition_consume_blueprint_10_diff_num_relatively',
 'airforce_ammunition_rarity',
 'elements_consume_blueprint_1_diff_num_relatively',
 'elements_consume_blueprint_10_diff_num_relatively',
 'elements_consume_reform_diff_num_relatively',
 'elements_rarity',
 'airforce_elements_consume_blueprint_1_diff_num_relatively',
 'airforce_elements_consume_blueprint_10_diff_num_relatively',
 'airforce_elements_consume_reform_diff_num_relatively',
 'airforce_elements_rarity',
 'replacement_wrench_consume_mechanic_1_diff_num_relatively',
 'replacement_wrench_rarity',
 'officer_exp_consume_lvlup_1_diff_num_relatively',
 'officer_exp_rarity',
 'component_consume_diff_num_relatively',
 'chest_choice_replacement_rarity',
 'buy_rate_last_week',
 'buy_rate_last_day_campaign_stage_trigger',
 'buy_rate_last_day'],
'id':['user_id','open_id','serverid','event_time','dt','role_type_'],
'dis_num':['bundle_type','his30maxpay','his30minpay']}


def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()









import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
MAX_HASH_SIZE=10000


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,batch_size, hidden_size=32):
        super(Actor, self).__init__()
        self.batch_size=batch_size
        self.embed_bool = nn.Embedding(2, 1)
        self.embed_ratio1 = nn.Linear(1, 5)
        
        self.embed_dis = nn.Embedding(10, 1)
        self.embed_dis2=nn.Embedding(44,5)
   
        self.embed_ratio2 = nn.Embedding(20, 5)
        self.embed_num = nn.Linear(1, 10)  

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE, 3)
        self.embed_num2 = nn.Embedding(40, 10)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
     
    
    def forward(self,bol,rat,rat2,num,num2,id,dis1,dis2):  

        x_bol=self.embed_bool(bol).view(self.batch_size,-1)
        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2)
        x_rat=x_rat.view(self.batch_size,-1)
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clip(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num.view(self.batch_size,-1)
        x_dis1=self.embed_dis(dis1).view(self.batch_size,-1)
        x_dis2=self.embed_dis2(dis2).view(self.batch_size,-1)
        x_id=self.embed_id(id).view(self.batch_size,-1)
        x = torch.cat((x_bol,x_num,x_rat,x_dis1,x_dis2,x_id), dim=-1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
    
    def evaluate(self,bol,rat,rat2,num,num2,id,dis1,dis2):
        logits = self.forward(bol,rat,rat2,num,num2,id,dis1,dis2)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action, dist
        
    def get_action(self,bol,rat,rat2,num,num2,id,dis1,dis2):
        logits = self.forward(bol,rat,rat2,num,num2,id,dis1,dis2)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size,batch_size, hidden_size=32, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.batch_size=batch_size

        self.embed_bool = nn.Embedding(2, 1)
        self.embed_ratio1 = nn.Linear(1, 5)
        
        self.embed_dis = nn.Embedding(10, 1)
        self.embed_dis2=nn.Embedding(44,5)
   
        self.embed_ratio2 = nn.Embedding(20, 5)
        self.embed_num = nn.Linear(1, 10)  

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE, 3)
        self.embed_num2 = nn.Embedding(40, 10)
        self.embed_act=nn.Embedding(9,1)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self,bol,rat,rat2,num,num2,id,dis1,dis2,action):
        x_bol=self.embed_bool(bol).view(self.batch_size,-1)

        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2)
        x_rat=x_rat.view(self.batch_size,-1)
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clip(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num.view(self.batch_size,-1)
        x_dis1=self.embed_dis(dis1).view(self.batch_size,-1)
        x_dis2=self.embed_dis2(dis2).view(self.batch_size,-1)
        x_id=self.embed_id(id).view(self.batch_size,-1)
        x_act=self.embed_act(action).view(self.batch_size,-1)
        x = torch.cat((x_bol,x_num,x_rat,x_dis1,x_dis2,x_id,x_act), dim=-1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, batch_size,hidden_size=32):
        super(Value, self).__init__()
        self.batch_size=batch_size

        self.embed_bool = nn.Embedding(2, 1)
        self.embed_ratio1 = nn.Linear(1, 5)
        
        self.embed_dis = nn.Embedding(10, 1)
        self.embed_dis2=nn.Embedding(44,5)
   
        self.embed_ratio2 = nn.Embedding(20, 5)
        self.embed_num = nn.Linear(1, 10)  

   
        self.embed_id = nn.Embedding(MAX_HASH_SIZE, 3)
        self.embed_num2 = nn.Embedding(40, 10)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, bol,rat,rat2,num,num2,id,dis1,dis2):
        x_bol=self.embed_bool(bol).view(self.batch_size,-1)
        x_rat=self.embed_ratio1(rat)+self.embed_ratio2(rat2).view(self.batch_size,-1)
        x_rat=x_rat.view(self.batch_size,-1)
        x_num=nn.functional.leaky_relu(self.embed_num(num))
        x_num=torch.clip(x_num, min=-1.0, max=1.0)+self.embed_num2(num2)
        x_num=x_num.view(self.batch_size,-1)
        x_dis1=self.embed_dis(dis1).view(self.batch_size,-1)
        x_dis2=self.embed_dis2(dis2).view(self.batch_size,-1)
        x_id=self.embed_id(id).view(self.batch_size,-1)
        x = torch.cat((x_bol,x_num,x_rat,x_dis1,x_dis2,x_id), dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)