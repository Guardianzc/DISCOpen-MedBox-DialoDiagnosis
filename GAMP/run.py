# -*- coding:utf-8 -*-
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
from agent import Agent
from running_steward import RunningSteward
os.chdir(os.path.dirname(sys.argv[0]))
parser = argparse.ArgumentParser()
#file0='./Data/new_data/mz10/'
#file0='./Data/dxy_dataset/dxy_dataset/'
file0='./Data/Fudan-Medical-Dialogue2.0/synthetic_dataset/'

parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
parser.add_argument("--disease_set", dest="disease_set", type=str, default=file0+'/disease_set.p',help='path and filename of the disease set')

parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
parser.add_argument("--goal_set_test", dest="goal_set_test", type=str, default=file0+'/goal_test_set.p',help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")

parser.add_argument("--train_mode", dest="train_mode", type=bool, default=True, help="Runing this code in training mode? [True, False]")
parser.add_argument("--load_old_model", dest="load", type=bool, default=False)
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=1000, help="The number of simulate epoch.")
parser.add_argument("--model_savepath", dest="model_savepath", type=str, default='./model_save/simulate/', help="The path for save model.")
parser.add_argument("--load_path", dest="load_path", type=str, default='/remote-home/czhong/RL/GAMP/model_save/dxy/0113112853/s0.644_obj-1.611_t2.394_mr0.213_outs0.125_e-129', help="The path for save model.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="The batchsize.")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=10, help="The maxturn.")


parser.add_argument("--wrong_prediction_reward", dest="n", type=int, default=-1)

parser.add_argument("--reward_shaping", dest="phi", type=int, default=0.25)

parser.add_argument("--Correct_prediction_reward", dest="m", type=int, default=1)
parser.add_argument("--reward_for_reach_max_turn", dest="out", type=int, default=-1)

parser.add_argument("--rebulid_factor", dest="beta", type=int, default=10)
parser.add_argument("--entropy_factor", dest="yita", type=int, default=0.007)
parser.add_argument("--discount_factor", dest="gamma", type=int, default=0.99)

parser.add_argument("--cuda_idx", dest="cuda_idx", type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
args = parser.parse_args()
parameter = vars(args)

def run(parameter, wandb):
    time.sleep(2)
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    disease_set = pickle.load(file=open(parameter["disease_set"], "rb"))
    train_mode = parameter.get("train_mode")
    simulate_epoch_number = parameter.get("simulate_epoch_number")
    goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
    agent = Agent(slot_set, disease_set, goal_set, parameter)


    if train_mode:
        
        agent.warm_start()
        best_success_rate_test = agent.train(simulate_epoch_number, wandb)
        print('SC = ', best_success_rate_test)
        
    else:
        agent.load(parameter['load_path'])
        #agent.load(parameter['model_savepath'] )
        success_rate_test, avg_turns_test, avg_object_test, hits, outs = agent.simulation_epoch(mode = 'test', epoch = 0, simulate_epoch_number = 1)
        print(success_rate_test, avg_turns_test, avg_object_test, hits, outs)

if __name__ == '__main__':
    '''
    wandb.init(project='GAMP-test', entity='guardian_zc')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = 0.01
    
    #print(params['run_info'])
    run(parameter=parameter, wandb = wandb)
    '''
    run(parameter=parameter, wandb = 0)