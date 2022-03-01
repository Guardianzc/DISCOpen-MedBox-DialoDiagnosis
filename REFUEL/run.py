# -*- coding:utf-8 -*-
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
from agent import Agent
os.chdir(os.path.dirname(sys.argv[0]))
parser = argparse.ArgumentParser()
#file0='./data/dxy_dataset/dxy_dataset/100symptoms//'
#file0 = './data/data/dataset/label/'
#file0='./data/new_data/mz10/allsymptoms/'
file0='./data/Fudan-Medical-Dialogue2.0/synthetic_dataset//allsymptoms/'
parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
parser.add_argument("--disease_set", dest="disease_set", type=str, default=file0+'/disease_set.p',help='path and filename of the disease set')

parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
parser.add_argument("--goal_set_test", dest="goal_set_test", type=str, default=file0+'/goal_test_set.p',help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")
parser.add_argument("--disease_test", dest="disease_test", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")

parser.add_argument("--train_mode", dest="train_mode", type=bool, default=True, help="Runing this code in training mode? [True, False]")
parser.add_argument("--load_old_model", dest="load", type=bool, default=True)
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=5000, help="The number of simulate epoch.")
parser.add_argument("--model_savepath", dest="model_savepath", type=str, default='./model_save/simulate/', help="The path for save model.")
parser.add_argument("--model_loadpath", dest="model_loadpath", type=str, default='/remote-home/czhong/RL/REFUEL/model_save/simulate/0204115245/s0.343_obj-0.137_t3.503_mr0.118_outs0.026_e-3312.pkl', help="The path for save model.")
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
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
args = parser.parse_args()
parameter = vars(args)

def run(parameter, wandb):
    time.sleep(2)
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    disease_set = pickle.load(file=open(parameter["disease_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    disease_test = pickle.load(file=open(parameter["disease_test"], "rb"))
    train_mode = parameter.get("train_mode")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    agent = Agent(slot_set, disease_set, parameter)


    if train_mode:
        best_success_rate_test = agent.train(simulate_epoch_number, wandb)
        print('SC = ', best_success_rate_test)
        
    else:
        #agent.load(parameter['model_savepath'] + '/newest/')
        agent.load(parameter['model_loadpath'] )
        success_rate_test, avg_turns_test, avg_object_test, avg_recall, avg_out = agent.simulation_epoch(mode = 'test', epoch = 0, simulate_epoch_number = 1)
        # self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out
        print(success_rate_test, avg_turns_test, avg_object_test, avg_recall, avg_out)

if __name__ == '__main__':
    '''
    wandb.init(project='REFUEL-test', entity='guardian_zc')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = 0.01
    
    #print(params['run_info'])
    run(parameter=parameter, wandb = wandb)
    '''
    run(parameter=parameter, wandb = 0)
    