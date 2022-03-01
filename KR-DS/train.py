from timeit import default_timer as timer
from datetime import timedelta
from utils.utils import *
import math
from usersim.usersim_test import TestRuleSimulator      
from usersim.usersim_rule import RuleSimulator
# from utils.wrappers import *
from agents.agent import AgentDQN
from dialog_system.dialog_manager import DialogManager
from tensorboardX import SummaryWriter
import argparse, json, copy
import dialog_config
import torch
import numpy as np
import os
import pickle

writer = SummaryWriter()
parser = argparse.ArgumentParser()

#parser.add_argument('--data_folder', dest='data_folder', type=str, default='./Data/Dialogue-System-for-Automatic-Diagnosis-master/dataset_dxy', help='folder to all data')
#parser.add_argument('--data_folder', dest='data_folder', type=str, default='./Data/mz10/dataset_dxy', help='folder to all data')
#parser.add_argument('--data_folder', dest='data_folder', type=str, default='./Data/mz4/dataset_dxy/', help='folder to all data')
parser.add_argument('--data_folder', dest='data_folder', type=str, default='./Data/Fudan-Medical-Dialogue2.0/synthetic_dataset//dataset_dxy/', help='folder to all data')
parser.add_argument('--max_turn', dest='max_turn', default=22, type=int, help='maximum length of each dialog (default=20, 0=no maximum length)')

# episode(片段)总数 在计算强化学习的价值函数时，样本是一些片段
parser.add_argument('--episodes', dest='episodes', default=5000, type=int, help='Total number of episodes to run (default=1)')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1, help='Epsilon to determine stochasticity(随机性) of epsilon-greedy agent policies')

parser.add_argument('--mode', dest='mode', default='train', type=str, help='train/test')


# RL agent parameters
parser.add_argument('--experience_replay_size', dest='experience_replay_size', type=int, default=10000, help='the size for experience replay')
parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='lr for DQN')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')

parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=100, help='the size of validation set')
parser.add_argument('--target_net_update_freq', dest='target_net_update_freq', type=int, default=1, help='update frequency')
parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=5000, help='the number of epochs for warm start')
parser.add_argument('--supervise', dest='supervise', type=int, default=1, help='0: no supervise; 1: supervise for training')
parser.add_argument('--supervise_epochs', dest='supervise_epochs', type=int, default=100, help='the number of epochs for supervise')

parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./checkpoints/exp_models/KR-DQN/simulate/3', help='write model to disk')
parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3, help='the threshold for success rate')
parser.add_argument('--learning_phase', dest='learning_phase', default='test', type=str, help='train/test; default is all')
parser.add_argument('--train_set', dest='train_set', default='train', type=str, help='train/test/all; default is all')
parser.add_argument('--test_set', dest='test_set', default='test', type=str, help='train/test/all; default is all')

args = parser.parse_args()
params = vars(args)

print('Dialog Parameters: ')
print(json.dumps(params, indent=2))

data_folder = params['data_folder']

goal_set = load_pickle('{}/goal_dict_original_dxy.p'.format(data_folder))
act_set = text_to_dict('{}/dia_acts_dxy.txt'.format(data_folder))  # all acts
slot_set = text_to_dict('{}/slot_set_dxy.txt'.format(data_folder))  # all slots with symptoms + all disease

sym_dict = text_to_dict('{}/symptoms_dxy.txt'.format(data_folder))  # all symptoms
dise_dict = text_to_dict('{}/diseases_dxy.txt'.format(data_folder))  # all diseases
req_dise_sym_dict = load_pickle('{}/req_dise_sym_dict_dxy.p'.format(data_folder))
dise_sym_num_dict = load_pickle('{}/dise_sym_num_dict_dxy.p'.format(data_folder))
dise_sym_pro = np.loadtxt('{}/dise_sym_pro_dxy.txt'.format(data_folder))
sym_dise_pro = np.loadtxt('{}/sym_dise_pro_dxy.txt'.format(data_folder))
sp = np.loadtxt('{}/sym_prio_dxy.txt'.format(data_folder))
tran_mat = np.loadtxt('{}/action_mat_dxy.txt'.format(data_folder))
learning_phase = params['learning_phase']
train_set = params['train_set']
test_set = params['test_set']

priority_replay = False
max_turn = params['max_turn']
num_episodes = params['episodes']


################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']

agent_params['experience_replay_size'] = params['experience_replay_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['lr'] = params['lr']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['supervise'] = params['supervise']
agent_params['fix_buffer'] = fix_buffer
agent_params['priority_replay'] = priority_replay
agent_params['target_net_update_freq'] = params['target_net_update_freq']

agent = AgentDQN(sym_dict, dise_dict, req_dise_sym_dict, dise_sym_num_dict, tran_mat, dise_sym_pro, sym_dise_pro, sp, act_set, slot_set, agent_params, static_policy=True)

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn


usersim_params['data_split'] = params['learning_phase']

user_sim = RuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
test_user_sim = TestRuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
################################################################################
# Dialog Manager
################################################################################
dm_params = {}
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, dm_params)
test_dialog_manager = DialogManager(agent, test_user_sim, act_set, slot_set, dm_params)
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']
supervise = params['supervise']
supervise_epochs = params['supervise_epochs']
success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']

""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
best_model['model'] = agent.model.state_dict()
best_res['success_rate'] = 0

best_te_model = {}
best_te_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
best_te_model['model'] = agent.model.state_dict()

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}


output = False

episode_reward = 0

""" Save model """


def save_model(path,  agent, cur_epoch, best_epoch=0, best_success_rate=0.0, best_ave_turns=0.0, tr_success_rate=0.0, te_success_rate=0.0,  best_hit = 0.0, phase="", is_checkpoint=False):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {}
    checkpoint['cur_epoch'] = cur_epoch
    checkpoint['state_dict'] = agent.model.state_dict()
    if is_checkpoint:
        file_name = 'checkpoint.pth.tar'
        checkpoint['eval_success'] = tr_success_rate
        checkpoint['test_success'] = te_success_rate
    else:
        file_name = '%s_%s_%s_%.3f_%.3f_%.3f.pth.tar' % (phase, best_epoch, cur_epoch, best_success_rate, best_ave_turns, best_hit)
        checkpoint['best_success_rate'] = best_success_rate
        checkpoint['best_epoch'] = best_epoch
    file_path = os.path.join(path, file_name)
    torch.save(checkpoint, file_path)


def simulation_epoch(simulation_epoch_size, output=False):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    episode_hit_rate = 0
    total_hit = 0
    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, dialog_status, hit_rate, stat = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                    if output: print("simulation episode %s: Success" % episode)
                else:
                    if output: print("simulation episode %s: Fail" % episode)
                cumulative_turns += dialog_manager.state_tracker.turn_count
                total_hit += len(dialog_manager.user.goal['implicit_inform_slots'])
                episode_hit_rate += hit_rate
    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    avg_hit_rate = episode_hit_rate / total_hit
    print("simulation success rate %s, ave reward %s, ave turns %s, ave recall %s" % (res['success_rate'], res['ave_reward'], res['ave_turns'],avg_hit_rate))
    return res


def eval(simu_size, data_split):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    user_sim.data_split = data_split
    res = {}
    episode_hit_rate = 0
    total_hit = 0
    avg_hit_rate = 0.0
    for episode in range(simu_size):
        dialog_manager.initialize_episode()
        episode_over = False

        while not episode_over:
            episode_over, r, dialog_status, hit_rate, stat = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                episode_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                cumulative_turns += test_dialog_manager.state_tracker.turn_count
                total_hit += len(test_dialog_manager.user.goal['implicit_inform_slots'])
    
    res['success_rate'] = float(successes) / simu_size
    res['ave_reward'] = float(cumulative_reward) / simu_size
    res['ave_turns'] = float(cumulative_turns) / simu_size
    avg_hit_rate = episode_hit_rate / total_hit
    print("%s hit rate %.4f, success rate %s, ave reward %s, ave turns %s" % (data_split, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

def test(simu_size, data_split):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    user_sim.data_split = data_split
    res = {}
    avg_hit_rate = 0.0
    agent.epsilon = 0
    episode_hit_rate = 0
    total_hit = 0
    test_dialog_manager.user.left_goal = copy.deepcopy(goal_set[data_split])
    request_state = {}
    #print(data_split)
    #print(len(test_dialog_manager.user.left_goal))
    for episode in range(simu_size):
        consult_id = test_dialog_manager.initialize_episode()
        episode_over = False
        request_list = list()
        #print(len(test_dialog_manager.user.left_goal))
        while not episode_over:
            episode_over, r, dialog_status, hit_rate, request_symptom = test_dialog_manager.next_turn()
            if request_symptom:
                request_list.append(request_symptom)
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                episode_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                cumulative_turns += test_dialog_manager.state_tracker.turn_count
                total_hit += len(test_dialog_manager.user.goal['implicit_inform_slots'])
        request_state[consult_id] = request_list[:-1:]
    #pickle.dump(file=open('./records/' + agent_params['trained_model_path'].split('/')[-3] + '/' + agent_params['trained_model_path'].split('/')[-2] + '.p', 'wb'), obj=request_state)
    avg_hit_rate = episode_hit_rate / total_hit
    res['success_rate'] = float(successes) / float(simu_size)
    res['ave_reward'] = float(cumulative_reward) / float(simu_size)
    res['ave_turns'] = float(cumulative_turns) / float(simu_size)
    res['hit_rate'] = avg_hit_rate
    print("%s hit rate %.4f, success rate %.4f, ave reward %.4f, ave turns %.4f" % (data_split, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    agent.epsilon = params['epsilon']
    test_dialog_manager.user.left_goal = copy.deepcopy(goal_set[data_split])
    
    return res, request_state

def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for episode in range(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, dialog_status, hit_rate, stat = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                # print ("warm_start simulation episode %s: Success" % episode)
                # else: print ("warm_start simulation episode %s: Fail" % episode)
                cumulative_turns += dialog_manager.state_tracker.turn_count
        warm_start_run_epochs += 1
        if len(agent.memory) >= agent.experience_replay_size:
            break
    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print("Current experience replay buffer size %s" % (len(agent.memory)))


def training(count):
    # use rule policy, and record warm start experience
    if  params['trained_model_path'] is None and warm_start == 1:
        print('warm_start starting ...')
        warm_start_simulation()
        print('warm_start finished, start RL training ...')
    start_episode = 0
    print(params['trained_model_path'])

    if params['trained_model_path'] is not None:
        trained_file = torch.load(params['trained_model_path'])
        if 'cur_epoch' in trained_file.keys():
            start_episode = trained_file['cur_epoch']

    # dqn simualtion, train dqn, evaluation and save model
    for episode in range(start_episode, count):
        print("Episode: %s" % episode)
        # simulation dialogs

        user_sim.data_split = train_set
        agent.predict_mode = True
        print("data split len " + str(len(user_sim.start_set[user_sim.data_split])))
        # simulate dialogs and save experience
        simulation_epoch(simulation_epoch_size)

        # train by current experience pool
        agent.train()
        agent.predict_mode = False

        # evaluation and test
        #eval_res = eval(5 * simulation_epoch_size, train_set)
        eval_res, request_state = eval(len(goal_set[test_set]), test_set)
        writer.add_scalar('eval/accracy', torch.tensor(eval_res['success_rate'], device=dialog_config.device), episode)
        writer.add_scalar('eval/ave_reward', torch.tensor(eval_res['ave_reward'], device=dialog_config.device), episode)
        writer.add_scalar('eval/ave_turns', torch.tensor(eval_res['ave_turns'], device=dialog_config.device), episode)
        test_res, request_state = test(len(goal_set[test_set]), test_set)
        #test_res = eval(5 * simulation_epoch_size, test_set)
        writer.add_scalar('test/accracy', torch.tensor(test_res['success_rate'], device=dialog_config.device), episode)
        writer.add_scalar('test/ave_reward', torch.tensor(test_res['ave_reward'], device=dialog_config.device), episode)
        writer.add_scalar('test/ave_turns', torch.tensor(test_res['ave_turns'], device=dialog_config.device), episode)

        if test_res['success_rate'] > best_te_res['success_rate']:
            best_te_model['model'] = agent.model.state_dict()
            best_te_res['success_rate'] = test_res['success_rate']
            best_te_res['ave_reward'] = test_res['ave_reward']
            best_te_res['ave_turns'] = test_res['ave_turns']
            best_te_res['epoch'] = episode
            best_te_res['hit_rate'] = test_res['hit_rate']
            save_model(params['write_model_dir'],  agent, episode, best_epoch=best_te_res['epoch'],  best_success_rate=best_te_res['success_rate'],  best_ave_turns=best_te_res['ave_turns'], best_hit=best_te_res['hit_rate'] , phase="test")
            pickle.dump(file=open('./records/' + params['write_model_dir'].split('/')[-2] + '/' + params['write_model_dir'].split('/')[-1] + '.p', 'wb'), obj=request_state)
        # is not fix buffer, clear buffer when accuracy promotes
        if eval_res['success_rate'] >= best_res['success_rate']:
            if eval_res['success_rate'] >= success_rate_threshold:  # threshold = 0.30
                agent.memory.clear()

        if eval_res['success_rate'] > best_res['success_rate']:
            best_model['model'] = agent.model.state_dict()
            best_res['success_rate'] = eval_res['success_rate']
            best_res['ave_reward'] = eval_res['ave_reward']
            best_res['ave_turns'] = eval_res['ave_turns']
            best_res['hit_rate'] = eval_res['hit_rate']
            best_res['epoch'] = episode
            save_model(params['write_model_dir'], agent, episode, best_epoch=best_res['epoch'], best_success_rate=best_res['success_rate'], best_ave_turns=best_res['ave_turns'], best_hit=best_res['hit_rate'] , phase="eval")
        save_model(params['write_model_dir'], agent, episode, is_checkpoint=True)  # save checkpoint each episode

if params['mode'] == 'train':
    training(num_episodes)
else:
    trained_file = torch.load(params['trained_model_path'])
    test_res = test(len(goal_set[test_set]), test_set)

