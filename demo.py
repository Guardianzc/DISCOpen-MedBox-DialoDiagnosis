import OpenMedicalChatBox as OMCB
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

HRL_test = OMCB.HRL(dataset_path = '.\Data\mz4\HRL\\', model_save_path = './simulate', groups = 2, model_load_path = './simulate', cuda_idx = 1, train_mode = True)
HRL_test.run()

KRDS_test = OMCB.KRDS(dataset_path = '.\Data\mz4\dataset_dxy\\', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 1, train_mode = True)
KRDS_test.run()


Flat_DQN_test = OMCB.Flat_DQN(dataset_path = '.\Data\mz4\\', model_save_path = './simulate',  model_load_path = './simulate', cuda_idx = 1, train_mode = True)
Flat_DQN_test.run()


GAMP_test = OMCB.GAMP(dataset_path = '.\Data\mz4\\', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 1, train_mode = True)
GAMP_test.run()

REFUEL_test = OMCB.REFUEL(dataset_path = '.\Data\mz4\\', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 0, train_mode = True)
REFUEL_test.run()
