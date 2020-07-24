from dataset import *
from post_process import Post_Process_NMS
from visualize import *
from forward import *
from evaluation.eval_func import *

# utils
from libs.utils import _init_fn
from libs.load_model import *

def prepare_dataloader(cfg, dict_DB):
    # train dataloader
    if cfg.run_mode == 'train':
        dataset = Train_Dataset_SLNet(cfg)
        trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=cfg.batch_size['img'],
                                                  shuffle=True,
                                                  num_workers=cfg.num_workers,
                                                  worker_init_fn=_init_fn)
        dict_DB['trainloader'] = trainloader

    # test dataloader
    if cfg.dataset == 'SEL':
        dataset = SEL_Test_Dataset(cfg)
        testloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=cfg.batch_size['img'],
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 worker_init_fn=_init_fn)
    else:
        dataset = SEL_Hard_Test_Dataset(cfg)
        testloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=cfg.batch_size['img'],
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 worker_init_fn=_init_fn)

    dict_DB['testloader'] = testloader

    return dict_DB

def prepare_model(cfg, dict_DB):

    if 'test' in cfg.run_mode:
        dict_DB = load_SLNet_for_test(cfg, dict_DB)

    if 'train' in cfg.run_mode:
        dict_DB = load_SLNet_for_train(cfg, dict_DB)

    dict_DB['forward_model'] = Forward_Model(cfg=cfg)

    return dict_DB


def prepare_postprocessing(cfg, dict_DB):

    dict_DB['NMS_process'] = Post_Process_NMS(cfg=cfg)
    return dict_DB

def prepare_visualization(cfg, dict_DB):

    dict_DB['visualize'] = Visualize_plt(cfg=cfg)
    return dict_DB

def prepare_evaluation(cfg, dict_DB):
    dict_DB['eval_func'] = Evaluation_Function(cfg=cfg)

    return dict_DB

def prepare_training(cfg, dict_DB):

    logfile = cfg.output_dir + 'train/log/logfile.txt'
    mkdir(path=cfg.output_dir + 'train/log/')

    if cfg.resume == True:
        rmfile(path=logfile)
        val_result = {'AUC_A': 0, 'AUC_P': 0,
                      'AUC_R_upper_P_0.80': 0,
                      'AUC_R_upper_P_0.82': 0}
        dict_DB['val_result'] = val_result
        dict_DB['epoch'] = 0

    dict_DB['logfile'] = logfile

    return dict_DB
