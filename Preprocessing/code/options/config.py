import os
import torch

class Config(object):
    def __init__(self):
        self.settings_for_system()
        self.settings_for_path()

        self.settings_for_image_processing()

        self.settings_for_visualization()
        self.settings_for_save()

    def settings_for_system(self):
        self.gpu_id = "0"
        self.seed = 123

        # GPU setting
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def settings_for_path(self):
        self.dir = dict()
        self.dir['head'] = '--system root'
        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'

        self.settings_dataset_path()

        self.dir['out'] = self.dir['proj'] + 'output_{}_{}/'.format(self.dataset_name, self.datalist_mode)

    def settings_dataset_path(self):
        self.dataset_name = 'SEL'  # ['SEL']
        self.datalist_mode = 'train'  # ['train', 'test']

        self.dir['dataset'] = dict()
        self.dir['dataset']['SEL'] = self.dir['head'] + '--SEL dataset root'
        self.dir['dataset']['SEL_img'] = self.dir['dataset']['SEL'] + 'ICCV2017_JTLEE_images/'

    def settings_for_image_processing(self):
        self.height = 400
        self.width = 400
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def settings_for_visualization(self):
        self.display = True

    def settings_for_save(self):
        self.save_pickle = True
