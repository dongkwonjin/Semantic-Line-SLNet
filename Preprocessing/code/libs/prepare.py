from visualizes.visualize import *
from libs.generator import *

def prepare_visualization(cfg, dict_DB):

    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

def prepare_label_generator(cfg, dict_DB):

    dict_DB['generator'] = generate_line(cfg, dict_DB['visualize'])
    return dict_DB


