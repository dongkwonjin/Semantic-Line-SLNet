from options.config import Config
from libs.prepare import *

def run_for_preprocessing(dict_DB):

    line_generator = dict_DB['generator']
    line_generator.run()

if __name__ == '__main__':
    # option
    cfg = Config()

    dict_DB = dict()

    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_label_generator(cfg, dict_DB)


    # Process
    run_for_preprocessing(dict_DB)
