import os, sys

from config import load_config
from main import main
# CUDA env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load config file
config = load_config('./config/Caltech101-20')
current_dir = sys.path[0]
config['main_dir'] = current_dir

# run
main(config)
