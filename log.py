import logging
import logging.config
import os
import sys
sys.path.append('/usr/lib/python3/dist-packages/')
import yaml

with open('./config_logging.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('mainLogger')