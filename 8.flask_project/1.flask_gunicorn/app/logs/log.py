import logging
import time
import os
import sys

# 日志管理
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler={
    'filename':'logs/{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))),
    'filenot':None
}
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n',
    filename=file_handler['filenot'])
logger = logging.getLogger(__name__)