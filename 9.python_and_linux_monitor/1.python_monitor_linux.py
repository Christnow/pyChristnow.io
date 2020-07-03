import os
import sys
import time
import datetime
import logging

import smtplib
import poplib
import imaplib
from email.mime.text import MIMEText
from email.header import Header
import time
from apscheduler.schedulers.blocking import BlockingScheduler

import psutil
from pynvml import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

risk = 0.95


#cpu模块
def cpu_info():
    status = True
    cpu = '%.2f%%' % psutil.cpu_percent(1)  #把cpu的值改成百分比的形式
    if float(cpu.strip('%')) / 100 >= risk:
        status = False
    return status, cpu


#内存模块
def mem_info():
    status = True
    mem = psutil.virtual_memory()
    mem_per = '%.2f%%' % mem[2]  #同上
    mem_total = str(int(mem[0] / 1024 / 1024)) + 'M'
    mem_used = str(int(mem[3] / 1024 / 1024)) + 'M'
    info = {'mem_per': mem_per, 'mem_total': mem_total, 'mem_used': mem_used}
    if float(mem_per.strip('%')) / 100 >= risk:
        status = False
    return status, info


#磁盘分区模块
def disk_info():
    status = True
    data_per = '%.2f%%' % psutil.disk_usage('/data')[3]
    info = {
        'data_per': data_per,
    }
    if float(data_per.strip('%')) / 100 >= risk:
        status = False
    return status, info


#网卡模块
def network_info():
    status = True
    network = psutil.net_io_counters()
    network_sent = str(int(network[0] / 8 / 1024 / 1024)) + 'M'
    network_recv = str(int(network[1] / 8 / 1024 / 1024)) + 'M'
    info = {'network_sent': network_sent, 'network_recv': network_recv}
    return status, info


#GPU模块
def gpu_info():
    infor = {}
    status = True
    status_list = []
    nvmlInit()
    count = nvmlDeviceGetCount()
    for i in range(count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        info = nvmlDeviceGetMemoryInfo(handle)
        infor[str(i)] = '{}-第{}号GPU使用量：{}'.format(
            name.decode('utf-8'), i, '%.2f%%' % (info.used / info.total))
        status_list.append(info.used / info.total)
    nvmlShutdown()
    for i in status_list:
        if i >= risk:
            status = False
    return status, infor


def process():
    status = True
    logger.info('Start obtain information\n')
    try:
        t1 = time.time()
        cpu_status, cpu = cpu_info()
        logger.info('Obtain cpu information time is {}\n'.format(time.time() -
                                                                 t1))
        t2 = time.time()
        memory_status, memory = mem_info()
        logger.info(
            'Obtain memory information time is {}\n'.format(time.time() - t2))
        t3 = time.time()
        disk_status, disk = disk_info()
        logger.info(
            'Obtain data disk information time is {}\n'.format(time.time() -
                                                               t3))
        t4 = time.time()
        metwork_status, network = network_info()
        logger.info(
            'Obtain network information time is {}\n'.format(time.time() - t4))
        t5 = time.time()
        gpu_status, gpu = gpu_info()
        logger.info('Obtain gpu information time is {}\n'.format(time.time() -
                                                                 t5))
        logger.info('Obtain all information is ok\n')
    except Exception as e:
        logger.info('Obtain information is not ok\n')
        logger.info('The reason is :\n{}'.format(e))
        msg = ''
        return status, msg
    msg = '''
    ====================    
    cpu使用率：{}
    ====================
    内存占用率：{}
    内存总量：{}
    内存使用量：{}
    ====================
    data盘使用率：{}
    ====================
    网卡发送量：{}
    网卡接收量：{}
    ====================
    {}
    {}
    {}
    {}
    ====================
    '''.format(cpu, memory.get('mem_per'), memory.get('mem_total'),
               memory.get('mem_used'), disk.get('data_per'),
               network.get('network_sent'), network.get('network_recv'),
               gpu['0'], gpu['1'], gpu['2'], gpu['3'])
    logger.info('The information is {}'.format(msg))
    logger.info('Return information is ok\n')
    if not cpu_status or not memory_status or not disk_status or not gpu_status:
        status = False
    return status, msg


class operate_email:
    def __init__(self, your_email_address, your_email_passwd):
        self.address = your_email_address
        self.password = your_email_passwd

    def send_email_by_smtp(self, receive_email_address, message_subject,
                           message_context):
        smtp_server_host = "smtp.exmail.qq.com"
        semtp_server_post = "465"
        message = MIMEText(message_context, 'plain', 'utf-8')
        message['From'] = Header(self.address, 'utf-8')
        message['To'] = Header(receive_email_address, 'utf-8')
        message['Subject'] = Header(message_subject, 'utf-8')
        email_client = smtplib.SMTP_SSL(smtp_server_host, semtp_server_post)
        try:
            email_client.login(self.address, self.password)
            logger.info(
                "smtp----login sucess,now will send email to {}".format(
                    receive_email_address))
        except Exception as e:
            logger.info(e)
            try:
                logger.info('smtp----login again.....')
                email_client.login(self.address, self.password)
                logger.info(
                    "smtp----login sucess,now will send email to {}".format(
                        receive_email_address))
            except Exception as e:
                logger.info(e)
                logger.info(
                    'smtp----sorry,check yourusername or yourpassword not correct or another problem occur'
                )
        else:
            email_client.sendmail(self.address, receive_email_address,
                                  message.as_string())
            logger.info('smtp----send email to {} finish'.format(
                receive_email_address))
        finally:
            email_client.close()


sched = BlockingScheduler()


@sched.scheduled_job('interval', seconds=3600)
def main():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_logging_path = './logging'
    my_email_address = 'email_address'
    my_email_passwd = 'email_passwd'
    receive_email_address = 'receive_email_address'
    message_subject = 'email information'
    status, message_context = process()
    if not status:
        client = operate_email(my_email_address, my_email_passwd)
        client.send_email_by_smtp(receive_email_address, message_subject,
                                  message_context)


if __name__ == '__main__':
    sched.start()
