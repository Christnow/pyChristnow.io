import gevent.monkey
gevent.monkey.patch_all()

# debug = True
# loglevel = 'deubg'
# 监听内网端口5000
bind = '0.0.0.0:6876'
# 并行工作进程数
workers = 4
# 指定每个工作者的线程数
threads = 2
# 设置最大并发量
worker_connections = 200
worker_class = 'gevent'
x_forwarded_for_header = 'X-FORWARDED-FOR'
