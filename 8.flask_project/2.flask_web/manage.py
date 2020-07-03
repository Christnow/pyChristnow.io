from app import create_app

# 导入Manager用来设置应用程序可通过指令操作
from flask_script import Manager, Server, Command

app = create_app.create_app()

# 构建指令，设置当前app受指令控制（即将指令绑定给指定app对象）
manage = Manager(app)

manage.add_command(
    'runserver',
    Server(host='0.0.0.0', port=5061, threaded=True, use_debugger=True))

#以下为当指令操作runserver时，开启服务。
if __name__ == '__main__':
    manage.run()
