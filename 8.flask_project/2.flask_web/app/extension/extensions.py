from flask_moment import Moment
from flask_json import FlaskJSON
from flask_bootstrap import Bootstrap
moment = Moment()
flaskjson = FlaskJSON()
bootstrap=Bootstrap()

# 初始化
def extension_config(app):
    moment.init_app(app)
    flaskjson.init_app(app)
    bootstrap.init_app(app)
