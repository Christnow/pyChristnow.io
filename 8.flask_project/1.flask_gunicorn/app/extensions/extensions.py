
from flask_moment import Moment
from flask_json import FlaskJSON

moment = Moment()
flaskjson = FlaskJSON()


# 初始化
def extension_config(app):
    moment.init_app(app)
    flaskjson.init_app(app)
