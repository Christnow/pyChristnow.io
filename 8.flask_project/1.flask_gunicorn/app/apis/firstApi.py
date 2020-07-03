import app.script.script1 as script1
from app.responses.responseList import responseList
from flask_restful import Api,Resource, reqparse
from app.logs.log import logger
from flask_json import JsonError, json_response, as_json
from flask import Flask, request
from app.config.config import config
from app.extensions.extensions import extension_config
from app.views import blueprint_config

app = Flask(__name__)
app.config.from_object(config['development'])
# 加载扩展
extension_config(app)

# 注册蓝本
blueprint_config(app)
api = Api()
api.init_app(app)


parser = reqparse.RequestParser()

parser.add_argument('name')
responseList = responseList.get('default')
'''
    sucessGet={'code':2000,'msg':'POST is only','data':''}
    sucessPost={'code':2000,'msg':'ok','data':''}
    parameterError={'code':3001,'msg':'Parameter Error','data':''}
    inputError={'code':3002,'msg':'Input Error','data':''}
    computerError={'code':3003,'msg':'Compute Error','data':''}
'''

class FirstApi(Resource):
    def get(self):
        result=responseList.sucessGet
        callResult = json_response(
            code=result.get('code'),
            msg=result.get('msg'),
            data=result.get('data'))
        logger.info('Result: {}'.format(result))
        return callResult

    def post(self):
        result=responseList.sucessPost
        args = parser.parse_args()
        name=args.get('name','')
        if not name:
            result = responseList.parameterError
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(
                name, result))
            return callResult
        try:
            scriptResult=script1.script1(name)
            result['data']=scriptResult
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(name,result))
        except:
            result=responseList.computerError
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(name,result))
            return callResult
        else:
            return callResult


parser.add_argument('text')


class SencondApi(Resource):
    def get(self):
        result = responseList.sucessGet
        callResult = json_response(
            code=result.get('code'),
            msg=result.get('msg'),
            data=result.get('data'))
        logger.info('Result: {}'.format(result))
        return callResult

    def post(self):
        result = responseList.sucessPost
        args = parser.parse_args()
        name = args.get('text', '')
        if not name:
            result = responseList.parameterError
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(name,result))
            return callResult
        try:
            scriptResult = script1.script2(name)
            result['data'] = scriptResult
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(name,result))
        except:
            result = responseList.computerError
            callResult = json_response(
                code=result.get('code'),
                msg=result.get('msg'),
                data=result.get('data'))
            logger.info('Parameter is {} and Result is {}'.format(name,result))
            return callResult
        else:
            return callResult


@app.route('/testv1', methods=['POST'])
def testv1():
    if request.method == 'GET':
        print(request.args)
        return json_response(code='2000', msg='GET is not', data=''), 301
    elif request.method == 'POST':
        try:
            data = request.json
            data = data.get('num', '')
            if not data:
                return json_response(code=3002, msg='para', data='')
            return json_response(code=2000, msg='ok', data=int(data) + 1)
        except Exception as e:
            logger.info(e)
            return json_response(code=2001, msg='not', data='')


api.add_resource(FirstApi, '/hellov1')

api.add_resource(SencondApi, '/hellov2')