import os
import sys
import time

from flask_restful import Api, Resource, reqparse
from flask_json import JsonError, json_response, as_json
from flask import Flask, request, jsonify, abort, Response, render_template

from app.log.log import logger
from app.config.config import config
from app.view import blueprint_config
from app.responses.responList import responseList
from app.extension.extensions import extension_config

########## 初始化配置app ##########
app = Flask(__name__)
app.config.from_object(config['development'])
# 加载扩展
extension_config(app)
# 注册蓝本
blueprint_config(app)
api = Api()
api.init_app(app)
########## 初始化配置app ##########

########## 实例化Neo4j模型 ##########

logger.info('Web index html is ok')


@app.route('/keke', methods=["GET", "POST"])
def indexmain():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return render_template('index.html')


@app.route('/appointment', methods=["GET", "POST"])
def indexmain1():
    if request.method == 'GET':
        return render_template('index1.html')
    elif request.method == 'POST':
        return render_template('index1.html')


@app.route('/photov1', methods=["GET", "POST"])
def indexmain2():
    if request.method == 'GET':
        return render_template('index2.html')
    elif request.method == 'POST':
        return render_template('index2.html')


@app.route('/photov2', methods=["GET", "POST"])
def indexmain3():
    if request.method == 'GET':
        return render_template('index3.html')
    elif request.method == 'POST':
        return render_template('index3.html')

@app.route('/photov3', methods=["GET", "POST"])
def indexmain4():
    if request.method == 'GET':
        return render_template('index4.html')
    elif request.method == 'POST':
        return render_template('index4.html')