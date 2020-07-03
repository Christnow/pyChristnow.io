class DefaultResponse(object):
    sucessGet={'code':2000,'msg':'POST is only','data':''}
    sucessPost={'code':2000,'msg':'ok','data':''}
    parameterError={'code':3001,'msg':'Parameter Error','data':''}
    inputError={'code':3002,'msg':'Input Error','data':''}
    computerError={'code':3003,'msg':'Compute Error','data':''}


class DevelopmentResponse(DefaultResponse):
    pass


class ProductionResponse(DefaultResponse):
    pass


responseList = {
    'development': DevelopmentResponse,
    'testing': ProductionResponse,
    'production': ProductionResponse,
    'default': DefaultResponse
}