
class DefaultConfig(object):
    DEBUG = True


class DevelopmentConfig(DefaultConfig):
    DEBUG = True
    JSON_ADD_STATUS = True
    JSON_STATUS_FIELD_NAME = 'http_status'
    JSON_AS_ASCII = False


class ProductionConfig(DefaultConfig):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'testing': ProductionConfig,
    'production': ProductionConfig,
    'default': DefaultConfig
}

if __name__ == '__main__':
    print(config['DevelopmentConfig'])