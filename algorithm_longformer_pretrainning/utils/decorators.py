import json


def load_config(conf_path, mongo_conf=True):
    """
    从配置中加载参数
    :return:
    """

    def decorator(func):
        def handle(*args, **kwargs):
            with open(conf_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if mongo_conf:
                config["mongo_uri"] = config[config["mongo_uri"]]
            func_res = func(*args, **kwargs, **config)
            return func_res

        return handle

    return decorator
