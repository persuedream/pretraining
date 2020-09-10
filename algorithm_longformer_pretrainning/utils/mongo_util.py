"""
该类主要是对mongo的一些操作
"""
import json
import logging
import os

from bson.objectid import ObjectId

from abcft_algorithm_cluster_service.base_db import BaseMongo

from algorithm_longformer_pretrainning.utils import decorators

log = logging.getLogger(__name__)

MONGO_VER = 1


class Mongo(BaseMongo):

    def __init__(self, **kwargs):
        kwargs["params"] = None
        self.version = MONGO_VER
        self.version_field = "financial_report_table_extraction_ver"
        super(Mongo, self).__init__(**kwargs)

    def create_index(self):
        pass

    def check_status(self):
        pass

    def is_valid(self, record):
        pass

    def find_missing(self, name, next_updated, prev_updated, n=1):
        pass

    def find_prev(self, name, last_updated, n=1):
        pass

    def find_next(self, name, last_updated, n=1):
        pass

    def _id_validate(self, _id):
        """
        id合法化
        :param _id:
        :return:
        """
        if isinstance(_id, str):
            if _id.isdigit():
                _id = int(_id)
            else:
                _id = ObjectId(_id)
        return _id

    def get_file_id(self, json_url):
        coll = self._get_text_coll("juchao" if self.sources == "finance.juchao.item" else self.sources)
        one_item = coll.find_one({"json_file": json_url})
        _id = one_item["_id"]
        return _id

    def _get_source_files_coll(self, name):
        if name == "finance.juchao.item":
            return self.db[name]
        return self.db[name + "_files"]

    def _update_source_files_coll(self, name, _id, update):
        coll = self._get_source_files_coll(name)
        update = BaseMongo.build_update(update)
        coll.update_one({"_id": self._id_validate(_id)}, update, upsert=True)

    def get_table_url_list(self, _id, data_type="json"):
        """
        获取所有table url list
        :param data_type:
        :param _id:
        :return:
        """
        table_url_list = []
        coll = self._get_tables_coll("juchao" if self.sources == "finance.juchao.item" else self.sources)
        all_items = coll.find({"fileId": self._id_validate(_id)})
        for item in all_items:
            if data_type == "json":
                table_url_list.append(item["data_file"])
            elif data_type == "html":
                table_url_list.append(item["html_file"])
            elif data_type == "all":
                table_url_list.append((item.get("data_file", ""), item.get("html_file", "")))
        return table_url_list

    def get_fulltext_url(self, _id, data_type="json"):
        fulltext_url = ""
        coll = self._get_text_coll("juchao" if self.sources == "finance.juchao.item" else self.sources)
        all_items = coll.find({"fileId": self._id_validate(_id)})
        for item in all_items:
            if data_type == "json":
                fulltext_url = item["json_file"]
            elif data_type == "html":
                fulltext_url = item["html_file"]

        return fulltext_url

    def get_table_url(self, _id, table_id, data_type="json"):
        """
        利用file id和table id获取html file的url
        :param data_type:
        :param _id:
        :param table_id:
        :return:
        """
        coll = self._get_tables_coll("juchao" if self.sources == "finance.juchao.item" else self.sources)
        all_items = coll.find({"fileId": self._id_validate(_id)})
        for item in all_items:
            if item["_id"] == table_id:
                if data_type == "json":
                    return item["data_file"]
                else:
                    return item["html_file"]

    def get_file_url(self, _id):
        """
        根据file_id获取file_url
        :param _id:
        :return:
        """
        coll = self._get_source_files_coll(self.sources)
        item = coll.find_one({
            "_id": self._id_validate(_id)
        })
        return item.get("file_url", "")

    def get_file_parse_status(self, _id):
        coll = self._get_source_files_coll(self.sources)
        item = coll.find_one({
            "_id": self._id_validate(_id)
        })
        return item.get("process_error", "SUCCESS")

    def write_extract_parameters(self, _id, extract_parameters):
        coll = self._get_source_files_coll(self.sources)
        item = coll.find_one({
            "_id": self._id_validate(_id)
        })
        item["extract_parameters"] = extract_parameters

        self._update_source_files_coll(self.sources, _id, item)

    def get_last_process_time(self, _id):
        """
        获取上次运行时间
        :param _id:
        :return:
        """
        coll = self._get_source_files_coll(self.sources)
        item = coll.find_one({
            "_id": self._id_validate(_id)
        })
        return item.get("last_process_time", "")


_Mongo = {}


@decorators.load_config(os.path.dirname(__file__) + "/mongo.json", mongo_conf=True)
def init(**kwargs):
    global _Mongo
    if _Mongo:
        log.info("_Mongo is already initialized")
        return
    _Mongo = Mongo(**kwargs)


def get_mongo() -> object:
    if not _Mongo:
        raise RuntimeError("_Mongo is not initialized")
    return _Mongo


def delete():
    global _Mongo
    log.info("close db...")
    if _Mongo:
        del _Mongo


def get_file_table_list(parse_file_id_list):
    if isinstance(parse_file_id_list, str):
        parse_file_id_list = json.loads(parse_file_id_list)
    table_url_list = []
    init()
    mongo = get_mongo()
    for file_id in parse_file_id_list:
        table_url_list.append(mongo.get_table_url_list(file_id))
    return table_url_list


def get_file_url_list(parse_file_id_list):
    if isinstance(parse_file_id_list, str):
        parse_file_id_list = json.loads(parse_file_id_list)
    file_url_list = []
    init()
    mongo = get_mongo()
    for file_id in parse_file_id_list:
        file_url_list.append(mongo.get_file_url(file_id))
    return file_url_list
