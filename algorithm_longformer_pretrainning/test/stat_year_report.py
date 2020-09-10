"""
该脚本用来统计年报的相关信息
"""
import functools
import json
import os
from collections import Counter
from datetime import datetime
from random import random

import requests
from bson import ObjectId

from algorithm_longformer_pretrainning.utils import mongo_util


count = 0

source_type_list = ["F004_1", "F004_2", "F004_3", "F004_4", "F004_5", "F004_6", "F004_7", "F004_8",
                    "F006_1", "F006_2", "F006_3", "F006_4", "F006_5", "F006_6", "F006_7", "F006_8",
                    "F006_9", "F006_10", "F006_11", "F006_12", "F006_13", "F006_14", "F006_14", "F006_16"]

download_path_gte4096 = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/gte4096"
download_path_lt4096 = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/lt4096"

def func(x, item):
    global count
    if count % 10000 == 0:
        print(f"processing: {item['_id']}, count: {count}")
    if isinstance(x, list):
        x.append({
            "_id": item["_id"].__str__(),
            "file_url": "http://abc-crawler.oss-cn-hangzhou.aliyuncs.com/" + item.get("oss_path", ""),
            "source_type": item.get("source_type", ""),
            "category_id": item.get("category_id", ""),
            "characterCount": item.get("characterCount", 0)

        })
    count += 1
    return x


def func_json_url(x, item):
    global count
    # if count > 140000:
    #     print(f"id: {item['_id']}, count: {count}")
    if count % 10000 == 0:
        print(f"processing: {item['_id']}, count: {count}")
    if isinstance(x, list):
        x.append({
            "json_file": item.get("json_file", ""),
            "_id": item.get("_id", "").__str__()
        })
    count += 1
    return x


def get_stats(bigger=True, source_type_list=[]):
    mongo_util.init()
    mongo = mongo_util.get_mongo()
    col = mongo._get_source_files_coll(mongo.sources)
    if bigger:
        all_items = col.find({
            "time": {"$gt": datetime(2019, 12, 31)},
            "characterCount": {"$gte": 4096},
            "source_type": {"$in": source_type_list}
        })
    else:
        all_items = col.find({
            "time": {"$gt": datetime(2019, 12, 31)},
            "characterCount": {"$lt": 4096},
            "source_type": {"$in": source_type_list}
        })

    source_infos = functools.reduce(lambda x, item: func(x, item), all_items, list())
    if bigger:
        fname = "year_report_stats.json"
    else:
        fname = "year_report_stats_lt.json"
    with open(fname, "w") as f:
        json.dump(source_infos, f, ensure_ascii=False, indent=4)


def func_add(x, item):
    x[item.get("source_type")] += 1
    x["total_character_count"] += item.get("characterCount", 0)
    return x


def get_file_type_stats(bigger=True):
    file_name = "year_report_stats.json"
    if not bigger:
        file_name = "year_report_stats_lt.json"
    with open(file_name, "r") as f:
        cont = json.load(f)
    counter = Counter()
    functools.reduce(lambda x, item: func_add(x, item), cont, counter)

    file_name = "year_report_stats2.json"
    if not bigger:
        file_name = "year_report_stats2_lt.json"
    with open(file_name, "w") as f:
        json.dump({
            "detail": cont,
            "summary": counter
        }, f, ensure_ascii=False, indent=4)


def get_query_list(iterable, coll, qury_dict, project={}, func=None, step=1000, sum_count=0):
    url_list = []
    cur = 0

    while cur + step < sum_count or cur < sum_count <= cur + step:
        if cur < sum_count <= cur + step:
            id_list = list(map(lambda x: ObjectId(x["_id"]), iterable[cur:]))
        else:
            id_list = list(map(lambda x: ObjectId(x["_id"]), iterable[cur: cur+step]))

        col_item = coll.find({
            "_id": {"$in": id_list}
        }, projection={"meta.creationDate": False, "meta.modificationDate": False,
                       "create_time": False, "last_updated": False})

        url_list = functools.reduce(lambda x, item: func_json_url(x, item), col_item, url_list)
        cur += step
    return url_list


def download_json(bigger=True):
    file_name = "year_report_stats.json"
    if not bigger:
        file_name = "year_report_stats_lt.json"
    with open(file_name, "r") as f:
        cont = json.load(f)
    mongo_util.init()
    mongo = mongo_util.get_mongo()
    coll = mongo._get_text_coll("juchao" if mongo.sources == "finance.juchao.item" else mongo.sources)

    sum_count = len(cont)
    cur = 0
    step = 10000
    json_url_list = []
    while cur + step < sum_count or cur < sum_count <= cur + step:
        if cur < sum_count <= cur + step:
            id_list = list(map(lambda x: ObjectId(x["_id"]), cont[cur:]))
        else:
            id_list = list(map(lambda x: ObjectId(x["_id"]), cont[cur: cur+step]))

        col_item = coll.find({
            "_id": {"$in": id_list}
        }, projection={"meta.creationDate": False, "meta.modificationDate": False,
                       "create_time": False, "last_updated": False, "meta.h": False, "meta.i": False})

        json_url_list = functools.reduce(lambda x, item: func_json_url(x, item), col_item, json_url_list)
        cur += step

    file_name = "year_report_json_files.json"
    if not bigger:
        file_name = "year_report_json_files_lt.json"
    with open(file_name, "w") as f:
        json.dump(json_url_list, f, ensure_ascii=False, indent=4)


def download_url(bigger=True):
    if bigger:
        file_name = "year_report_json_files.json"
    else:
        file_name = "year_report_json_files_lt.json"
    with open(file_name, "r") as f:
        cont = json.load(f)
    from multiprocessing.pool import ThreadPool
    results = ThreadPool(4).imap_unordered(lambda item: download(item, bigger=bigger), cont)
    for ind, i in enumerate(results):
        print(f"ind: {ind}   " + i + " has downloaed!")
    # list(map(lambda item: download(item, bigger), cont))


def download(item, bigger=True, max_retry=5):
    global count
    if count % 10000 == 0:
        print(f"downloading: {item.get('_id', '')}")
    file_name = download_path_gte4096
    if not bigger:
        file_name = download_path_lt4096
    for i in range(max_retry):
        try:
            if not item["json_file"].startswith("http:"):
                print(f"{item.get('json_file', '')} is not formated, sikped!!!!")
                return item.get("_id", "")
            r = requests.get(item["json_file"], stream=True)
        except requests.exceptions.ConnectionError as e:
            print(f"id: {item.get('_id', '')}, exception: {e}, try again!!!")
        else:
            break

    with open(os.path.join(file_name, item.get("_id", str(random())) + ".json"), "wb") as f:
        f.write(r.content)
    count += 1
    return item.get("_id", "")


if __name__ == "__main__":
    start = datetime.now()
    download_url()
    end = datetime.now()
    print(f"download 1000 pdf cost: {end-start}")