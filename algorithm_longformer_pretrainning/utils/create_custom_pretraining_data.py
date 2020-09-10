"""
根据下载的json构造训练样本
策略:
1. 过滤掉table/chart/图片
2. 过滤页眉页脚
3.不同的段落直接拼接成长度为4096的窗口大小的一行样本,相邻窗口间有一定的重叠,比如512字符,最小的处理单位为段落
"""
import functools
import itertools
import json
import os
import threading
import logging

log = logging.getLogger(__file__)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)
COUNT = 0

from algorithm_longformer_pretrainning.test.stat_year_report import download_path_gte4096


def get_sample(p, out_path, splide_size=4096, overlap_size=512, need_clear_out_path=True):
    if need_clear_out_path:
        if os.path.isfile(out_path):
            with open(out_path, "w+") as f:
                f.seek(0)
        elif os.path.isdir(out_path):
            for o_p in os.listdir(out_path):
                if os.path.isdir(os.path.join(out_path, o_p)):
                    continue
                with open(os.path.join(out_path, o_p), "w+") as f:
                    f.seek(0)
    cur, step = 0, 2000
    all_paths = os.listdir(p)
    length_all_paths = len(all_paths)
    # length_all_paths = 132000
    thread_list = []
    while cur + step < length_all_paths or cur < length_all_paths <= cur + step:
        if cur < length_all_paths <= cur + step:
            sub_paths = all_paths[cur: length_all_paths]
            if os.path.isfile(out_path):
                sub_out_path = os.path.join(os.path.dirname(out_path), str(cur+1) + "_" + str(length_all_paths) + ".txt")
            else:
                sub_out_path = os.path.join(out_path, str(cur+1) + "_" + str(length_all_paths) + ".txt")

        else:
            sub_paths = all_paths[cur: cur+step]
            if os.path.isfile(out_path):
                sub_out_path = os.path.join(os.path.dirname(out_path), str(cur+1) + "_" + str(cur + step + 1) + ".txt")
            else:
                sub_out_path = os.path.join(out_path, str(cur+1) + "_" + str(cur + step + 1) + ".txt")

        get_train_data_list(p, sub_paths, sub_out_path, splide_size=splide_size, overlap_size=overlap_size)
        # t = threading.Thread(target=get_train_data_list,
        #                      args=(p, sub_paths, sub_out_path),
        #                      kwargs=dict({"splide_size": splide_size, "overlap_size": overlap_size}))
        # thread_list.append(t)

        cur += step

    # for t in thread_list:
    #     t.start()
    # for t in thread_list:
    #     t.join()

            
def get_train_data_list(p, sub_paths, sub_out_path, splide_size=4096, overlap_size=512):
    global COUNT
    for item in sub_paths:
        # if item != "5ec651ee2efa2424e71c1535.json":
        #     continue
        COUNT += 1
        log.info(f"processing file: {item}, file_size: {os.path.getsize(os.path.join(p, item))}, count: {COUNT}\n\n")
        with open(os.path.join(p, item), "r") as f:
            cont = json.load(f)
            count = get_train_data(cont, sub_out_path, splide_size=splide_size,
                           overlap_size=overlap_size)
    return (item, count)


def get_train_data(cont, out_path, splide_size=4096, overlap_size=512):
    """
    从解析结果里面获取窗口大小为splide_size,相邻窗口之际爱呢重叠大小为overlap_size的训练样本
    :param need_clear_out_path:
    :param cont:
    :param splide_size:
    :param overlap_size:
    :return:
    """
    def func_para(para):
        return not para.get("is_footer", False) \
               and not para.get("is_header", False) \
               and not para.get("cover_table", "") \
               and para.get("text", "").strip().__len__() > 1

    def func_page(x, item):
        paras = filter(lambda x: func_para(x), item["paragraphs"])
        x = itertools.chain(x, paras)
        return x

    def write_func(x):
        f.write(x)
        f.write("\n\n\n")

    paras = []
    pages = cont["pages"]
    paras = functools.reduce(lambda x, item: func_page(x, item), pages, paras)
    # paras = map(lambda x: x.get("text", "").strip(), paras)
    paras = list(filter(lambda x: x.get("text", "").replace(" ", "").__len__() >= 0.2 * x.get("text", "").__len__(), paras))
    paras_size = len(paras)

    # 以下代码是按照滑动窗口进行采样
    cur = 0
    out_lines = []
    temp_line = ""
    count = 0
    last_cur = 0
    last_para = None
    while cur < paras_size:
        if len(temp_line) + len(paras[cur]) < splide_size:
            if not temp_line:
                temp_line += paras[cur].get("text", "").strip()
            if last_para and last_para.get("nextPageParagraph", -1) == paras[cur].get("pid"):
                temp_line += paras[cur].get("text", "").strip()
            else:
                temp_line += "[unused1]" + paras[cur].get("text", "").strip()
            last_para = paras[cur]
            cur += 1
        else:
            out_lines.append(temp_line)
            temp_line = ""
            acc_length = 0

            while cur > 0 and acc_length < overlap_size:
                acc_length += len(paras[cur-1].get("text", "").strip())
                cur -= 1

            if last_cur == cur:
                cur += 1

            last_cur = cur
            if len(out_lines) > 10:
                with open(out_path, "a+") as f:
                    list(map(lambda x: write_func(x), out_lines))
                log.debug(f"out_lines: \n {json.dumps(out_lines, ensure_ascii=False, indent=4)}")
                count += len(out_lines)
                out_lines = []

    if temp_line:
        out_lines.append(temp_line)
        temp_line = ""

    with open(out_path, "a+") as f:
        list(map(lambda x: write_func(x), out_lines))
    log.debug(f"out_lines: \n{json.dumps(out_lines, ensure_ascii=False, indent=4)}")
    return count + len(out_lines)


if __name__ == "__main__":
    out_path = os.path.dirname(download_path_gte4096)
    get_sample(download_path_gte4096, out_path, need_clear_out_path=True)

    # out_path = os.path.dirname(download_path_gte4096) + "/74001_76001.txt"
    # with open(out_path, "r") as f:
    #     for line in f:
    #         print(line)