"""
该文件的功能是将将正文的语料转换成指定格式的训练样本的格式,并写入指定的文件目录
"""
import json
import logging
import os
import sys
import random
from functools import reduce

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEED = 2020

length_segments = [512, 1024, 2048, 3072, 4096]
avg_length = [0, 0, 0, 0, 0, 0]
max_length = [0, 0, 0, 0, 0, 0]
min_length = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]
acc_length = [0, 0, 0, 0, 0, 0]
sample_count = [0, 0, 0, 0, 0, 0]

avg_length_all = [0, 0, 0, 0, 0, 0]
max_length_all = [0, 0, 0, 0, 0, 0]
min_length_all = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]
acc_length_all = [0, 0, 0, 0, 0, 0]
sample_count_all = [0, 0, 0, 0, 0, 0]


def stat_line(l):
    """
    统计一行文本的相关信息
    :param l:
    :return:
    """
    l = l.strip("\n")
    temp_length = len(l)
    if temp_length < length_segments[0]:
        sample_count[0] += 1
        sample_count_all[0] += 1

    for i in range(0, len(length_segments) + 1):
        if 1 <= i < len(length_segments) and length_segments[i - 1] <= temp_length < length_segments[i] or \
                i == 0 and temp_length < length_segments[i] or \
                i == len(length_segments) and temp_length >= length_segments[i - 1]:
            sample_count[i] += 1
            sample_count_all[i] += 1

            acc_length[i] += temp_length
            acc_length_all[i] += temp_length

            if temp_length > max_length[i]:
                max_length[i] = temp_length
            if temp_length > max_length_all[i]:
                max_length_all[i] = temp_length
            if temp_length < min_length[i]:
                min_length[i] = temp_length
            if temp_length < min_length_all[i]:
                min_length_all[i] = temp_length

            avg_length[i] = acc_length[i] / (sample_count[i] + 1e-6)
            acc_length_all[i] = acc_length_all[i] / (sample_count_all[i] + 1e-6)


def process_file(p, f_train_out, f_test_out, f_dev_out, train_test_ratio):
    """
    处理单个文件的数据
    :param train_test_ratio:
    :param p:
    :param f_train_out:
    :param f_test_out:
    :param f_dev_out:
    :return:
    """
    with open(p, "r") as f_in:

        logger.info(f"processing {p}")
        lines = f_in.readlines()
        indexes = list(range(len(lines)))
        rng = random.Random()
        rng.seed(SEED)
        rng.shuffle(indexes)
        for ind in indexes:
            line = json.loads(lines[ind])
            temp_line = line.get("text", "")
            if not temp_line:
                temp_line = line.get("title", "").strip() + "——" + line.get("content", "").strip()
            if temp_line == "——":
                temp_line = line.get("title", "").strip() + "——" + line.get("answer", "").strip()
            if temp_line == "——":
                temp_line = line.get("english", "").strip() + "——" + line.get("chinese", "").strip()

            line = temp_line
            temp_lines = line.split("\n")
            temp_lines = reduce(lambda l1, l2: l1 + l2.split("\r"), temp_lines, [])
            if reduce(lambda x, y: x + len(y), temp_lines, 0) < 50:
                continue
            # seperators = "== " + data_path.split("/")[-1] + "/" + d + \
            #              "/" + sub_d + "/" + str(ind) + " =="

            if ind < len(lines) * train_test_ratio:
                # f_train_out.writelines(["\n"] * 2 + seperators.split("\n") + ["\n"] * 2)
                f_train_out.writelines(["\n"] * 2 + temp_lines)

            elif ind < len(lines) * (0.5 + 0.5 * train_test_ratio):
                # f_test_out.writelines(["\n"] * 2 + seperators.split("\n") + ["\n"] * 2)
                f_test_out.writelines(["\n"] * 2 + temp_lines)
            else:
                # f_dev_out.writelines(["\n"] * 2 + seperators.split("\n") + ["\n"] * 2)
                f_dev_out.writelines(["\n"] * 2 + temp_lines)

            # 以下代码是统计样本的信息
            # stat_line(seperators)
            line = reduce(lambda x, y: x + y, temp_lines)
            if not len(line) or line.isspace():
                continue
            stat_line(line)


def reformat_data(data_path, train_output_path,
                  test_output_path=None,
                  dev_output_path=None, train_test_ratio=0.8, need_clear_out_path=False):
    """
    将data_path中的文件格式转换成指定格式,并保存到outpath文件中
    :param need_clear_out_path:
    :param dev_output_path:
    :param test_output_path:
    :param data_path:
    :param train_output_path:
    :return:
    """
    for i in range(len(length_segments) + 1):
        avg_length[i] = 0
        max_length[i] = 0
        min_length[i] = float("inf")
        acc_length[i] = 0
        sample_count[i] = 0

    if os.path.isdir(data_path):
        with open(train_output_path, "w+") as f_train_out:
            if dev_output_path and test_output_path:
                with open(dev_output_path, "w+") as f_dev_out, open(test_output_path, "w+") as f_test_out:
                    if need_clear_out_path:
                        f_train_out.seek(0)
                        f_test_out.seek(0)
                        f_dev_out.seek(0)
                    for d in os.listdir(data_path):
                        if os.path.isdir(os.path.join(data_path, d)):
                            for sub_d in os.listdir(os.path.join(data_path, d)):
                                if os.path.isdir(os.path.join(data_path, d, sub_d)):
                                    continue
                                process_file(os.path.join(data_path, d, sub_d), f_train_out, f_test_out, f_dev_out, train_test_ratio)

                        else:
                            process_file(os.path.join(data_path, d), f_train_out, f_test_out, f_dev_out, train_test_ratio)

    print(f"data_path: {data_path}")
    print("\n\n")

    print(f"length_segments: {length_segments}")
    print(f"avg_length: {avg_length}")
    print(f"max_length: {max_length}")
    print(f"min_length: {min_length}")
    print(f"sample_count: {sample_count}")
    sample_count_percentage = [s/sum(sample_count) for s in sample_count]
    print(f"sample_count_percentage: {sample_count_percentage}")

    avg_length_s = 0
    for ind, length in enumerate(avg_length):
        avg_length_s += avg_length[ind] * sample_count[ind]
    avg_length_s /= (sum(sample_count) + 1e-6)
    print(f"avg_length_s: {avg_length_s}")

    print("\n\n\n")
    print(f"length_segments: {length_segments}")
    print(f"avg_length_all: {avg_length_all}")
    print(f"max_length_all: {max_length_all}")
    print(f"min_length_all: {min_length_all}")
    print(f"sample_count_all: {sample_count_all}")
    sample_count_all_percentage = [s/sum(sample_count_all) for s in sample_count_all]
    print(f"sample_count_all_percentage: {sample_count_all_percentage}")

    avg_length_all_s = 0
    for ind, length in enumerate(avg_length_all):
        avg_length_all_s += avg_length_all[ind] * sample_count_all[ind]
    avg_length_all_s /= (sum(sample_count_all) + 1e-6)
    print(f"avg_length_all_s: {avg_length_all_s}")

    print("\n\n\n")
    print("\n\n\n")


if __name__ == "__main__":
    assert len(sys.argv) >= 3, "you must at least specify data path and output path"
    data_path = sys.argv[1]
    output_path = sys.argv[2]

    data_paths = data_path.split(",")

    for ind, data_path in enumerate(data_paths):

        if os.path.isdir(output_path):
            train_output_path = os.path.join(output_path, "train_" + os.path.basename(data_path).split(".")[0] + ".txt")
            test_output_path = os.path.join(output_path, "test_" + os.path.basename(data_path).split(".")[0] + ".txt")
            dev_output_path = os.path.join(output_path, "dev_" + os.path.basename(data_path).split(".")[0] + ".txt")
        else:
            train_output_path = output_path
            test_output_path, dev_output_path = None, None

        reformat_data(data_path,
                      train_output_path=train_output_path,
                      test_output_path=test_output_path,
                      dev_output_path=dev_output_path,
                      train_test_ratio=0.9,
                      need_clear_out_path=True)
