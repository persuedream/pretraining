import json
import os

from algorithm_longformer_pretrainning.algorithms.zh_wiki import zh2Hans
from algorithm_longformer_pretrainning.test.stat_year_report import download_path_lt4096, download_path_gte4096

def filter_hans():
    lt4096_path = os.path.dirname(download_path_lt4096) + "/another_samples"
    lt4096_filter_path = os.path.dirname(download_path_lt4096) + "/another_samples_filtered"
    gte4096_path = os.path.dirname(download_path_gte4096) + "/samples"
    gte4096_filter_path = os.path.dirname(download_path_gte4096) + "/samples_filtered"

    acc_hans_count = 0
    acc_count = 0
    for d in os.listdir(gte4096_path):
        with open(os.path.join(gte4096_path, d), "r") as fin, \
                open(os.path.join(gte4096_filter_path, d), "w+") as fout:
            fout.seek(0)
            for ind, line in enumerate(fin):
                if not line.strip():
                    fout.write(line)
                    continue
                acc_hans = 0
                has_has = False
                for c in line:
                    if c in zh2Hans and c not in zh2Hans.values():
                        acc_hans += 1
                    if acc_hans > len(line) * 0.1 and acc_hans > 10:
                        has_has = True
                        acc_hans_count += 1
                        print(f"行号: {ind}, 累计繁体行数: {acc_hans_count}, file_name: {d}, 繁体字: {line}")
                        break
                if not has_has:
                    acc_count += 1
                    fout.write(line)
                else:
                    continue
            print(f"\n\n\nfile_name: {d}, 累计行数: {acc_count}")
            fout.flush()

def add_yq_new_line_between():
    p = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/yuq/pretrain_data"
    p_out = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/yuq/pretrain_data_filtered"
    acc_count = 0
    for d in os.listdir(p):
        file_count = 0
        with open(os.path.join(p, d), "r") as fin, open(os.path.join(p_out, d), "w+") as fout:
            for ind, line in enumerate(fin):
                if not line:
                    continue
                fout.write(line)
                fout.write("\n\n")
                file_count += 1
                acc_count += 1
            fout.flush()
        print(f"file: {d}, 行数: {file_count}, 总行数: {acc_count}")


def get_wechat_public_accouts():
    p = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/wechat_public_accounts/nohup.out"
    p_out = "/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/wechat_public_accounts_filtered/articles.json"
    def func_line(line):
        try:
            cont = json.loads(line)
        except json.decoder.JSONDecodeError as e:
            print(line)
            print(f"exception: {e}")
            return line
        return cont.get("content", "")
    acc_count = 0
    acc_line_length = 0
    with open(p, "r") as f, open(p_out, "w+") as fout:
        cont = map(lambda line: func_line(line), f)

        for line in cont:
            if line:
                line = line.replace("\n", "")
                acc_line_length += len(line)
                fout.write(line)
                fout.write("\n\n")
                acc_count += 1
        fout.flush()
        print(f"总行数: {acc_count}, 平均每行字符长度: {acc_line_length / acc_count}")


if __name__ == "__main__":
    get_wechat_public_accouts()