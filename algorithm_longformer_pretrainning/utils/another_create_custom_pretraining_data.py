import os

from algorithm_longformer_pretrainning.test.stat_year_report import download_path_lt4096
from algorithm_longformer_pretrainning.utils.create_custom_pretraining_data import get_train_data_list


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
    cur, step = 0, 5000
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


if __name__ == "__main__":
    out_path = os.path.dirname(download_path_lt4096) + "/another_samples"
    get_sample(download_path_lt4096, out_path, need_clear_out_path=True)