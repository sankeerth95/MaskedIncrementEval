import os
from itertools import product
import subprocess
from run_benchmark import compare_benchmarks_gavin
import torch
import time
import tqdm

def replace_number(replace_dictionary):
    """
    Go to the folder and replace
    :param replace_dictionary:
    :return:
    """
    if "ext_conv.template" in os.listdir():
        folder_to_template = "."
    else:
        folder_to_template = "./ext/pointops/_C_ext/pointops/"

    # with open("./ext/pointops/_C_ext/pointops/ext_conv.template", "r") as ext_conv_fh:
    with open(os.path.join(folder_to_template, "ext_conv.template"), "r") as ext_conv_fh:
        ext_conv_fh_template = ext_conv_fh.read()

    for key, val in replace_dictionary.items():
        ext_conv_fh_template = ext_conv_fh_template.replace(key, str(val))

    with open(os.path.join(folder_to_template, "ext_conv.cu"), "w") as ext_conv_fh_new:
        ext_conv_fh_new.write(ext_conv_fh_template)

def compile():
    """
    Go to the folder and compile
    :return:
    """
    if "ext_conv.cu" in os.listdir():
        pass
    else:
        os.chdir("./ext/pointops/_C_ext/pointops/")

    my_env = os.environ.copy()
    # my_env["MAX_JOBS"] = "8"

    before = time.time()
    out = subprocess.run(["python3", "setup.py", "build_ext", "--inplace"], capture_output=True, env=my_env)
    after = time.time()

    assert out.returncode == 0, f"the out returned a non-zero return code: {out.returncode=}, {out=}"


def run_experiment(master_size_list):
    single_experiment_result = {}
    for size_list in master_size_list:
        return_val = compare_benchmarks_gavin(in_shape=size_list[0], shape=size_list[1], k=size_list[2])
        # print(f"{size_list=}, {return_val=}")
        single_experiment_result[str(size_list)] = return_val

    return single_experiment_result


def replace_compile_and_run(replace_dictionary, master_size_list):
    """
    Iterate through a specific set of experiment (with multiple shapes, but only 1 ext_conv)
    :param replace_dictionary:
    :param master_size_list:
    :return:
    """
    if replace_dictionary["OUT_CHANNELS_PER_BLOCK_TAG"] > replace_dictionary["THREADS_TAG"]:
        return None

    if replace_dictionary["TAG_IMAGE_DIM1"] != replace_dictionary["TAG_IMAGE_DIM2"]:
        return None

    replace_number(replace_dictionary)
    compile()
    single_experiment_result = run_experiment(master_size_list)
    return single_experiment_result


def after_trial_and_error_cleanup(multiple_experiment_result):
    for key in multiple_experiment_result:
        multiple_experiment_result[key] = sorted(multiple_experiment_result[key], key=lambda x: x[0], reverse=True)

    return multiple_experiment_result

def trial_and_error(master_replace_dictionary, master_size_list):
    """
    Iterate through all sets of experiments
    :param master_replace_dictionary:
    :param master_size_list:
    :return:
    """
    size_list = []
    for key, val_list in master_replace_dictionary.items():
        size_list.append(list(range(len(val_list))))

    multiple_experiment_result = {}
    selections = list(product(*size_list))
    total_experiment_number = len(selections)
    for selection in tqdm.tqdm(selections):
        replace_dictionary = {}
        for (key, val_list), selected_idx in zip(master_replace_dictionary.items(), selection):
            replace_dictionary[key] = val_list[selected_idx]

        single_experiment_result = replace_compile_and_run(replace_dictionary, master_size_list)

        if single_experiment_result is not None:
            for size_list_str, result in single_experiment_result.items():
                multiple_experiment_result.setdefault(size_list_str, []).append((result, replace_dictionary))

    return multiple_experiment_result


def main():
    master_replace_dictionary = {
        "TAG_IMAGE_DIM1": [3, 4, 5, 6],
        "TAG_IMAGE_DIM2": [3, 4, 5, 6],
        "THREADS_TAG": [128, 192, 256],
        "OUT_CHANNELS_PER_BLOCK_TAG": [128, 192, 256],
    }

    # master_replace_dictionary = {
    #     "TAG_IMAGE_DIM1": [3, 4],
    #     "TAG_IMAGE_DIM2": [3, 4],
    #     "THREADS_TAG": [128],
    #     "OUT_CHANNELS_PER_BLOCK_TAG": [128],
    # }

    master_size_list = [
        [torch.Size([1, 32, 256, 256]), [32, 2], 1],
        [torch.Size([1, 64, 128, 128]), [64, 2], 1],
        [torch.Size([1, 128, 64, 64]), [128, 2], 1],
        [torch.Size([1, 128, 128, 128]), [128, 64], 3],
        [torch.Size([1, 130, 256, 256]), [130, 32], 3],
        [torch.Size([1, 256, 32, 32]), [256, 2], 1],
        [torch.Size([1, 256, 64, 64]), [256, 128], 3],
        [torch.Size([1, 258, 128, 128]), [258, 64], 3],
        [torch.Size([1, 512, 16, 16]), [512, 512], 3],
        [torch.Size([1, 512, 32, 32]), [512, 256], 3],
        [torch.Size([1, 514, 64, 64]), [514, 128], 3],
        [torch.Size([1, 1024, 16, 16]), [1024, 512], 3],
        [torch.Size([1, 1024, 32, 32]), [1024, 256], 3],
    ]

    multiple_experiment_result = trial_and_error(master_replace_dictionary, master_size_list)
    multiple_experiment_result_sorted = after_trial_and_error_cleanup(multiple_experiment_result)

    result_string = ""
    for each_size, result_list in multiple_experiment_result_sorted.items():
        print("")
        print(each_size)

        result_string += "\n"
        result_string += str(each_size)
        result_string += "\n"

        for result in result_list:
            print(result)

            result_string += str(result)
            result_string += "\n"

    with open("./automated_result.txt", "w") as fh:
        fh.write(result_string)

    # print(multiple_experiment_result_sorted)


if __name__=="__main__":
    main()


