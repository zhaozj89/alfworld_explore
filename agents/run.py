import os
import multiprocessing

devices = ['0', '1'] #, '2', '3']
tasks = [5, 6] #1, 2, 3, 4]
# devices = ['0']
# tasks = [1]
save_path = 'results'


def worker(device, task, save_path):
    os.system(
        'python dagger/train_dagger.py --device {:s} --task {:d} --save_path {:s} --config_file {:s}'
        .format(device, task, save_path, "config/base_config.yaml"))

for device, task in zip(devices, tasks):
    p = multiprocessing.Process(target=worker, args=(device, task, save_path))
    p.start()


# python eval/run_eval.py --task 2 --name_agent_exp test_24010_exp --name_agent_exec test_24010 --config_file config/eval_config.yam