import os
import json
import torch
import yaml
import argparse
import importlib
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--device", help="which gpu to use [0,1,2,3]", type=str, default='0')
parser.add_argument("--task", help="which task to train [1,2,3,4,5,6]", type=int, default=3)
parser.add_argument("--name_agent_exp", help="exploration agent name", type=str, default="test_50010_exp")
parser.add_argument("--name_agent_exec", help="exploration agent name", type=str, default="test_50010")
# parser.add_argument("--name_agent_exp", help="exploration agent name", type=str, default="test_100010_exp_final")
# parser.add_argument("--name_agent_exec", help="exploration agent name", type=str, default="test_100010_final")
parser.add_argument("--log_path", type=str, default="/home/amax/zzhaoao/alfworld_explore/agents/results_alltasks10")
parser.add_argument("--config_file", help="path to config file", default="config/eval_config.yaml")
args = parser.parse_args()

os.environ['ALFRED_ROOT'] = '/home/amax/zzhaoao/alfworld_explore'
os.environ['CUDA_VISIBLE_DEVICES']=args.device
print(torch.cuda.device_count())

import sys
sys.path.insert(0, os.environ['ALFRED_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents'))

from agents.agent import TextDAggerAgent, TextDQNAgent
import agents.modules.generic as generic
from agents.eval import evaluate_dagger, evaluate_dqn
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_eval():
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    config["env"]["task_types"] = [args.task]

    config["general"]["training_method"] = "dqn"
    exp_agent = TextDQNAgent(config)
    config["general"]["training_method"] = "dagger"
    agent = TextDAggerAgent(config)

    # data_dir = "/home/amax/zzhaoao/alfworld_explore/agents/results"
    data_dir = args.log_path
    output_dir = data_dir + "/task" + str(args.task)
    # output_dir = data_dir + "/all"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load model from checkpoint
    # pdb.set_trace()
    if os.path.exists(data_dir + "/" + args.name_agent_exp + ".pt"):
        exp_agent.load_pretrained_model(data_dir + "/" + args.name_agent_exp + ".pt")
        exp_agent.update_target_net()
    else:
        raise ValueError

    if os.path.exists(data_dir + "/" + args.name_agent_exec + ".pt"):
        agent.load_pretrained_model(data_dir + "/" + args.name_agent_exec + ".pt")
        agent.update_target_net()
    else:
        raise ValueError

    training_method = config["general"]["training_method"]
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    # iterate through all environments
    for eval_env_type in eval_envs:
        # iterate through all controllers
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            print("Setting controller: %s" % controller_type)
            # iterate through all splits
            for eval_path in eval_paths:
                print("Evaluating: %s" % eval_path)
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                alfred_env = getattr(importlib.import_module("environment"), config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
                eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)

                # evaluate method
                if training_method == "dagger":
                    # pdb.set_trace()
                    results = evaluate_dagger(eval_env, agent, alfred_env.num_games*repeats, exp_agent)
                elif training_method == "dqn":
                    results = evaluate_dqn(eval_env, agent, alfred_env.num_games*repeats)
                else:
                    raise NotImplementedError()

                # save results to json
                split_name = eval_path.split("/")[-1]
                experiment_name = config["general"]["evaluate"]["eval_experiment_tag"]
                results_json = os.path.join(output_dir, "{}_{}_{}_{}.json".format(experiment_name, eval_env_type.lower(), controller_type, split_name))

                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=4, sort_keys=True)
                print("Saved %s" % results_json)

                eval_env.close()


if __name__ == '__main__':
    run_eval()
