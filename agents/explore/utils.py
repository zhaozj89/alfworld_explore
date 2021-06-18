import copy
import numpy as np
import os
import torch
from torch.utils import tensorboard

from agents.modules.generic import EpisodicCountingMemory, ObjCentricEpisodicMemory
import agents.modules.generic as generic


class EpisodeAndStepWriter(object):
    """Logs to tensorboard against both episode and number of steps."""

    def __init__(self, log_dir):
        self._episode_writer = tensorboard.SummaryWriter(
            os.path.join(log_dir, "episode"))

    def add_scalar(self, key, value, episode):
        self._episode_writer.add_scalar(key, value, episode)


MAX_NB_STEPS_PER_EPISODE = 10


def run_episode(env, gamefiles, agent, episode_no, problem_handler = None, update = True):
    if update==True:
        assert problem_handler is not None

    most_recent_observation_strings = None
    dqn_loss = None

    # episodic counting based memory
    episodic_counting_memory = EpisodicCountingMemory()
    obj_centric_episodic_counting_memory = ObjCentricEpisodicMemory()

    obs, infos = env.reset(gamefiles)
    batch_size = len(obs)
    if update==True:
        problem_ids = [float(item) for item in infos["extra.id"]]

    previous_dynamics = None

    chosen_actions = []
    prev_step_dones, prev_rewards = [], []
    for _ in range(batch_size):
        chosen_actions.append("restart")
        prev_step_dones.append(0.0)
        prev_rewards.append(0.0)

    observation_strings = list(obs)
    task_desc_strings, observation_strings = agent.get_task_and_obs(
        observation_strings)
    task_desc_strings = agent.preprocess_task(task_desc_strings)
    observation_strings = agent.preprocess_observation(observation_strings)

    first_sight_strings = copy.deepcopy(observation_strings)
    agent.observation_pool.push_first_sight(first_sight_strings)

    action_candidate_list = list(infos["admissible_commands"])
    action_candidate_list = agent.preprocess_action_candidates(
        action_candidate_list)
    observation_only = observation_strings
    # appending the chosen action at previous step into the observation
    observation_strings = [item + " [SEP] " + a for item,
                           a in zip(observation_strings, chosen_actions)]
    # update init observation into memory
    episodic_counting_memory.push(observation_only)
    obj_centric_episodic_counting_memory.push(observation_only)

    transition_cache = []
    still_running_mask = []
    sequence_game_rewards, sequence_count_rewards, sequence_novel_object_rewards = [], [], []
    print_actions = []

    traj_embeddings = None
    # here I only use 10 steps for efficiency
    for step_no in range(MAX_NB_STEPS_PER_EPISODE):
        agent.observation_pool.push_batch(observation_strings)
        most_recent_observation_strings = agent.observation_pool.get()

        # if agent.noisy_net:
        #     agent.reset_noise()
        # let's first try selecting admissible acts
        chosen_actions, chosen_indices, current_dynamics = agent.admissible_commands_act(
            most_recent_observation_strings, task_desc_strings, action_candidate_list, previous_dynamics, random=False)

        # trajectory encoding
        traj_embeddings = agent.traj_encoder(
            most_recent_observation_strings, task_desc_strings, traj_embeddings)

        replay_info = [most_recent_observation_strings,
                       task_desc_strings, action_candidate_list, chosen_indices]
        transition_cache.append(replay_info)
        obs, _, dones, infos = env.step(chosen_actions)
        scores = [float(item) for item in infos["won"]]
        dones = [float(item) for item in dones]

        if update==True:
            with torch.no_grad():
                problem_embeddings = problem_handler.get_problem_embeddings(problem_ids)
                similarity_scores = torch.norm(traj_embeddings-problem_embeddings, dim=1)
            similarity_scores_list = similarity_scores.detach().cpu().tolist()
            scores = [0.5*score + 0.5*similarity_scores_list[i] for i, score in enumerate(scores)]

        observation_strings = list(obs)
        observation_strings = agent.preprocess_observation(observation_strings)

        action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(
            action_candidate_list)
        observation_only = observation_strings
        # appending the chosen action at previous step into the observation
        observation_strings = [item + " [SEP] " + a for item,
                               a in zip(observation_strings, chosen_actions)]
        seeing_new_states = episodic_counting_memory.is_a_new_state(
            observation_only)
        seeing_new_objects = obj_centric_episodic_counting_memory.get_object_novelty_reward(
            observation_only)
        # update new observation into memory
        episodic_counting_memory.push(observation_only)
        obj_centric_episodic_counting_memory.push(
            observation_only)  # update new observation into memory
        previous_dynamics = current_dynamics

        # if problem_ids is not None:
        #     # if step_no % agent.update_per_k_game_steps == 0:
        #     dqn_loss, _ = agent.update_dqn(problem_handler)
        if update==True:
            dqn_loss, _ = agent.update_dqn(problem_handler)

        if step_no == MAX_NB_STEPS_PER_EPISODE - 1:
            dones = [1.0 for _ in dones]

        still_running = [1.0 - float(item)
                         for item in prev_step_dones]  # list of float
        prev_step_dones = dones
        step_rewards = [float(curr) - float(prev) for curr,
                        prev in zip(scores, prev_rewards)]  # list of float
        # list of float
        count_rewards = [
            r * agent.count_reward_lambda for r in seeing_new_states]
        # list of novel object rewards
        novel_object_rewards = [
            r * agent.novel_object_reward_lambda for r in seeing_new_objects]
        prev_rewards = scores
        still_running_mask.append(still_running)
        sequence_game_rewards.append(step_rewards)
        sequence_count_rewards.append(count_rewards)
        sequence_novel_object_rewards.append(novel_object_rewards)
        print_actions.append(chosen_actions[0] if still_running[0] else "--")

        if np.sum(still_running) == 0:
            break

    still_running_mask_np = np.array(still_running_mask)
    game_rewards_np = np.array(sequence_game_rewards) * \
        still_running_mask_np  # step x batch
    count_rewards_np = np.array(
        sequence_count_rewards) * still_running_mask_np  # step x batch
    novel_object_rewards_np = np.array(
        sequence_novel_object_rewards) * still_running_mask_np
    game_rewards_pt = generic.to_pt(
        game_rewards_np, enable_cuda=False, type='float')  # step x batch
    count_rewards_pt = generic.to_pt(
        count_rewards_np, enable_cuda=False, type='float')  # step x batch
    novel_object_rewards_pt = generic.to_pt(
        novel_object_rewards_np, enable_cuda=False, type='float')

    if update==True:
        # push experience into replay buffer (dqn)
        avg_reward_in_replay_buffer = agent.dqn_memory.get_avg_rewards()
        for b in range(game_rewards_np.shape[1]):
            if still_running_mask_np.shape[0] == MAX_NB_STEPS_PER_EPISODE and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                avg_reward = game_rewards_np[:, b].tolist() + [0.0]
                _need_pad = True
            else:
                avg_reward = game_rewards_np[:, b]
                _need_pad = False
            avg_reward = np.mean(avg_reward)
            is_prior = avg_reward >= avg_reward_in_replay_buffer

            mem = []
            for i in range(game_rewards_np.shape[0]):
                observation_strings, task_strings, action_candidate_list, chosen_indices = transition_cache[
                    i]
                mem.append([observation_strings[b],
                            task_strings[b],
                            action_candidate_list[b],
                            chosen_indices[b],
                            game_rewards_pt[i][b], count_rewards_pt[i][b], novel_object_rewards_pt[i][b], problem_ids[b]])
                if still_running_mask_np[i][b] == 0.0:
                    break
            if _need_pad:
                observation_strings, task_strings, action_candidate_list, chosen_indices = transition_cache[-1]
                mem.append([observation_strings[b],
                            task_strings[b],
                            action_candidate_list[b],
                            chosen_indices[b],
                            game_rewards_pt[-1][b] * 0.0, count_rewards_pt[-1][b] * 0.0, novel_object_rewards_pt[-1][b] * 0.0, problem_ids[b]])
            agent.dqn_memory.push(is_prior, avg_reward, mem)

    # finish game, maybe find a better time to update
    agent.finish_of_episode(episode_no, batch_size)

    return most_recent_observation_strings, dqn_loss
