## Meta-Reinforcement Learning for Mastering Multiple Skills and Generalizing across Environments in Text-based Games

This code repository is modified from an old version of [alfworld](https://github.com/alfworld/alfworld/tree/b14152c9241b7dc778c60c82a59bf62df76da6cf).

The pre-trained model for reproducing the experiment results can be downloaded at [here](https://drive.google.com/file/d/13UsyvYSH43CnpxISJBLyDboCBfr0h-mf/view?usp=sharing).

After setting up the development environment following [alfworld](https://github.com/alfworld/alfworld/tree/b14152c9241b7dc778c60c82a59bf62df76da6cf) (a customized [TextWorld](https://github.com/zhaozj89/TextWorld) needs to be installed) and navigating to `./agents`, we can train and evaluate the model with the following scripts. 

#### train

`python dagger/train_dagger.py --device <device_id> --task <task_id> --save_path <save_path> --config_file config/base_config.yaml`

#### evaluate

`python eval/run_eval.py --task <task_id> --name_agent_exp test_50010_exp.pt --name_agent_exec test_50010_final.pt --config_file config/eval_config.yaml`

## References

Marc-Alexandre C{\^o}t{\'e}, {\'A}kos K{\'a}d{\'a}r, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. 2018. Textworld: A learning environment for text-based games. In Computer Games Workshop at ICML/IJCAI 2018.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. ALFRED: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10740–10749.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre C{\^o}t{\'e}, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR).