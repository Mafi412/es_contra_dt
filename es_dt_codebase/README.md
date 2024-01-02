# Evolution Strategy / Novelty Search / Quality-Diversity *contra* Decision Transformer

This project contains codebase for our implementation of three distributed algorithms from the class of evolution strategies, [OpenAI-ES](https://arxiv.org/abs/1703.03864) (*es* folder), its novelty-based variant [NS-ES](https://arxiv.org/abs/1712.06560) (*ns* folder), and a basic quality-diversity version [NSR-ES](https://arxiv.org/abs/1712.06560) (*qd* folder), as well as means to analyze the data gathered during the runs.

There are also a few scripts running experiments to test those algorithms in a MuJoCo Humanoid locomotion environment on a simple feed-forward model and on a [Decision Transformer](https://arxiv.org/abs/2106.01345) architecture. Additionaly, there is a script running an experiment with Atari games environment and the Decision transformer.

For the Decision Transformer, there is a possibility to use a pretrained model to seed the search. Pretrained models can be found in folder *pretrained_ckpts*. Code used for their training can be found in *supervised_pretraining* folder.

## How-to:

### Start a training

The provided training scripts (*train_\*.py*) provide a showcase of how the training script should look like. The argumens are described in the scripts, or You can see a help by running the scripts with *-h*, or *--help* option.

### Simulate a trained agent

There are scripts for simulating the agent in the environment (*play_*.py*). For Humanoid environment, there can either be no visual output, just the return obtained and runtime of the epsiode; or there can be a video-recording of the agent's rollouts; or there can be a classical visual output. Atari environment currently does not support the recording of the rollouts. The provided scripts stand as a showcases of how to implement custom replay script.

### Perform a basic data analysis

A script *plot_experiments.py* is provided, which plots the data collected during the run of the given algorithm.

### Utilize a custom ...

#### Agent policy architecture

In *es_utilities* folder, *wrappers.py* file, there is a **EsModelWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions. The custom policy will be stored in a field *model*. There is even possibility to utilize a Virtual Batch Normalization, provided by the wrapper. Examples might be found in folder *wrapped_components*.

#### Environment

In *es_utilities* folder, *wrappers.py* file, there is a **EsEnvironmentWrapper** class, which You need to derive from. The resulting class should then override the non-implemented class functions, and even the *state_shape* property, in case the current implemantation would not return the desired state shape of the given environment. The custom environment will be stored in a field *env*. For novelty based searches, the class should be derived from a **NsEnvironmentWrapper** class present in folder *novelty_utilities*, file *ns_wrappers.py*. This adds one more function to override, returning the behavior of an agent in the environment since last reset. Examples might be found in folder *wrapped_components*.

### Create and use a behavior characteristic

To create a custom behavior characteristic, You must derive your own custom class from the **AbstractBehaviorCharacterization** class found in *novelty_utilities* folder in file *behavior.py*, again overriding its compare_to method. (Yet, ideally invoking *super().compare_to(...)* at the beginning of the custom implementation.) This behavior should then be used in the 
class derived from **NsEnvironmentWrapper**, as mentioned earlier.

## Remark to the current implementation

The current implementation uses Python *multiprocessing*, and therefore the algorithms can only be run on a single machine with multiple CPUs, it is currently not able to run on cluster. This is planned to be fixed in the future.