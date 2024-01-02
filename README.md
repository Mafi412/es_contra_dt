# Evolution Strategy / Novelty Search / Quality-Diversity *contra* Decision Transformer

In this repository, there is a codebase (*es_dt_codebase* directory) and experiment data (*data* directory) for a master thesis by Matyáš Lorenc bearing the name *Evolutionary strategies for policy optimization in transformers*, which focuses on experiments regarding an ability of OpenAI-ES (evolution strategy algorithm), NS-ES (novelty search algorithm), and NSR-ES (quality-diversity algorithm) to train a Decision Transformer architecture to perform well in a reinforcement learning environment. The environment chosen is MuJoCo Humanoid locomotion environment. (Even though implementation even for Atari games environment is available, but the experiments were not carried out.)

More information about the algorithms, the architecture, the experiments performed, and even the motivation behind this work can be found in the text of the thesis.

## Directory *es_dt_codebase*

In es_dt_codebase directory, we can find our implementation of the three aforementioned algorithms - OpenAI-ES as es, NS-ES as ns, and NSR-ES as qd. There are scripts carrying out the training (*train_\*.py*), the scripts enabling us to simulate the trained agents (*play_\*.py*), and a script to plot the resulting logs (*plot_experiment.py*).

## Directory *data*

Here in this directory, we can find results of the experiments. This means trained models, log files, plots, and videos of the trained models in action. There are two subdirectories, *main_experiments_data* and *case_studies_data*. In the first, we can find the data for the main experiments performed, while in the second there are data of a few experiments helping us set some hyperparameters.