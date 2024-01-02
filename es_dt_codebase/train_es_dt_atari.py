# Forbid multithreading for Numpy.
import os
N_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS

# Forbid multithreading for PyTorch.
import torch
torch.set_num_threads(1)

from wrapped_components.model_dt_atari_wrappers import get_new_wrapped_dt_for_ale_environment
from wrapped_components.env_ale_atari_wrappers import ALEAtariWrapper
from es.es import es

from components.ale_atari_env.ale_env import ALEModern

from argparse import ArgumentParser
import random


def main(args):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    main_seed = args.seed
    game = args.game.replace(" ", "")
    env = ALEModern(
        game,
        random.randint(0, 100_000) if main_seed is None else main_seed,
        torch.device("cpu"),
        clip_rewards_val=False,
        sticky_action_p=0,
        sdl=False
    )
    test_environment = ALEAtariWrapper(env, main_seed)
    model = get_new_wrapped_dt_for_ale_environment(
        env,
        args.rtg,
        not args.dont_sample_action,
        args.context_length,
        int(1e4),
        main_seed,
        args.optimizer,
        args.learning_rate
    )
    num_of_workers = args.num_of_workers
    size_of_population = args.size_of_population
    num_of_iterations = args.num_of_iterations
    noise_deviation = args.noise_deviation
    weight_decay_factor = args.weight_decay_factor
    batch_size = args.batch_size
    update_vbn_stats_probability = args.update_vbn_stats_probability
    path_for_checkpoints = args.path_for_checkpoints
    logging_path = args.logging_path
    
    if args.load_model is not None:
        model.load_parameters(args.load_model)
    
    resulting_model = es(
        model,
        test_environment,
        num_of_workers,
        size_of_population,
        num_of_iterations,
        main_seed,
        noise_deviation,
        weight_decay_factor,
        batch_size,
        update_vbn_stats_probability,
        path_for_checkpoints,
        logging_path
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_for_checkpoints", type=str, help="Base path, from which the paths to the individual checkpoint files will be created.")
    parser.add_argument("logging_path", type=str, help="Base path, from which the paths to files, where fitnesses, runtimes and wall-clock times (and evaluation results) will be logged, will be constructed.")
    parser.add_argument("rtg", type=float, help="Return-to-go that should be passed.")
    parser.add_argument("--num_of_workers", type=int, default=1, help="Number of worker processes to be launched.")
    parser.add_argument("--size_of_population", type=int, default=20000, help="Size of population, or more precisely number of tested noises within one generation / iteration.")
    parser.add_argument("--num_of_iterations", type=int, default=120, help="Number of iterations the ES will run.")
    parser.add_argument("--seed", type=int, default=None, help="Main seed.")
    parser.add_argument("--noise_deviation", type=float, default=0.02, help="Deviation of the noise added during training.")
    parser.add_argument("--weight_decay_factor", type=float, default=0.995, help="Factor of the weight decay.")
    parser.add_argument("--batch_size", type=int, default=100, help="A size of a batch for a batched weighted sum of noises during model update.")
    parser.add_argument("--update_vbn_stats_probability", type=float, default=0.01, help="How often to use data obtained during evaluation to update the Virtual Batch Norm stats.")
    parser.add_argument("--optimizer", type=str, default="SGDM", help="Optimizer to be used. Either \"ADAM\", or \"SGDM\" (standing for SGD with Momentum), or \"SGD\".")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate (or can be called step size).")
    parser.add_argument("--load_model", type=str, default=None, help="Path from which to load the weights (and possibly vbn stats) into the model.")
    parser.add_argument("--context_length", type=int, default=30, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument("--game", type=str, default="Star Gunner")
    parser.add_argument("--dont_sample_action", action="store_true")
    
    main(parser.parse_args())
