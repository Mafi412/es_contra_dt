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

from wrapped_components.model_dt_mujoco_wrappers import get_new_wrapped_dt_humanoid
from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapper
from es.es import es

import gym

from argparse import ArgumentParser


def main(args):
    # Gym does sometimes output those, but we don"t really care, they just clog our output.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    main_seed = args.seed   
    env = gym.make("Humanoid-v4", render_mode=None)
    scale = 1000.
    test_environment = GymMujocoWrapper(env, main_seed, scale)
    model = get_new_wrapped_dt_humanoid(
        args.rtg / scale,
        test_environment.timestep_limit,
        args.context_length,
        args.embed_dim,
        args.n_layer,
        args.n_head,
        args.activation_function,
        args.dropout,
        args.fixed_ln,
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
    parser.add_argument("--size_of_population", type=int, default=10000, help="Size of population, or more precisely number of tested noises within one generation / iteration.")
    parser.add_argument("--num_of_iterations", type=int, default=140, help="Number of iterations the ES will run.")
    parser.add_argument("--seed", type=int, default=None, help="Main seed.")
    parser.add_argument("--noise_deviation", type=float, default=0.02, help="Deviation of the noise added during training.")
    parser.add_argument("--weight_decay_factor", type=float, default=0.995, help="Factor of the weight decay.")
    parser.add_argument("--batch_size", type=int, default=1000, help="A size of a batch for a batched weighted sum of noises during model update.")
    parser.add_argument("--update_vbn_stats_probability", type=float, default=0.01, help="How often to use data obtained during evaluation to update the Virtual Batch Norm stats.")
    parser.add_argument("--optimizer", type=str, default="SGDM", help="Optimizer to be used. Either \"ADAM\", or \"SGDM\" (standing for SGD with Momentum), or \"SGD\".")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate (or can be called step size).")
    parser.add_argument("--load_model", type=str, default=None, help="Path from which to load the weights (and possibly vbn stats) into the model.")
    parser.add_argument("--context_length", type=int, default=20, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fixed_ln", action="store_true", help="Fixes the LayerNorm layers so that they are not further trained by the algorithm.")
    
    main(parser.parse_args())
