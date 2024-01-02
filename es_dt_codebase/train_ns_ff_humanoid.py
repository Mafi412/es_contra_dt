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

from wrapped_components.model_ff_mujoco_wrappers import get_new_wrapped_ff_humanoid
from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapperNsEndXY
from ns.ns import ns

import gym

from argparse import ArgumentParser


def main(args):
    # Gym does sometimes output those, but we don"t really care, they just clog our output.
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    main_seed = args.seed
    metapopulation_size = args.metapopulation_size
    env = gym.make("Humanoid-v4", render_mode=None)
    test_environment = GymMujocoWrapperNsEndXY(env, main_seed, 1000.)
    models = [get_new_wrapped_ff_humanoid(main_seed + i if main_seed is not None else None, args.optimizer, args.learning_rate) for i in range(metapopulation_size)]
    num_of_workers = args.num_of_workers
    size_of_population = args.size_of_population
    num_of_iterations = args.num_of_iterations
    noise_deviation = args.noise_deviation
    weight_decay_factor = args.weight_decay_factor
    batch_size = args.batch_size
    update_vbn_stats_probability = args.update_vbn_stats_probability
    path_for_checkpoints = args.path_for_checkpoints
    logging_path = args.logging_path
    max_archive_size = args.max_archive_size
    num_of_nearest_neighbors_to_average = args.num_of_nearest_neighbors_to_average
    
    resulting_models = ns(
        models,
        test_environment,
        num_of_workers,
        size_of_population,
        num_of_iterations,
        main_seed,
        max_archive_size,
        num_of_nearest_neighbors_to_average,
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
    parser.add_argument("--num_of_workers", type=int, default=1, help="Number of worker processes to be launched.")
    parser.add_argument("--size_of_population", type=int, default=5000, help="Size of population, or more precisely number of tested noises within one generation / iteration.")
    parser.add_argument("--metapopulation_size", type=int, default=5, help="Size of metapopulation, or more precisely number of base models from which we choose, which one to update.")
    parser.add_argument("--num_of_iterations", type=int, default=600, help="Number of iterations the ES will run.")
    parser.add_argument("--seed", type=int, default=None, help="Main seed.")
    parser.add_argument("--noise_deviation", type=float, default=0.02, help="Deviation of the noise added during training.")
    parser.add_argument("--weight_decay_factor", type=float, default=0.995, help="Factor of the weight decay.")
    parser.add_argument("--batch_size", type=int, default=1000, help="A size of a batch for a batched weighted sum of noises during model update.")
    parser.add_argument("--update_vbn_stats_probability", type=float, default=0.01, help="How often to use data obtained during evaluation to update the Virtual Batch Norm stats.")
    parser.add_argument("--optimizer", type=str, default="SGDM", help="Optimizer to be used. Either \"ADAM\", or \"SGDM\" (standing for SGD with Momentum), or \"SGD\".")
    parser.add_argument("--learning_rate", type=float, default=5e-2, help="Learning rate (or could be called step size).")
    parser.add_argument("--max_archive_size", type=int, default=None, help="Maximal size of the archive for novelty computation.")
    parser.add_argument("--num_of_nearest_neighbors_to_average", type=int, default=10, help="Number of nearest neighbors in archive that will be considered and distance to them averaged when computing the novelty.")
    
    main(parser.parse_args())
