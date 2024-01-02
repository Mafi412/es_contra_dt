import gym

from argparse import ArgumentParser
import os

from wrapped_components.env_gym_mujoco_wrappers import GymMujocoWrapper
from wrapped_components.model_dt_mujoco_wrappers import get_new_wrapped_dt_humanoid
from es_utilities.play import simulate


def main(args):
    main_seed = args.seed
    
    env = gym.make("Humanoid-v4", render_mode=("rgb_array" if args.record else (None if args.dont_show_gameplay else "human")))
    
    if args.record is not None:
        if args.record.lower() == "ckpt":
            record_folder = os.path.join(os.path.dirname(args.ckpt_path), "videos")
            video_prefix = "video"
            
        else:
            record_folder = os.path.dirname(args.record)
            video_prefix = os.path.basename(args.record)
            
        env = gym.wrappers.RecordVideo(
            env,
            record_folder,
            name_prefix=video_prefix
        )
    
    scale = 1000.
    wrapped_environment = GymMujocoWrapper(env, main_seed, scale)
    
    wrapped_model = get_new_wrapped_dt_humanoid(
        args.rtg / scale,
        wrapped_environment.timestep_limit,
        args.context_length,
        args.embed_dim,
        args.n_layer,
        args.n_head,
        args.activation_function,
        args.dropout,
        main_seed,
        None,
        None
    ) 
    wrapped_model.train(False)
    wrapped_model.load_parameters(args.ckpt_path)
    
    episode_returns, episode_lengths = simulate(wrapped_model, wrapped_environment, args.episodes)
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Checkpoint path.")
    parser.add_argument("rtg", type=float, help="Return-to-go that should be passed.")
    parser.add_argument("--context_length", type=int, default=20, help="Size of blocks (number of steps in the sequence passed to the transformer).")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("-e", "--episodes", default=1, type=int, help="Number of episodes.")
    parser.add_argument("-d", "--dont_show_gameplay", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    parser.add_argument("-r", "--record", nargs="?", type=str, default=None, const="ckpt", help="Flags whether to record a video. If no value is provided, the recording is saved into a new subfolder \"videos\" of the folder the checkpoint was loaded from. Otherwise, the given value is used as a full path the recordings are saved to.")
    
    main(parser.parse_args())
