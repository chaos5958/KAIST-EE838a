import os, argparse

parser = argparse.ArgumentParser(description="PyTorch Super Resolution")

parser.add_argument("--num_epoch", type=int, default=1500)
parser.add_argument("--num_batch_per_epoch", type=int, default=200)
parser.add_argument("--lr", type=int, default=5e-05)
parser.add_argument("--model_dir", type=str, default='model')
parser.add_argument("--board_dir", type=str, default='log')
parser.add_argument("--data_dir", type=str, default='data')
parser.add_argument("--gpu_idx", type=int, default=0)
parser.add_argument("--num_patch", type=int, default=20000)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--num_batch", type=int, default=16)
parser.add_argument("--load_on_memory", action='store_true')
args = parser.parse_args()

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.board_dir, exist_ok=True)
