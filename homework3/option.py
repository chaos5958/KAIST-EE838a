import sys, os, logging, argparse

parser = argparse.ArgumentParser(description="PyTorch Super Resolution")

parser.add_argument("--use_cuda", action="store_true", help='Use GPUs')

parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--epoch", type=int, default=100, help="Epoch")
parser.add_argument("--num_batch_train", type=int, default=2, help="trainig_batch_size")
parser.add_argument("--num_batch_test", type=int, default=1, help="test_batch_size")

parser.add_argument("--model_dir", type=str, default='model', help="Model directory")
parser.add_argument("--result_dir", type=str, default='result', help="Result directory")
parser.add_argument("--train_dir", type=str, default='train', help="Train HDR image directory")
parser.add_argument("--test_dir", type=str, default='test', help="Test HDR image directory")
parser.add_argument("--model_name", type=str, default='', help="Model name")

opt = parser.parse_args()

os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.result_dir, exist_ok=True)
