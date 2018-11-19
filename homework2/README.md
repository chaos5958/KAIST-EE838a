# KAIST-EE838a

# Dependent Packages

1. PyTorch (0.41)
2. imageio (2.3.0)

# How to train/test?

a. Train (CPU)
python train.py --train_dir [HDR image path] --test_dir [HDR image path]
b. Train (GPU)
python train.py --use_cuda --train_dir [HDR image path] --test_dir [HDR image path] 
- Models are saved in [model/...]

c. Test (CPU) 
python test.py --test_dir [HDR image path]
d. Test (GPU)
python test.py --use_cuda --test_dir [HDR image path]
- Images are saved in [result/...]
