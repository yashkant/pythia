# CUDA_VISIBLE_DEVICES=0 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.3}'
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:true, loss:"softmaxKL"}'
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:true, loss:"softmaxKL", hard_scores:false}'

CUDA_VISIBLE_DEVICES=0,1 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:10.0, weight_complement_decay:true, weight_complement_decay_factor:0.1, weight_complement_decay_iters:2000}'
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_softmax:0.0, weight_complement:1000.0, weight_complement_decay:true, weight_complement_decay_factor:0.1, weight_complement_decay_iters:2000}'


# Next to run:

# CUDA_VISIBLE_DEVICES=0 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.01}'
# CUDA_VISIBLE_DEVICES=0 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.05}'
# CUDA_VISIBLE_DEVICES=0 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.005}'

