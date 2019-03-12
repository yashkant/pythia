# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.7}'
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:true, loss:"softmaxKL", training_parameters:{complement_max_grad_l2_norm:0.025}}'
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:1.0}'

# CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:16.0, weight_complement_decay:true, weight_complement_decay_factor:0.5, weight_complement_decay_iters:2000}'

CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:1.0, weight_complement_decay:true, weight_complement_decay_factor:0.5, weight_complement_decay_iters:1000}'

CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_softmax:0.0, weight_complement:1.0, weight_complement_decay:true, weight_complement_decay_factor:0.1, weight_complement_decay_iters:2000}'
CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_softmax:0.0, weight_complement:1000000.0, weight_complement_decay:true, weight_complement_decay_factor:0.1, weight_complement_decay_iters:2000}'
