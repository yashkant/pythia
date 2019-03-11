CUDA_VISIBLE_DEVICES=1 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:0.5}'
CUDA_VISIBLE_DEVICES=2,3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:true, loss:"softmaxKL", training_parameters:{max_grad_l2_norm:2.5}}'

CUDA_VISIBLE_DEVICES=1 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:false, loss:"combinedLoss", weight_complement:10.0}'


# Try SGD with relaxing the grad-norms
CUDA_VISIBLE_DEVICES=3 python train.py --config_overwrite '{data:{image_fast_reader:false}, use_complement_loss:true, loss:"softmaxKL", SGD:true, training_parameters:{lr_steps:[2000, 4000, 6000, 8000, 9000, 10000, 11000]}}'

# Build the complement weight_decay scheme! with initial_rate, decay_factor, iterations_no to decay after