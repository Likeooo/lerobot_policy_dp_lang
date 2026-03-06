# export CUDA_VISIBLE_DEVICES=3,4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

port=29609
num_gpu=8
batch_size=16
lr=1e-4

task_name=libero_object_multiview
model_tpye=dp_lang
output_dir=outputs/${model_tpye}/${task_name}_2camera_gpu_${num_gpu}_bz_${batch_size}


accelerate launch \
  --multi_gpu \
  --num_processes=${num_gpu} \
  --num_machines=1 \
  --dynamo_backend=no \
  --mixed_precision=bf16 \
  --main_process_port=${port} \
  scripts/train.py \
  --dataset.repo_id=LikeinLai/${task_name} \
  --dataset.use_imagenet_stats=false \
  --policy.type=${model_tpye} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir=${output_dir} \
  --batch_size=${batch_size} \
  --policy.optimizer_lr=${lr} \
  --policy.scheduler_name=linear \
  --steps=25000 \
  --save_freq=5000 \
  --log_freq=10 \
  --num_workers=2 \
  --wandb.enable=false
  
