export CUDA_VISIBLE_DEVICES=3
  
batch_size=8
lr=1e-4

task_name=libero_object_multiview
model_tpye=dp_lang
output_dir=outputs/${model_tpye}/debug_dp_lang


python -m debugpy --listen 127.0.0.1:9502 --wait-for-client \
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