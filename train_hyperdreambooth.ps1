# run train
python -m accelerate.commands.launch $launch_args `
  --num_processes=1 --num_cpu_threads_per_process=8 "./train_hyperdreambooth.py" `
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
  --instance_data_dir "F:\nn\datasets\celeba-hq-512-4.8k" `
  --instance_prompt "A [V] face" `
  --output_dir "./outputs" `
  --resolution 512 `
  --learning_rate 0.0001 `
  --lr_scheduler cosine `
  --lr_warmup_steps 100 `
  --allow_tf32 `
  --enable_xformers_memory_efficient_attention `
  --pre_compute_text_embeddings `
  --checkpoints_total_limit 2 `
  --checkpointing_steps 500 `
  --decoder_blocks 8 `
  --rank 1 `
  --down_dim 96 `
  --up_dim 48 `
  --train_batch_size 8 `
  --mixed_precision "bf16" `
  --pre_opt_weight_path "./outputs" `
  --num_train_epochs 10 `
  --report_to "wandb" `

Write-Output "Train finished"