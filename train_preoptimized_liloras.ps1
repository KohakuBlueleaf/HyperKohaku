# run train
python -m accelerate.commands.launch $launch_args `
  --num_processes=1 --num_cpu_threads_per_process=8 "./train_preoptimized_liloras.py" `
  --pretrained_model_name_or_path "./models/kohaku-v2.1" `
  --instance_data_dir "F:\nn\datasets\celeba-hq-512-4.8k" `
  --instance_prompt "A [V] face" `
  --output_dir "./outputs" `
  --resolution 512 `
  --learning_rate 0.01 `
  --lr_scheduler constant `
  --allow_tf32 `
  --enable_xformers_memory_efficient_attention `
  --pre_compute_text_embeddings `
  --checkpoints_total_limit 2 `
  --checkpointing_steps 100 `
  --train_steps_per_identity 10 `
  --rank 1 `
  --down_dim 96 `
  --up_dim 48 `
  --train_batch_size 12 `
  --put_in_cpu `
  --mixed_precision "bf16" `

Write-Output "Train finished"