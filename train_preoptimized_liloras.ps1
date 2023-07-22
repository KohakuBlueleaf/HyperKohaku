# run train
python -m accelerate.commands.launch $launch_args `
  --num_processes=1 --num_cpu_threads_per_process=8 "./train_preoptimized_liloras.py" `
  --pretrained_model_name_or_path "./models/kohaku-v2.1" `
  --instance_data_dir "F:\nn\datasets\celeba-hq-512x512" `
  --instance_prompt "A [V] face" `
  --output_dir "./outputs" `
  --resolution 512 `
  --num_train_epochs 10 `
  --learning_rate 0.001 `
  --lr_scheduler constant `
  --allow_tf32 `
  --enable_xformers_memory_efficient_attention `
  --pre_compute_text_embeddings `
  --checkpoints_total_limit 2 `
  --rank 4 `
  --put_in_cpu `

Write-Output "Train finished"