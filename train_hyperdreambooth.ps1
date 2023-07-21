# run train
python -m accelerate.commands.launch $launch_args `
  --num_processes=1 --num_cpu_threads_per_process=8 "./train_hyperdreambooth.py" `
  --pretrained_model_name_or_path "./models/kohaku-v2.1" `
  --instance_data_dir "F:\nn\datasets\celeba-hq-512x512" `
  --instance_prompt "A [V] face" `
  --output_dir "./outputs" `
  --resolution 512 `
  --num_train_epochs 10 `
  --gradient_checkpointing `
  --learning_rate 0.00002 `
  --lr_scheduler cosine `
  --lr_warmup_steps 100 `
  --use_8bit_adam `
  --allow_tf32 `
  --enable_xformers_memory_efficient_attention `
  --pre_compute_text_embeddings `

Write-Output "Train finished"