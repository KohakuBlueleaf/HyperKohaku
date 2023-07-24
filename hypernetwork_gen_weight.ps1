# run train
python "./hypernetwork_gen_weight.py" `
  --pretrained_model_name_or_path "./models/kohaku-v2.1" `
  --hyperkohaku_model_path "./outputs" `
  --output_dir "./outputs" `
  --decode_iter 8 `
  --rank 1 `
  --down_dim 96 `
  --up_dim 48 `
  --reference_image_path "../datasets/celeba-hq-512x512/10312.jpg"

Write-Output "Train finished"