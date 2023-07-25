# run train
python "./hypernetwork_gen_weight.py" `
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
  --hyperkohaku_model_path "./outputs" `
  --output_dir "./outputs" `
  --decode_iter 12 `
  --rank 1 `
  --down_dim 96 `
  --up_dim 48 `
  --reference_image_path "../datasets/celeba-hq-512x512/10472.jpg"

Write-Output "Train finished"