SAMPLE_FLAGS="--num_samples 2000  --dataset voronoi --batch_size 8  --set_name test --use_image_features False"
CUDA_VISIBLE_DEVICES='0' python image_sample.py  --set_name test --model_path ckpts/preds/model002000.pt  $SAMPLE_FLAGS
