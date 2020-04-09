python run.py \
 --mode=infer \
 --config="example_configs/speech2text/jasper10x5_LibriSpeech_nvgrad_masks.py" \
 --logdir="../checkpoint/" \
 --num_gpus=1 \
 --use_horovod=False \
 --decoder_params/use_language_model=False \
 --infer_output_file=model_output.pickle

python3 calculate_wer.py
