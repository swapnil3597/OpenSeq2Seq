python scripts/decode.py \
 --logits=model_output.pickle \
 --labels="/data/librispeech/librivox-test-clean.csv" \
 --lm="language_model/4-gram.binary"  \
 --vocab="open_seq2seq/test_utils/toy_speech_data/vocab.txt" \
 --alpha=0.01 \
 --beta=0.01 \
 --beam_width=3
