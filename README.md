[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/documentation-github.io-blue.svg)](https://nvidia.github.io/OpenSeq2Seq/html/index.html)
<div align="center">
  <img src="./docs/logo-shadow.png" alt="OpenSeq2Seq" width="250px">
  <br>
</div>

# OpenSeq2Seq: toolkit for distributed and mixed precision training of sequence-to-sequence models

OpenSeq2Seq main goal is to allow researchers to most effectively explore various
sequence-to-sequence models. The efficiency is achieved by fully supporting
distributed and mixed-precision training.
OpenSeq2Seq is built using TensorFlow and provides all the necessary
building blocks for training encoder-decoder models for neural machine translation, automatic speech recognition, speech synthesis, and language modeling.

## Documentation and installation instructions 
https://nvidia.github.io/OpenSeq2Seq/

# Instructions for running Inference from this forked Repository

### Step 1: Docker setup
First follow the step from [this](https://nvidia.github.io/OpenSeq2Seq/html/installation.html) link to run the container which is required to run the OpenSeq2Seq Toolkit. If you are using VM Instance make sure you are using GPU instance with P100 or V100.

### Step 2: Keep docker running for development environment

### Step 3:Installing OpenSeq2Seq for Speech Recognition
Install requirements:
```bash
git clone https://github.com/swapnil3597/OpenSeq2Seq/
cd OpenSeq2Seq
pip install -r requirements.txt
```
Install CTC decoders:
```bash
bash scripts/install_decoders.sh
python scripts/ctc_decoders_test.py
```
All these above intructions are also available [here](https://nvidia.github.io/OpenSeq2Seq/html/installation.html)

### Step 4: Downloading Acoustic model(Jasper) and Language Model
Find the links for latest **Acoustic model checkpoint and config file** from [here](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition.html)

To download from drive link follow these commands:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=12CQvNrTvf0cjTsKjbaWWvdaZb7RxWI6X&export=download # This is an example, use the latest drive link for Jasper checkpoint
```

To download the **Language model** follow these steps:
```bash
bash scripts/install_kenlm.sh
bash scripts/download_lm.sh
```
After running this command a `language_model/` dir would be created containing the binary file for 4-gram ARPA language model.

### Step 5: Running Inference:

First in `run_inference.sh` script make sure you provide the correct path for `--config` and `--logdir` for Acoustic model (Jasper). 

There are two ways to run inference:

**1. With Greedy Decoder:** 
Make sure that in config file `"decoder_params"` section has `'infer_logits_to_pickle': False` line and that `"dataset_files"` field of `"infer_params"` section contains a target CSV file. Then run:
```bash
bash run_inference.sh # You will get desired output in model_output.pickle file
```
**1. With Language Model Rescoring:** 
In the file `run_decoding.sh` provide the correct binary file path for language model in `--lm` and
make sure that in config file `"decoder_params"` section has `'infer_logits_to_pickle': True` line and that `"dataset_files"` field of `"infer_params"` section contains a target CSV file. Then run:
```bash
bash run_inference.sh # You will get acoustic model logits in model_output.pickle file
# To decode the logits run:
bash run_decoding.sh
# For --mode as 'infer' you will get output in --infer_output_file 'inference_output_lm.csv'
```

## Features
1. Models for:
   1. Neural Machine Translation
   2. Automatic Speech Recognition
   3. Speech Synthesis
   4. Language Modeling
   5. NLP tasks (sentiment analysis)
2. Data-parallel distributed training
   1. Multi-GPU
   2. Multi-node
3. Mixed precision training for NVIDIA Volta/Turing GPUs

## Software Requirements
1. Python >= 3.5
2. TensorFlow >= 1.10
3. CUDA >= 9.0, cuDNN >= 7.0 
4. Horovod >= 0.13 (using Horovod is not required, but is highly recommended for multi-GPU setup)

## Acknowledgments
Speech-to-text workflow uses some parts of [Mozilla DeepSpeech](https://github.com/Mozilla/DeepSpeech) project.

Beam search decoder with language model re-scoring implementation (in `decoders`) is based on [Baidu DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).

Text-to-text workflow uses some functions from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).

## Disclaimer
This is a research project, not an official NVIDIA product.

## Related resources
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT](http://opennmt.net/)
* [Neural Monkey](https://github.com/ufal/neuralmonkey)
* [Sockeye](https://github.com/awslabs/sockeye)
* [TF-seq2seq](https://github.com/google/seq2seq)
* [Moses](http://www.statmt.org/moses/)

## Paper
If you use OpenSeq2Seq, please cite [this paper](https://arxiv.org/abs/1805.10387)
```
@misc{openseq2seq,
    title={Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq},
    author={Oleksii Kuchaiev and Boris Ginsburg and Igor Gitman and Vitaly Lavrukhin and Jason Li and Huyen Nguyen and Carl Case and Paulius Micikevicius},
    year={2018},
    eprint={1805.10387},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
