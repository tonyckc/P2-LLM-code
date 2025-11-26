
# Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need


This is the official PyTorch implementation of our paper:

[Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need](https://openreview.net/pdf/e583c137b82a12d3f190f7cfbf7bd07f69b6c559.pdf). 

If you have any problems, please email me (ck.ee@my.cityu.edu.hk).


# Usage
1. Install the packages using requirements file

2. dataset: DIV2K validation set:
   finetuning training set : ./benchmark/DIV2K-test/large_data_train_p16_half_full.json
   finetuning validation set: ./benchmark/DIV2K-test/large_data_valid_p16_half_full.json

3. Pretrained model: Llama model 3.1-8B at './cache/huggingface/hub/' (corresponding parser.add_argument("--cache_dir", default='./cache/huggingface/hub/'))

4. You should set the DeepSpeed for acceleration

5. run accelerate launch test_Lora_finetuning_llama.py


