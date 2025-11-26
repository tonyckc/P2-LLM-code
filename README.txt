#Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need

1. Install the packages using requirements file

2. dataset: DIV2K validation set:
   finetuning training set : /root/autodl-tmp/benchmark/DIV2K-test/large_data_train_p16_half_full.json
   finetuning validation set: /root/autodl-tmp/benchmark/DIV2K-test/large_data_valid_p16_half_full.json

   To download dataset, please login my autoDL account (user:18080556093 password: ckc2015ji) and the used machine (A800 / 017机03dd44a781-c9bf73c8)

3. Pretrained model: Llama model 3.1-8B at '/root/autodl-tmp/cache/huggingface/hub/' (corresponding parser.add_argument("--cache_dir", default='/root/autodl-tmp/cache/huggingface/hub/'))
   To download model, you can also login my autoDL account


3. You should set the DeepSpeed for acceleration, at Shell window for setting:
   ❯ accelerate config                                                                           ─╯
        ----------------------------In which compute environment are you running?
        Please select a choice using the arrow or number keys, and selecting with enter
         ➔  This machine
            AWS (Amazon SageMaker)
        ----------------------------Which type of machine are you using?
        Please select a choice using the arrow or number keys, and selecting with enter
            No distributed training
            multi-CPU
            multi-XPU
        ➔  multi-GPU
            multi-NPU
            TPU
        How many different machines will you use (use more than 1 for multi-node training)? [1]: 1  #
        Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no
        Do you wish to optimize your script with torch dynamo?[yes/NO]:no
        Do you want to use DeepSpeed? [yes/NO]: yes
        Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no
        ----------------------------What should be your DeepSpeed's ZeRO optimization stage?
        Please select a choice using the arrow or number keys, and selecting with enter
            0
        ➔  1
            2
            3

        How many gradient accumulation steps you're passing in your script? [1]: 1  #
        Do you want to use gradient clipping? [yes/NO]: no  #
        Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: no
        Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no
        How many GPU(s) should be used for distributed training? [1]: YOUR MACHINE  #
        ----------------------------Do you wish to use FP16 or BF16 (mixed precision)?
        Please select a choice using the arrow or number keys, and selecting with enter
            no  #
        ➔  fp16  # fp16,
            bf16  # bf16,
            fp8
        accelerate configuration saved at /home/variantconst/.cache/huggingface/accelerate/default_config.yaml


4. run accelerate launch test_Lora_finetuning_llama.py
