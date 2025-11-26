import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = './cache/'
import cv2
import torch
import os
import pickle
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from skimage import transform as tra
from skimage import io
import numpy as np
import gc
import time
#from unsloth import FastLanguageModel
from omegaconf import OmegaConf
from PIL import Image
from typing import Callable,Union,Iterator
from utils_trainer import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    PeftSavingCallback,
    neftune_post_forward_hook,
)
from huggingface_hub import login
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from io import BytesIO
from PIL import Image
from copy import copy
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from glob import glob
from tqdm import tqdm
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling,pipeline
import copy
from utils import *
from arithmetic_coder import *
import torchvision.transforms as transforms
from utils import unpickle
import imagecodecs
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from datasets import load_dataset
import polars as pl
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    PretrainedConfig,
)
import wandb
from trl import setup_chat_format,SFTTrainer,SFTConfig
import bitsandbytes as bnb



def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn
def process_string(data):
    """
    :param data: numpy array
    :return: array to string
    """
    data_string = ''.join([str(data[i]) for i in range(data.shape[0])]).replace("[ ", "[").replace(" ]", "]")
    return data_string
def process_string_batch(batch_data):
        data_string = [''.join([str(data[i]) for i in range(data.shape[0])]).replace("[ ", "[").replace(" ]", "]") for data in batch_data]
        return data_string
def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16 bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)


def load_imagenet32(dir_root):
    data_dir = glob(os.path.join(dir_root, '*'))
    data = np.concatenate([unpickle(dir_)['data'] for dir_ in data_dir], axis=0)
    return data
def compute_pixel_token_ids(tokenizer):
    '''
    :return: 0-255 token id in LLM vocabunary
    '''
    pixel_value = list(range(256))
    pixel_token_ids = [tokenizer.encode(str(value))[1] for value in pixel_value]
    return pixel_token_ids
def split_patch(array,flatten=True):
    # Patch size
    patch_height, patch_width = 32, 32

    # Calculate the number of patches along each dimension
    num_patches_y = array.shape[0] // patch_height
    num_patches_x = array.shape[1] // patch_width

    # List to store patches
    patches = []

    # Extract patches
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = array[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width, :]
            if flatten:
                patch = np.reshape(patch,[-1,3])
            patches.append(patch)

    # Convert list of patches to a NumPy array
    patches_array = np.array(patches)
    return patches_array


def test_compress_dataset(dir=None, type='llm'):
    images = glob(dir + '/*.png')
    compression_ratio_total = 0
    bpsp_total = 0
    if type == 'llm' or 'ED':
        tokenizer, model = LLM_initiation()
    for image_path in tqdm(images,desc='process image'):
        image = Image.open(image_path)
        image = np.array(image)
        image_patch = split_patch(image,flatten=True)
        if type=='llm':
            compression_data_per_RGB = 0
            channel = ['R','G','B']
            for idx,ch in enumerate(channel):
                image_patch_channel = image_patch[:, :, idx]
                image_patch_channel = add_prompt_label(image_patch_channel, channel=ch)
                _, compression_data_per = test_compress_per_image(tokenizer, model,image_patch_channel,N=image_patch.shape[0],bpsp_or_compression_ratio=2)
                compression_data_per_RGB += compression_data_per
            bpsp_total += (compression_data_per_RGB*8)/(np.prod(image.shape))
            compression_ratio_total += np.prod(image.shape)/(compression_data_per_RGB)
        elif type == 'jpeg-xl':
            compression_ratio_total += test_compress_jpegxl_per_image(image_patch_channel,N=image_patch.shape[0],type='1D')
        elif type == 'ED':
            compress_and_decompress(tokenizer, model,image_patch_channel,N=image_patch.shape[0],bpsp_or_compression_ratio=2)
    print('{} average compression ratio:{}'.format(type, compression_ratio_total/len(images)))
    print('{} average bpsp:{}'.format(type, bpsp_total / len(images)))


def test_compress_jpegxl_per_image(training_data_context,N,type='1D'):
    if type == '1D':
        image = training_data_context[:, :-1].reshape([-1, ]).reshape([-1, 1, 1]).astype(np.uint8)
        compressed_image = imagecodecs.jpegxl_encode(image, lossless=True)
        print('1D-chunk=overall sequence, jpeg xl compression ratio (raw size/compressed size): ', (N * 1024)/len(compressed_image))
        return (N * 1024)/len(compressed_image)
    elif type== '2D':

        image = training_data_context[:, :-1]
        total_compressed = 0
        for idx in range(image.shape[0]):
            data = image[idx].reshape([32, 32, 1]).astype(np.uint8)
            compressed_image = imagecodecs.jpegxl_encode(data, lossless=True)
            total_compressed += len(compressed_image)
        print('2D-32*32, jpeg xl compression ratio (raw size/compressed size): ',
              (N * 1024) / total_compressed)
        return (N * 1024)/total_compressed,total_compressed


def test_compress_traditional(args):
    compression_type = args.t_type
    N = args.test_size
    w,h = 32,32
    num_pixel = w*h
    config = {'validate_size': N, 'training_size': N}
    training_data_context_all, training_data_context_all = load_dataset_numpy(None, config)
    if args.test_ch == 'R':
        training_data_context = training_data_context_all[:N, :]
    elif args.test_ch =='G':
        training_data_context = training_data_context_all[N:N*2, :]
    elif args.test_ch == 'B':
        training_data_context = training_data_context_all[N*2:, :]
    print(training_data_context.shape)

    '''
    #############################################################################################
    image = training_data_context[:, :-1].reshape([-1, ]).reshape([-1, 1, 1]).astype(np.uint8)
    if compression_type == 'jpegxl':
        compressed_image = imagecodecs.jpegxl_encode(image, lossless=True)
    elif compression_type == 'webp':
        compressed_image = imagecodecs.webp_encode(image, lossless=True)
    #print(compressed_image)
    print(N,' imagenet-1k 32x32 images, 1D-chunk=overall sequence, jpeg xl compression ratio (raw size/compressed size): ', (N * 1024)/len(compressed_image))

    image = training_data_context[:, :-1]
    total_compressed  = 0
    for idx in range(image.shape[0]):
        data = image[idx].reshape([num_pixel,1,1]).astype(np.uint8)
        if compression_type == 'jpegxl':
            compressed_image = imagecodecs.jpegxl_encode(image, lossless=True)
        elif compression_type == 'webp':
            compressed_image = imagecodecs.webp_encode(image, lossless=True)
        total_compressed += len(compressed_image)
    print(N, ' imagenet-1k 32x32 images , 1D-chunk=1024,jpeg xl compression ratio (raw size/compressed size): ',
          (N * num_pixel) / total_compressed)
    '''
    if compression_type != 'webp': # not support signle channel
        image = training_data_context[:, :-1].astype(np.uint8)
        total_compressed = 0
        for idx in tqdm(range(image.shape[0]),desc='compress ...'):
            data = image[idx].reshape([w, h, 1]).astype(np.uint8)
            if compression_type == 'jpegxl':
                compressed_image = imagecodecs.jpegxl_encode(data, lossless=True)
            elif compression_type == 'webp':
                compressed_image = imagecodecs.webp_encode(data, lossless=True)
            total_compressed += len(compressed_image)
        print('-'*20,)
        print(N, ' imagenet-1k 32x32 images, 2D-32*32, {} compression ratio (raw size/compressed size): '.format(compression_type),
              (N * num_pixel) / total_compressed)
        ##################################################################################################################


    image = training_data_context_all[:,:-1].astype(np.uint8)
    image_3D = np.stack([image[:N,:].reshape([N,w,h]),image[N:N*2,:].reshape([N,w,h]),image[N*2:,:].reshape([N,w,h])],axis=-1) # N*1024
    total_compressed = 0
    for idx in tqdm(range(image_3D.shape[0]),desc='compress ...'):
        data = image_3D[idx].astype(np.uint8)
        if compression_type == 'jpegxl':
            compressed_image = imagecodecs.jpegxl_encode(data, lossless=True)
        elif compression_type == 'webp':
            compressed_image = imagecodecs.webp_encode(data, lossless=True)
        total_compressed += len(compressed_image)
    print('-' * 20, )
    print(N,
          ' imagenet-1k 32x32 images, 3D-32*32, {} compression ratio (raw size/compressed size): '.format(compression_type),
          (N * 1024 * 3) / total_compressed)

    #print('LLM compression ratio (raw size/compressed size): ', (N * 1024) / 6542)
    exit()
def LLM_initiation():
    device = 'cuda:0'
    # Model
    base_model_url = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cache_dir = './cache/huggingface/hub'
    new_model_url = "./llama-3.1-pixel-prediction/checkpoint-10000"
    tokenizer = AutoTokenizer.from_pretrained('./cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5/',)

    base_model_reload = AutoModelForCausalLM.from_pretrained(
        './cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5/',
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    model = PeftModel.from_pretrained(base_model_reload, new_model_url)
    # only use in base model
    # tokenizer.pad_token = tokenizer.eos_token
    model = model.merge_and_unload()
    # model = base_model_reload
    print('model merge success!')
    return  tokenizer,model
def encoder_process_per(inputs,symbols,model,pixel_token_ids,num_pixel):
    #############################################################################################################
    ######## Encoder
    ############################################################################################################
    device = 'cuda:0'
    probs = list()
    num_pixel_seg = inputs.shape[1]
    data = {}
    for subsequence_length in tqdm(range(num_pixel_seg - 1),desc='Start to compress each subpixel'):
        # pixel_sequence: [B, subsequence length]
        pixel_sequence = torch.tensor(inputs[:, :(subsequence_length + 1)]).to(device)
        # right shift, plus 1 based on  original length
        max_length = pixel_sequence.shape[1] + 1
        # mask == 1
        attention_mask = torch.ones_like(pixel_sequence)
        # top_k = 100 follows the https://arxiv.org/abs/2309.10668
        output = model.generate(input_ids=pixel_sequence, attention_mask=attention_mask, max_length=max_length,
                                do_sample=False,
                                return_dict_in_generate=True, output_logits=True, return_legacy_cache=True)
        ## logists to probability, logits is a tuple
        #  Note that the length of this logits tuple is equal to that of shift right (i.e., the prompt or condition token is not included)
        # data[tokenizer.decode(torch.argmax(output.logits[0],dim=-1))] = subsequence_length
        probability_softmax = F.softmax(output.logits[0][:, pixel_token_ids], dim=-1)
        probs.append(probability_softmax)
    probs = torch.cat(probs, dim=0)[(-1 - num_pixel):-1]
    # [n*3,]
    raw_data = symbols[0, :].reshape([-1, ])
    output = list()
    ARITHMETIC_CODER_BASE = 2
    ARITHMETIC_CODER_PRECISION = 32
    encoder = Encoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )
    for pdf, symbol in tqdm(zip(probs, raw_data),desc='Start to Arithmetic Coding '):
        pdf = pdf.detach().cpu().numpy()
        encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()
    compressed_bits = ''.join(map(str, output))
    compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)
    return compressed_bytes,num_padded_bits

# prompt according to the channel
def prompt_token(prompt_string,tokenizer,channel='R'):
    template = 'channel of a flatten RGB image '
    # as a condition
    if channel == 'R':
        input_string = prompt_string
    elif channel == 'G':
        input_string = 'G {}'.format(template) + "Completion:"
    elif channel == 'B':
        input_string = 'B {}'.format(template) + "Completion:"
    else:
        print('No label')
        exit()
    token_result = list(filter(lambda x: x != 'Ġ', tokenizer.tokenize(input_string, add_special_tokens=True)))
    encoding = tokenizer.convert_tokens_to_ids(token_result)
    row = encoding
    return row



def decoder_process_per(compressed_bytes,num_padded_bits,model,tokenizer,prompt_tokens,pixel_token_ids,num_pixel):
    device = 'cuda:0'
    data_iter = iter(bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits))

    # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
    # from the compressed input and returns `None` when the input is exhausted.
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Union[int, None]:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    ARITHMETIC_CODER_BASE = 2
    ARITHMETIC_CODER_PRECISION = 32
    decoder = Decoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )

    length_prompt_token = len(prompt_tokens)
    string = []
    for subsequence_length in tqdm(range(length_prompt_token)):
        # pixel_sequence: [B, subsequence length]
        pixel_sequence = torch.tensor(prompt_tokens[:(subsequence_length + 1)]).to(device).unsqueeze(0)
        # right shift, plus 1 based on  original length
        max_length = pixel_sequence.shape[1] + 1
        # mask
        attention_mask = torch.ones_like(pixel_sequence)
        # top_k = 100 follows the https://arxiv.org/abs/2309.10668
        output = model.generate(input_ids=pixel_sequence, attention_mask=attention_mask,max_length=max_length, do_sample=False,
                                return_dict_in_generate=True, output_logits=True)
        ## logists to probability, logits is a tuple
        #  Note that the length of this logits tuple is equal to that of shift right (i.e., the prompt or condition token is not included)
        # data[tokenizer.decode(torch.argmax(output.logits[0],dim=-1))] = subsequence_length
        probability_softmax = F.softmax(output.logits[0][:, pixel_token_ids], dim=-1)
        string.append(tokenizer.decode(torch.argmax(output.logits[0], dim=-1)))
        probs = probability_softmax
    sequence_array = np.array(prompt_tokens).reshape([-1, ])

    decode_values = []
    for idx in tqdm(range(num_pixel), desc='decode from compressed data'):
        probs = probs.detach().cpu().numpy()
        # print(probs.shape)
        subpixel = decoder.decode(
            normalize_pdf_for_arithmetic_coding(probs[0])
        )
        decode_values.append(subpixel)
        token_id = tokenizer.encode(str(subpixel), add_special_tokens=False)
        sequence_array = np.insert(sequence_array, len(sequence_array), token_id)
        max_length = len(sequence_array) + 1
        # exclude the last item (dummy token)
        pixel_sequence = sequence_array
        # top_k = 100 follows the https://arxiv.org/abs/2309.10668
        pixel_sequence = torch.tensor(pixel_sequence[None]).to(device)
        output = model.generate(input_ids=pixel_sequence, max_length=max_length, do_sample=False,
                                return_dict_in_generate=True, output_logits=True)
        probability_softmax = F.softmax(output.logits[0][:, pixel_token_ids], dim=-1)
        probs = probability_softmax
        # return sub
    decode_data = np.array(decode_values).reshape([-1, ])
    return decode_data
def compress_and_decompress(tokenizer, LLM,training_data_context,N,bpsp_or_compression_ratio=1):
    device = 'cuda:0'
    num_pixel = 1024 # 32*32 patch
    def format_chat_template(row):
        template = 'channel of a flatten RGB image '
        # as a condition
        if row[-1] == -1:
            input_string = 'Prompt: R {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -2:
            input_string = 'G {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -3:
            input_string = 'B {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        else:
            print('No label')
            exit()
        token_result = list(filter(lambda x: x != 'Ġ', tokenizer.tokenize(input_string, add_special_tokens=True)))
        encoding = tokenizer.convert_tokens_to_ids(token_result)
        row = encoding
        return row
    # only select one patch
    uncompressed_data = copy.deepcopy(training_data_context)[0:1, :-1]
    print('-'*20,'uncompressed data shape:',uncompressed_data.shape)
    print('-'*20)
    ##### add the prompt to compress
    training_data_context = np.apply_along_axis(format_chat_template, 1, training_data_context)
    ds_validate = Dataset.from_dict({'input_ids': training_data_context})
    ds_validate = ds_validate.batch(batch_size=1)
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    # list batch N,
    inputs = np.array(ds_validate[0]['input_ids'])
    ###########################################   Encoder    ###########################################################
    ###############################################################################################
    ##### encode
    compressed_bytes, num_padded_bits = encoder_process_per(inputs,uncompressed_data,LLM,pixel_token_ids,num_pixel)
    print('-' * 20)
    print('check some compressed bytes=> {}'.format(compressed_bytes[:20]),'....')
    print('-' * 20)
    print('compression ratio (raw size / compressed size):{}'.format(num_pixel/len(compressed_bytes)))
    print('-' * 20)
    ###########################################   Decoder    ###########################################################
    #################### Prompt Generation #################
    template = 'channel of a flatten RGB image '
    channel = 'R'
    prompt_string = 'Prompt: {} {}'.format(channel,template) + "Completion:"
    prompt_tokens = prompt_token(prompt_string,tokenizer)  #
    ############# decode
    decompressed_data = decoder_process_per(compressed_bytes,num_padded_bits,LLM,tokenizer,prompt_tokens,pixel_token_ids,num_pixel)
    print('-' * 20)
    print('decompressed data:',decompressed_data)
    uncompressed_data = np.squeeze(uncompressed_data) # 1*N => N,
    print('compressed data:',)
    print(decompressed_data.shape)
    print(uncompressed_data.shape)
    print('decompressed_data == uncompressed_data ?? => ', np.array_equal(decompressed_data,uncompressed_data))
    print('-' * 20)
    exit()




def test_compress_per_image(tokenizer, model,training_data_context,N,bpsp_or_compression_ratio=1):
    '''
    :param training_data_context: N* 1024
    :return:
    '''
    device = 'cuda:0'
    def format_chat_template(row):
        template = 'channel of a flatten RGB image '
        # as a condition
        if row[-1] == -1:
            input_string = 'Prompt: R {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -2:
            input_string = 'G {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -3:
            input_string = 'B {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        else:
            print('No label')
            exit()
        token_result = list(filter(lambda x: x != 'Ġ', tokenizer.tokenize(input_string, add_special_tokens=True)))
        encoding = tokenizer.convert_tokens_to_ids(token_result)
        row = encoding
        return row

    symbols = copy.deepcopy(training_data_context)[:,:-1]
    training_data_context = np.apply_along_axis(format_chat_template, 1, training_data_context)
    ds_validate = Dataset.from_dict({'input_ids':training_data_context})
    ds_validate = ds_validate.batch(batch_size=32)

    probs = list()
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    for batch_data in tqdm(ds_validate):
        inputs = batch_data['input_ids']
        samples = torch.tensor(inputs).to(device)
        logits = model(samples).logits
        #max_idx = torch.argmax(logits, dim=-1)
        #decode_result = [tokenizer.decode(id) for id in max_idx][(-2-1024):-2]
        logits_number = logits[:, (-2-1024):-2,:]
        logits_number = logits_number[:,:,pixel_token_ids]
        probability_softmax = F.softmax(logits_number, dim=-1)
        probs.append(probability_softmax)


    probs = torch.squeeze(torch.cat(probs,dim=0)).reshape([-1, 256])
    # [n*3,]
    sequence_array = symbols.reshape([-1, ])
    output = list()
    ARITHMETIC_CODER_BASE = 2
    ARITHMETIC_CODER_PRECISION = 32
    encoder = Encoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )
    print('star to AE')
    for pdf, symbol in tqdm(zip(probs, sequence_array)):
        pdf = pdf.detach().cpu().numpy()
        encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()
    print('done')
    compressed_bits = ''.join(map(str, output))
    compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)
    if bpsp_or_compression_ratio == 1:
        print('bpsp: ',(symbols.shape[0]*symbols.shape[1]) / len(compressed_bytes))
        return None
    else:
        print('compression ratio: ', (symbols.shape[0] * symbols.shape[1]) / len(compressed_bytes))
        return (symbols.shape[0] * symbols.shape[1]) / len(compressed_bytes), len(compressed_bytes)
def test_compress(args):
    device = 'cuda:0'
    # Model
    base_model_url = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cache_dir = './cache/huggingface/hub'
    new_model_url = "./llama-3.1-pixel-prediction/checkpoint-12500"
    tokenizer = AutoTokenizer.from_pretrained(base_model_url,cache_dir=cache_dir)

    base_model_reload= AutoModelForCausalLM.from_pretrained(
        './cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5/',
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir
    )
    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    model = PeftModel.from_pretrained(base_model_reload, new_model_url)
    # only use in base model
    #tokenizer.pad_token = tokenizer.eos_token
    model = model.merge_and_unload()
    #model = base_model_reload
    print('model merge success!')

    N= args.test_size

    config = {'validate_size':N,'training_size':N}
    training_data_context_all, training_data_context_all = load_dataset_numpy(None, config)
    # training data context all N*3, 1025
    if args.test_ch=='R':
        training_data_context = training_data_context_all[:N,:]
    elif args.test_ch=='G':
        training_data_context = training_data_context_all[N:N*2, :]
    elif args.test_ch == 'B':
        training_data_context = training_data_context_all[N*2:, :]
    else:
        exit()
    symbols = copy.deepcopy(training_data_context[:,:-1])

    def format_chat_template(row):
        template = 'channel of a flatten RGB image '
        # as a condition
        if row[-1] == -1:
            input_string = 'Prompt: R {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -2:
            input_string = 'G {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        elif row[-1] == -3:
            input_string = 'B {}'.format(template) + "Completion:" + ' '.join(
                map(str, row[:-1])) + "<|end_of_text|>"
        else:
            print('No label')
            exit()
        token_result = list(filter(lambda x: x != 'Ġ', tokenizer.tokenize(input_string, add_special_tokens=True)))
        encoding = tokenizer.convert_tokens_to_ids(token_result)
        row = encoding
        return row
    training_data_context = np.apply_along_axis(format_chat_template, 1, training_data_context)
    ds_validate = Dataset.from_dict({'input_ids':training_data_context})
    ds_validate = ds_validate.batch(batch_size=args.test_bs)

    probs = list()
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    for batch_data in tqdm(ds_validate):
        inputs = batch_data['input_ids']
        samples = torch.tensor(inputs).to(device)
        logits = model(samples).logits
        ''' for debug
        max_idx = torch.squeeze(torch.argmax(logits, dim=-1))
        decode_result = [tokenizer.decode(id) for id in max_idx]#[(-2-1024):-2]
        print(decode_result)
        print(len(decode_result))
        print(training_data_context[0])
        data = [i for i in training_data_context_all[N][:-1]]
        for i in range(1024):
            print(decode_result[i+9],data[i])
        '''
        logits_number = logits[:, (-2-1024):-2,:]
        logits_number = logits_number[:,:,pixel_token_ids]
        probability_softmax = F.softmax(logits_number, dim=-1)
        probs.append(probability_softmax)


    probs = torch.squeeze(torch.cat(probs,dim=0)).reshape([-1, 256])
    # [n*3,]
    sequence_array = symbols.reshape([-1, ])
    print(probs.shape)
    print(sequence_array.shape)

    output = list()
    ARITHMETIC_CODER_BASE = 2
    ARITHMETIC_CODER_PRECISION = 32
    encoder = Encoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )

    for pdf, symbol in tqdm(zip(probs, sequence_array),desc='start to decode'):
        pdf = pdf.detach().cpu().numpy()
        encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()

    compressed_bits = ''.join(map(str, output))
    compressed_bytes, num_padded_bits = bits_to_bytes(compressed_bits)
    print('compression ratio (raw size/compressed size): ',(symbols.shape[0]*symbols.shape[1]) / len(compressed_bytes))


def initiation_config(args,config):

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
    )
    #bnb_4bit_quant_type = "nf4",
    #bnb_4bit_compute_dtype = torch_dtype,
    #bnb_4bit_quant_storage = torch_dtype,
    #bnb_4bit_use_double_quant = True,

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["o_proj","q_proj", "k_proj", "v_proj"],#"all-linear"#["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    training_arguments = SFTConfig(
        output_dir=args.new_model,
        per_device_train_batch_size= config['batch_size'],
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        optim="paged_adamw_32bit",
        num_train_epochs=2,
        eval_strategy="steps",
        max_steps=125000,
        eval_steps=2000,
        save_steps=2000,
        logging_steps=2000,
        load_best_model_at_end=True,
        save_strategy='steps',
        save_total_limit=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=0,
        logging_strategy="steps",
        learning_rate=1.82241935483871e-05,
        fp16=True,
        bf16=False,
        group_by_length=False,
        #eval_on_start = True,
        report_to="wandb",
        dataset_kwargs={'skip_prepare_dataset': True}
    )
    return bnb_config,torch_dtype,attn_implementation,peft_config,training_arguments
def add_prompt_label(data,channel):
    if channel == 'R':
        c = -1
    elif channel == 'G':
        c = -2
    elif channel == 'B':
        c = -3
    label = np.full((data.shape[0], 1), c)
    data = np.append(data, label, axis=1)
    return data
def load_dataset_numpy(args,config):
    patch_pixel = 1024
    ################################################################
    training_data = load_imagenet32('./imagenet32/val').astype(np.int16)

    R_array = training_data[:, :patch_pixel]
    G_array = training_data[:, patch_pixel:int(patch_pixel * 2)]
    B_array = training_data[:, patch_pixel * 2:]

    # label R: -1 G: -2 B: -3
    R_label = np.full((R_array.shape[0], 1), -1)
    R_array = np.append(R_array, R_label, axis=1)

    G_label = np.full((G_array.shape[0], 1), -2)
    G_array = np.append(G_array, G_label, axis=1)

    B_label = np.full((B_array.shape[0], 1), -3)
    B_array = np.append(B_array, B_label, axis=1)

    N_validate = config['validate_size']
    training_size = config['training_size']
    print('-------- load {} data for validation'.format(N_validate) )
    print('-------- load {} data for training'.format(training_size))
    training_data_context = np.concatenate(
        (R_array[:training_size, :], G_array[:training_size, :], B_array[:training_size, :]), axis=0)
    validate_data_context = np.concatenate((R_array[training_size:training_size + N_validate, :],
                                            G_array[training_size:training_size + N_validate, :],
                                            B_array[training_size:training_size + N_validate, :]), axis=0)
    return training_data_context,validate_data_context

def load_dataset_preprocessing_ids(args,config):
    patch_pixel = 1024
    ################################################################
    import numpy as np
    from glob import glob

    # Load training data in chunks
    chunk_size = 1000
    training_data_path = glob('./benchmark/HighResolution/DIV2K_train_p32_npy/*.npy')
    training_data_chunks = [np.load(data_path) for data_path in training_data_path]
    length_training = len(training_data_chunks)
    # Process training data in chunks
    training_data_context = []
    for i in tqdm(range(length_training)):
        data = training_data_chunks.pop()
        training_data_context.extend(data) #np.concatenate(training_data_chunks, axis=0)
        del data
        gc.collect()
    # Clear memory after processing each chunk
    del training_data_chunks
    gc.collect()
    # Load validation data in chunks
    valid_data_path = glob('./benchmark/HighResolution/DIV2K_valid_p32_npy_sub/*.npy')
    valid_data_chunks = [np.load(data_path) for data_path in valid_data_path]
    length_valid = len(valid_data_chunks)
    # Process training data in chunks
    validate_data_context = []
    for i in tqdm(range(length_valid)):
        data = valid_data_chunks.pop()
        validate_data_context.extend(data)  # np.concatenate(training_data_chunks, axis=0)
        del data
        gc.collect()
    # Clear memory after processing each chunk
    del valid_data_chunks
    gc.collect()
    print(len(training_data_context))
    print(len(validate_data_context))
    return training_data_context,validate_data_context





def main(args):

    max_seq_length = 1034
    configs_dir = os.path.join('./', '{}.yml'.format(args.running_name))
    with open(configs_dir, "rb") as f:
        config = yaml.safe_load(f)

    ###################################################################
    # model initiation
    ###################################################################
    bnb_config,torch_dtype,attn_implementation,peft_config,training_arguments = initiation_config(args,config)


    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              cache_dir=args.cache_dir,) # device_map="auto",
    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                 cache_dir=args.cache_dir, torch_dtype=torch_dtype,attn_implementation=attn_implementation,)#.to('cuda') # device_map="auto", device_map='balanced_low_0'


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    ##############################
    ### add LORA
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)

    #######################################################################
    ######################################################################
    ### dataset model
    #######################################################################
    print('-'*20,'load dataset')
    #training_data_context, validate_data_context = load_dataset_preprocessing_ids(args,config)# load_dataset_numpy(args,config)


    #############################################################
    #### dataset map and preprocessing
    #############################################################
    data_files = {"train": './benchmark/DIV2K-test/large_data_train_p16_half_full.json',
                  'validation': './benchmark/DIV2K-test/large_data_valid_p16_half_full.json'}
    ds = load_dataset(
        "json", data_files=data_files, streaming=True
    )
    shuffled_dataset = ds.shuffle(buffer_size=10_000, seed=42)
    ##################################################################
    #################################################################
    ### To train
    ################################################################
    print('-'*20,'training mode FP16:{}'.format(training_arguments.fp16))
    trainer = SFTTrainer(
        model=model,
        train_dataset=shuffled_dataset['train'],
        eval_dataset=shuffled_dataset['validation'],
        peft_config=peft_config,
        max_seq_length=798,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train() #
    trainer.model.save_pretrained(args.new_model)

if __name__ == "__main__":
    import warnings


    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--new_model", default="./llama-3.1-pixel-prediction-HR-Three-50-kqvo-r-64-continue-validation-div")
    parser.add_argument("--model_size", default="medium")
    parser.add_argument("--running_name", default="finetuning_llama3.1_8B")
    parser.add_argument("--image_dir_train", type=str, default='./imagenet32/train')
    parser.add_argument("--image_dir_validate", type=str, default='./imagenet32/val')
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--config", type=str, default='./configs/')
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--seed", default=2)
    parser.add_argument("--save_dir", default='/LLM-coding/model_save')
    parser.add_argument("--model_id", default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument("--cache_dir", default='./cache/huggingface/hub/')
    parser.add_argument("--gpu_id", default='0')
    parser.add_argument("--prompt_template",default="You are image RGB pixel predictor. You predict pixel is from the flatten 1D format of an 2D image. I will give you previous RGB values,you predict next RGB values by considering all previous pixels with potential image semantic or color change.")
    parser.add_argument("--generate_length",default=9)
    parser.add_argument("--test_ch", default='R',choices=['R', 'G', 'B'], help='Choose a color')
    parser.add_argument("--test_bs", default=16)
    parser.add_argument("--test_size", default=100,type=int)

    args = parser.parse_args()
    main(args)

