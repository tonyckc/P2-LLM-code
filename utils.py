
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions."""

import chex
import numpy as np
import random
import os
import torch
from typing import Tuple
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from torch.distributed import init_process_group, destroy_process_group

import pickle
import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path :
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.module.state_dict(), path)	#
        self.val_loss_min = val_loss




def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12315"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def bits_to_bytes(bits: str) -> Tuple[bytes, int]:
  """Returns the bytes representation of bitstream and number of padded bits."""
  # Pad the string with zeros if the length is not a multiple of 8.
  padded_bits = bits.zfill((len(bits) + 7) // 8 * 8)
  num_padded_bits = len(padded_bits) - len(bits)

  # Split the string into 8-bit chunks.
  chunks = [padded_bits[i : i + 8] for i in range(0, len(padded_bits), 8)]

  # Convert each chunk to an integer and then to a byte.
  bytes_data = bytes([int(chunk, base=2) for chunk in chunks])

  return bytes_data, num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> str:
  """Returns the bitstream of bytes data accounting for padded bits."""
  return ''.join([bin(byte)[2:].zfill(8) for byte in data])[num_padded_bits:]


def right_shift_bytes_by_one(data: bytes) -> Tuple[bytes, int]:
  """Returns right-shifted bytes, i.e., divided by 2, and the number of bytes.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  However, for certain data types (e.g., images), masking the MSB and leaving
  the rest of the byte unchanged will destroy the structure of the data. Thus,
  we instead divide the number by two (i.e., we shift the bits to the right by
  one).

  Args:
    data: The bytes to be shifted.
  """
  return bytes([byte >> 1 for byte in data]), len(data)


def zero_most_significant_bit_if_not_ascii_decodable(
    data: bytes,
) -> Tuple[bytes, int]:
  """Returns ascii-decodable data & the number of zeroed most significant bits.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  Args:
    data: The bytes to be shifted.
  """
  masked_bits = 0
  masked_data = list()

  for byte in data:
    if chr(byte).isascii():
      masked_data.append(byte)
    else:
      masked_bits += 1
      masked_data.append(byte & 0x7F)

  return bytes(masked_data), masked_bits

def set_seed(seed):
    '''
    seed  setting
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize_pdf_for_arithmetic_coding(pdf: chex.Array) -> chex.Array:
  """Normalizes the probabilities for arithmetic coding.

  Arithmetic coding converts the floating-point pdf to integers to avoid
  numerical issues. To that end, all pdf values need to be larger than the
  machine epsilon (to yield different integer values) and the sum of the pdf
  cannot exceed 1 (minus some precision tolerance).

  Args:
    pdf: The probabilities to be normalized.

  Returns:
    The normalized probabilities.
  """

  machine_epsilon = np.finfo(np.float32).eps
  # Normalize the probabilities to avoid floating-point errors.
  pdf = pdf / np.cumsum(pdf)[-1]
  # Ensure all probabilities are sufficiently large to yield distinct cdfs.
  pdf = (1 - 2 * pdf.shape[0] * machine_epsilon) * pdf + machine_epsilon
  return pdf
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def np_unranked_unique(nparray):
    n_unique = len(np.unique(nparray))
    ranked_unique = np.zeros([n_unique])
    i = 0
    for x in nparray:
        if x not in ranked_unique:
            ranked_unique[i] = x
            i += 1
    return ranked_unique


def rgb_to_lab_image(rgb_image):
    # Normalize RGB values to [0, 1]
    rgb_image  = (rgb_image+ 1) * 127.5
    rgb_normalized = rgb_image / 255.0
    # Convert RGB to LAB
    lab_image = color.rgb2lab(rgb_normalized)
    return lab_image


def construct_mapping(clusters,feature_extractor):


    '''
    space = 40
    segment = 256 // space

    self_construct_mapping = []
    sample = np.arange(256)
    for seg in range(1,segment+1):
        self_construct_mapping_sub = []
        for i in range((seg-1)*space,seg*space):
             self_construct_mapping_sub = search_color(i, None, clusters, self_construct_mapping_sub)
        self_construct_mapping.extend(self_construct_mapping_sub)
    self_construct_mapping_sub = []
    for i in range(segment * space, 256):
        self_construct_mapping_sub = search_color(i, None, clusters, self_construct_mapping_sub)
    self_construct_mapping.extend(self_construct_mapping_sub)

    print(np.unique(self_construct_mapping).shape)
    #exit()
    '''

    self_construct_mapping = []
    sample = np.arange(256)

    #############################################
    def l2_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def image_similarity(lab_image1, lab_image2):
        lab1 = LabColor(lab_l=lab_image1[0], lab_a=lab_image1[1], lab_b=lab_image1[2])
        lab2 = LabColor(lab_l=lab_image2[0], lab_a=lab_image2[1], lab_b=lab_image2[2])
        return delta_e_cie2000(lab1, lab2)

    patch_size = 32
    current_seg = np.concatenate((np.arange(256),np.arange(256),np.arange(256),np.arange(256)))  # np.random.randint(0, 256, size=(1, 1024)) #image_1D[(seg - 1) * num_pixel_seg:seg * num_pixel_seg, c]#
    current_seg_2d = np.repeat(current_seg.reshape([patch_size, patch_size, 1]), repeats=3,
                               axis=2)  # current_seg.reshape([patch_size,patch_size,3]) #
    encoding = feature_extractor([current_seg_2d], return_tensors="pt")
    samples = encoding.input_ids[0][:256]
    samples = np.array(samples)
    tokens = np_unranked_unique(samples)
    #used_tokens = tokens


    from collections import Counter
    counts = Counter(samples)
    # Extract the counts into a list
    frequency_repeat = list(counts.values())
    # maxiumn iterative loop
    max_iterative_loop = np.max(np.array(frequency_repeat))


    repeat_end = 0
    top_k_in_each_segment = []
    top_k_in_each_cluster = []
    ### we extract the top-k in each repeated segment and get the top-k index.
    clusters = rgb_to_lab_image(clusters)

    for idx in range(len(frequency_repeat)):
        current_token = int(tokens[idx])
        current_token_cluster = clusters[current_token]
        pixel_array = [[value,value,value] for value in range(repeat_end,repeat_end+frequency_repeat[idx])]
        pixel_array = np.array(pixel_array) / 127.5 - 1 #rgb_to_lab_image((np.array(pixel_array) / 255.)) #
        distances = np.array([l2_distance(current_token_cluster, element) for element in pixel_array])
        top_k_indices = np.argsort(distances)
        #####################################
        # for cluster
        distances = np.array([l2_distance(current_token_cluster, element) for element in clusters])
        k = 256
        ## ignore the self point
        top_k_indices_cluster = np.argsort(distances)[:k]
        top_k_in_each_cluster.append(list(top_k_indices_cluster))
        ############################################
        top_k_in_each_segment.append(list(top_k_indices))
        repeat_end += frequency_repeat[idx]
    new_mapping_tokens = np.zeros(256)*999
    used_tokens = []

    for loop_idx in range(max_iterative_loop):
        repeat_end = 0
        for idx in range(len(frequency_repeat)):
            # empty top-k sub-list
            if len(top_k_in_each_segment[idx]) != 0:
                # assign token
                current_position = loop_idx
                for token in top_k_in_each_cluster[idx]:
                    if token not in used_tokens:
                        new_mapping_tokens[repeat_end+current_position] = token
                        used_tokens.append(token)
                        break

                #pixel_array = [[value, value, value] for value in range(repeat_end, repeat_end + frequency_repeat[idx])]
                #pixel_array = np.array(pixel_array) / 127.5 - 1
                top_k_in_each_segment[idx].pop(0)

            repeat_end += frequency_repeat[idx]


    new_mapping_tokens = np.array(new_mapping_tokens).astype(np.int)


    ###########################################

    #clusters_lab = rgb_to_lab_image(clusters)

    #for i, value in enumerate(sample):
    #       self_construct_mapping = search_color(i, value, clusters, self_construct_mapping)
    self_construct_mapping = new_mapping_tokens
    #for i in self_construct_mapping:

    #samples_img = np.reshape(np.rint(127.5 * (clusters[self_construct_mapping] + 1.0)), [16,16, 3]).astype(np.uint8)
    #plt.imshow(samples_img)
    #plt.show()
    #exit()

    all_token_ids = np.ones(512)
    #print(len(self_construct_mapping))
    all_token_ids[self_construct_mapping] = 1
    #print(all_token_ids)
    #exit()

    color_database = np.zeros([512,3])
    for indx,value in enumerate(self_construct_mapping):
      color_database[value] = [indx,indx,indx]

    return np.array(self_construct_mapping),all_token_ids





def search_color(i, value, clusters, self_construct_mapping):

  def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

  def image_similarity(lab_image1,lab_image2):
      lab1 = LabColor(lab_l=lab_image1[0], lab_a=lab_image1[1], lab_b=lab_image1[2])
      lab2 = LabColor(lab_l=lab_image2[0], lab_a=lab_image2[1], lab_b=lab_image2[2])
      return delta_e_cie2000(lab1, lab2)


  A = (np.array([i, i, i]) / 127.5 - 1)
  #A = rgb_to_lab_image((np.array([i, i, i]) / 255.))


  distances = np.array([l2_distance(A, element) for element in clusters])

  #
  k = 256
  top_k_indices = np.argsort(distances)[:k]

  for id in range(k):
    if top_k_indices[id] in self_construct_mapping:

      if id == k - 1:
        print('no match')
        exit()
      continue
    else:
      self_construct_mapping.append(top_k_indices[id])
      break
  return self_construct_mapping

from glob import glob
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict