import os
import random
import time
import torch
import numpy as np

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str,
                    help='model file path')
parser.add_argument('data', type=str,
                    help='utterance data path')

os.makedirs('./dataset/chihweif/enroll/', exist_ok=True)

args = parser.parse_args()
print('Using model path: %s' % args.model)

npy_files = os.listdir(args.data)

for npy_file in npy_files:
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(args.model))
    embedder_net.eval()

    path = os.path.join(args.data, npy_file)
    speaker_id = os.path.splitext(npy_file)[0]

    print('Enrolling %s' % speaker_id)

    utters = np.load(path)
    utters = np.transpose(utters, axes=(0,2,1))
    utters = torch.Tensor(utters)
    print(utters.size())

    utters = utters
    results = embedder_net(utters)

    results = results.cpu().detach().numpy()
    emb = np.mean(results, axis=0)
    
    np.save('./dataset/chihweif/enroll/%s.npy' % speaker_id, emb)