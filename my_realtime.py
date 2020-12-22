import time
import os
import random
import time
import argparse

import torch
import torch.nn.functional as F
import pyaudio
import librosa
import numpy as np
import webrtcvad

from scipy import spatial
from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

class SpeakerIdentifier:
    def __init__(self, model_path, enroll_dir):
        self.embedder = SpeechEmbedder()
        self.embedder.load_state_dict(torch.load(model_path))
        self.embedder.eval()

        self.speakers = dict()
        files = os.listdir(enroll_dir)
        for spkr_file in files:
            speaker_id = os.path.splitext(spkr_file)[0]
            path = os.path.join(enroll_dir, spkr_file)
            self.speakers[speaker_id] = np.load(path)
    
    def identify(self, samples):
        S = librosa.core.stft(y=samples, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * hp.data.sr), hop_length=int(hp.data.hop * hp.data.sr))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)

        S = S.T
        S = np.reshape(S, (1, -1, hp.data.nmels))

        batch = torch.Tensor(S)

        results = self.embedder(batch)
        results = results.reshape((1, hp.model.proj))
        
        scores = dict()
        for speaker_id, speaker_emb in self.speakers.items():
            speaker_emb_tensor = torch.Tensor(speaker_emb).reshape((1, -1))
            output = F.cosine_similarity(results, speaker_emb_tensor)
            output = output.cpu().detach().numpy()[0]

            scores[speaker_id] = output

        return scores

class AudioHandler(object):
    def __init__(self, identifier):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = hp.data.sr
        self.CHUNK = 8000
        self.p = None
        self.stream = None
        self.identifier = identifier

    def start(self):
        self.p = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        np_arr = np.frombuffer(in_data, dtype=np.float32)
        vad_arr = (np_arr * 32768).astype(np.int16).tobytes()
        vad_arr = vad_arr[:int(2*hp.data.sr*30/1000)]
        active = self.vad.is_speech(vad_arr, hp.data.sr)
        if not active:
            print('silence')
            return None, pyaudio.paContinue

        results = self.identifier.identify(np_arr)

        max_sim = 0
        max_id = ''
        for speaker_id, sim in results.items():
            if sim > max_sim:
                max_id = speaker_id
                max_sim = sim

        print(max_id, results)

        return None, pyaudio.paContinue

    def mainloop(self):
        try:
            while (self.stream.is_active()):
                time.sleep(2.0)
        except KeyboardInterrupt:
            print('Ctrl+C received, stopping..')
            return

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str,
                    help='model file path')
parser.add_argument('enroll', type=str,
                    help='enroll file dir')

args = parser.parse_args()

identifier = SpeakerIdentifier(args.model, args.enroll)

audio = AudioHandler(identifier)
audio.start()
audio.mainloop()
audio.stop()