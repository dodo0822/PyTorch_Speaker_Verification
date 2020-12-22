import glob
import os
import librosa
import numpy as np

from hparam import hparam as hp
from pydub import AudioSegment

# downloaded dataset path
base_path = './dataset/chihweif/train_wav/'
audio_path = os.listdir(base_path)

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs('./dataset/chihweif/train_npy/', exist_ok=True)   # make folder to save train file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    print("total speaker number : %d" % total_speaker_num)
    
    for i, folder in enumerate(audio_path):
        print("%dth speaker (%s) processing..." % (i, folder))
        utterances_spec = []
        for utter_name in os.listdir(os.path.join(base_path, folder)):
            if utter_name[-4:] == '.wav':
                utter_path = os.path.join(base_path, folder, utter_name)         # path of each utterance

                sound = AudioSegment.from_file(utter_path, 'wav')
                normalized_sound = match_target_amplitude(sound, -20.0)
                normalized_sound.export(utter_path, format='wav')

                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        np.save(os.path.join('dataset/chihweif/train_npy/', "%s.npy" % folder), utterances_spec)
        
if __name__ == '__main__':
    save_spectrogram_tisv()