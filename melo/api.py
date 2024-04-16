import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        import threading
        import queue
        import sounddevice as sd
        import numpy as np

        # Assuming `generate_audio` is a function that generates audio data for a given text
        # def generate_audio(text):
        #     # Placeholder for actual audio generation logic
        #     return np.random.rand(44100) # Example: Generate 1 second of random audio

        # Function to play audio from the queue
        def play_audio(audio_queue):
            while True:
                audio = audio_queue.get()
                if audio is None: # Signal to stop
                    break
                sd.play(audio, samplerate=44100)
                sd.wait()

        # Initialize the queue and the audio player thread
        audio_queue = queue.Queue()
        audio_player = threading.Thread(target=play_audio, args=(audio_queue,))
        audio_player.start()
        
        
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for i,t in enumerate(tx):
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                print(audio.shape)
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            # audio = self.audio_numpy_concat(audio, sr=self.hps.data.sampling_rate, speed=speed)
            # soundfile.write(f"./{t}.wav", audio, self.hps.data.sampling_rate, format=format)
            # Save each piece of audio to a separate file
            # Generate a unique filename for each piece of audio
            audio_queue.put(audio)
            filename = f"./audio_{i}.wav"
            soundfile.write(filename, audio, self.hps.data.sampling_rate)
            audio_list.append(audio)
        torch.cuda.empty_cache()
        print(audio.shape)
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        print(audio.shape)
        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
        audio_queue.put(None)
        audio_player.join()

    def tts_to_sound(self, text, speaker_id, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, position=None, quiet=False,):
            import threading
            import queue
            import sounddevice as sd
            import numpy as np

            # Assuming `generate_audio` is a function that generates audio data for a given text
            # def generate_audio(text):
            #     # Placeholder for actual audio generation logic
            #     return np.random.rand(44100) # Example: Generate 1 second of random audio

            # Function to play audio from the queue
            def play_audio(audio_queue):
                while True:
                    audio = audio_queue.get()
                    if audio is None: # Signal to stop
                        break
                    sd.play(audio, samplerate=44100)
                    sd.wait()

            # Initialize the queue and the audio player thread
            audio_queue = queue.Queue()
            audio_player = threading.Thread(target=play_audio, args=(audio_queue,))
            audio_player.start()
            
            
            language = self.language
            texts = self.split_sentences_into_pieces(text, language, quiet)
            if pbar:
                tx = pbar(texts)
            else:
                if position:
                    tx = tqdm(texts, position=position)
                elif quiet:
                    tx = texts
                else:
                    tx = tqdm(texts)
            for i,t in enumerate(tx):
                if language in ['EN', 'ZH_MIX_EN']:
                    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
                device = self.device
                bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
                with torch.no_grad():
                    x_tst = phones.to(device).unsqueeze(0)
                    tones = tones.to(device).unsqueeze(0)
                    lang_ids = lang_ids.to(device).unsqueeze(0)
                    bert = bert.to(device).unsqueeze(0)
                    ja_bert = ja_bert.to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                    del phones
                    speakers = torch.LongTensor([speaker_id]).to(device)
                    audio = self.model.infer(
                            x_tst,
                            x_tst_lengths,
                            speakers,
                            tones,
                            lang_ids,
                            bert,
                            ja_bert,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=1. / speed,
                        )[0][0, 0].data.cpu().float().numpy()
                    del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                audio_queue.put(audio)
                # f = open(f"./audio_{i}.txt", "w")
                # f.write(audio)
                # f.close()
                # np.savetxt(f"./audio_{i}.txt",audio)
            torch.cuda.empty_cache()
            audio_queue.put(None)
            audio_player.join()