import random
from torch.utils.data import Dataset
import os
import mutagen
import fnmatch
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import torch
from threading import Lock
import torchaudio

def find_audio_files(directory_path: str, extensions: list = ['mp3', 'flac', 'wav', 'webm']):
    """Recursively find all MP3 files within the given directory.

    Args:
    directory_path (str): The path to the directory to search in.

    Returns:
    list: A list of paths to the MP3 files found.
    """
    mp3_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            #print(file)
            if any(fnmatch.fnmatch(file, f'*.{extension}') for extension in extensions):
                mp3_files.append(os.path.join(root, file))
    return mp3_files


class AudioDataset(Dataset):
    def __init__(self, directory, transform=None, audio_length=16_000*2, target_samplerate=16_000):
        self.audio_length = audio_length
        self.directory = directory
        self.filenames = find_audio_files(directory)
        random.shuffle(self.filenames)
        
        self.transform = transform
        self.sample_rate = target_samplerate

        def load(file_path):
            try:
                audio = mutagen.File(file_path)
                
                l = audio.info.length
                sr = audio.info.sample_rate
                del audio
                return file_path, l, sr
            except Exception as e:
                print(e)
                pass
                                        
        with ThreadPool(8) as pool:
            files = [x for x in list(tqdm(pool.imap_unordered(load, self.filenames), total=len(self.filenames))) if x is not None]


            self.files = [t for t, _, _ in files]
            self.sizes = ((torch.tensor([t*x for _, t, x in files]) // self.audio_length) - 1).cumsum(dim=0)# - 1
            self.sample_rates = torch.tensor([t for _, _, t in files])
            self.locks = [Lock() for _ in range(len(files))]    
        #self.files = [load(filepath) for filepath in tqdm(self.filenames)]

        self.n = 12

        #self.sizes = (torch.tensor([torchaudio.info(t).num_frames - self.audio_length for t in tqdm(self.filenames)]) // self.audio_length).cumsum(dim=0) - 1

        #self.sizes = (torch.tensor(list(map(lambda t: torchaudio.info(t).num_frames - self.audio_length, self.filenames))) // self.audio_length).cumsum(dim=0) - 1

        #print(self.sizes.cumsum().max())
        

    def __len__(self):
        return int(self.sizes.max())

    def find_position_and_t(self, idx):
        n = len(self.filenames)
        # Bringing idx into the range of 0 to n
        position = idx % n
        # Finding at which t idx is
        t_position = idx // n
        return position

    def __getitem__(self, in_idx):
        idx = (self.sizes >= in_idx).nonzero(as_tuple=True)[0][0].item()
        with self.locks[idx]:
            offset = in_idx
            if idx > 0:
                offset = (in_idx - self.sizes[idx - 1])
            file_path = self.files[idx]
            sample_rate = self.sample_rates[idx]

            print(idx, in_idx, offset, sample_rate, file_path)
            
            waveform, sample_rate = torchaudio.load(file_path, frame_offset=offset*sample_rate, num_frames=self.audio_length*sample_rate)
            
            resampled = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sample_rate).mean(dim=1)
            
            resampled = torch.nn.functional.pad(resampled, (0, self.audio_length - resampled.size(0)))
            #print(resampled.shape)
            return resampled