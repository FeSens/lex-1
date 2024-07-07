import numpy as np
import torch

import torchaudio
import torch
from encodec.model import EncodecModel
from tqdm import tqdm
import os

def encode_audio_with_overlap_and_batching(
    filepath,
    output_path,
    segment_duration=1.0,
    overlap=0.1,
    batch_size=64
):
    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform.to("mps")
    # print the time it took to load the audio file and make it with tabs justify to the right
    # Initialize model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.to("mps")
    # Resample if necessary
    if sample_rate != model.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.sample_rate).to("mps")
        waveform = resampler(waveform)
    # Prepare to process in overlapping chunks
    segment_samples = 25600#int(model.sample_rate * segment_duration)
    overlap_samples = 2560#int(segment_samples * overlap)
    step = segment_samples - overlap_samples//2
    total_samples = waveform.shape[1]
    all_codes = []

    # Temp buffer for batching
    batch = []
    # Process each chunk with overlap
    for start  in tqdm(range(0, total_samples - segment_samples + 1, step)):
        end = start + segment_samples + overlap_samples
        segment = waveform[:, start:end].unsqueeze(0)
        # Add batch dimension
        batch.append(segment)

        # If batch is full or it's the last possible batch, process it
        if len(batch) == batch_size or (start + step > total_samples - segment_samples):
            concatenated_batch = torch.cat(batch, dim=0)  # Shape: [B, 1, SAMPLES]
            batch = []  # Reset the batch

            with torch.no_grad():
                encoded_frames = model.encode(concatenated_batch)
                # For each item in batch, extract non-overlapping part

                codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
                # Ignore the overlap in the output codes
                valid_start = (overlap_samples // 2) // 320
                valid_end = codes.shape[-1] - ((overlap_samples // 2) // 320)
                valid_codes = codes[:, :, valid_start:valid_end]
                all_codes.append(valid_codes)

                    # Concatenate all codes and save
    all_encoded = torch.cat(all_codes, dim=0).cpu().int()  # concatenate along the batch dimension
    torch.save(all_encoded, output_path)
    print(f"Saved encoded audio with shape {all_encoded.shape}")

if __name__ == '__main__':
    os.makedirs("./encoded", exist_ok=True)
    encode_audio_with_overlap_and_batching('/Users/felipebonetto/Code/lex-1/data/lexfridman/episodes/lex_ai_sam_altman_2.mp3', './encoded/lex_ai_sam_altman_2.pt')
