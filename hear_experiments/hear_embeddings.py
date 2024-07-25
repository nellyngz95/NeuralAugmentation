import torch
import hearbaseline.wav2vec2
import hearbaseline
import torchaudio 
import numpy as np
import glob
import os
# We are uploading both Naive and Wav2Vec2 models 
#model = hearbaseline.load_model("/homes/nva01/hear-baseline/saved_models/naive_baseline.pt")
model=hearbaseline.wav2vec2.load_model(model_file_path="")
sample_rate = model.sample_rate
#Uploading the dataset
file_path = '/homes/nva01/TransformedDataset'
#Creating a directory to save the embeddings
emb_dir='/homes/nva01/EmbeddingsWav2Vec2T'
os.mkdir(emb_dir)
# Loop over all audio files in the dataset
for  index, file_path in enumerate(glob.glob(f"{file_path}/**/*.wav", recursive=True)):
    print("files processed: ", index+1)
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)
    # Compute embeddings
    waveform=waveform.to("cuda")
    #embeddings = hearbaseline.get_scene_embeddings(waveform,model)
    embeddings=hearbaseline.wav2vec2.get_scene_embeddings(waveform, model)
    embeddings = embeddings.cpu()
    
    # Save embeddings
    np.save(f"{emb_dir}/{os.path.basename(file_path)}.npy", embeddings)
    print(embeddings.shape)
    print(embeddings)
    print(f"{emb_dir}/{os.path.basename(file_path)}.npy")

