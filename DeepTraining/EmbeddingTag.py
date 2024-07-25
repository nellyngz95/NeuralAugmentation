#This program is to create a .csv before training the model
#The .csv will contain the file path and the label of the file
import glob
import os   
import numpy as np  
import pandas as pd
#path to the embeddings
file_dir = '/homes/nva01/EmbeddingsWav2Vec2T'

# Initialize lists to store data
file_names = []
embeddings = []
class_labels = []

# Iterate over each file in the directory
for file_path in glob.glob(f"{file_dir}/**/*.npy", recursive=True):
    try:
        # Load the numpy array
        embedding = np.load(file_path)
        
        # Extract file name with extension
        fname = os.path.basename(file_path)
        
        # Extract class label from file name (assuming it's the number after the last hyphen before .npy)
        class_label = int(fname.split('-')[-1].split('.')[0])
        
        # Append file name, embedding, and class label to lists
        file_names.append(fname)
        embeddings.append(embedding)
        class_labels.append(class_label)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Determine the number of unique classes
unique_classes = sorted(set(class_labels))
num_classes = len(unique_classes)

# Create a mapping from class label to one-hot encoding index
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

# Create one-hot encodings for each class label
one_hot_encodings = np.eye(num_classes)[[class_to_index[label] for label in class_labels]]

# Create a DataFrame to store data
data = {
    'filename': file_names,
    'Embedding': embeddings,
    'label': class_labels,
    'One-HotEncoding': list(one_hot_encodings)
}
df = pd.DataFrame(data)

# Define the path 
csv_file_path = '/homes/nva01/EmbeddingsWav2Vec2T.csv'

# Write DataFrame to CSV
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")
