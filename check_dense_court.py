import numpy as np

embedding = np.load('./data/processed/_dense_court/0.npy')

print(embedding.shape)
print(embedding[0].shape)