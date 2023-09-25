import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if 'wv' not in globals():
  import os
  import gensim.downloader as api

  # Path to save/load the model
  model_path = '/Users/okursun/uk/word2vec-google-news-300'  #good for GDrive

  # Check if model exists in Google Drive
  if not os.path.exists(model_path):
      wv = api.load('word2vec-google-news-300')
      wv.save(model_path)
  else:
      from gensim.models import KeyedVectors
      wv = KeyedVectors.load(model_path)

#%%
train_words = ['king','man','woman', 'girl', 'boy', 'queen']
test_words = ['student', 'professor', 'rector', 'nurse', 'police', 'soldier', 'housekeeper', 'bodybuilder', 'doctor']
all_words = train_words + test_words

embedding_matrix = []
for word in all_words:
    if word != 'null':
        embedding_matrix.append(wv[word])
    else:
        embedding_matrix.append(np.zeros_like(wv['null']))
        
np.array([wv[word] for word in all_words])

# Compute PCA for the train words
pca = PCA(n_components=2,)
transformed_embeddings = pca.fit_transform(embedding_matrix[:len(train_words)])

# Project the test words onto the PCA space
projected_candidates = pca.transform(embedding_matrix[len(train_words):])

# Visualize the PCA projections
plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], color='blue', label='Train')
for i, word in enumerate(train_words):
    plt.annotate(word, (transformed_embeddings[i, 0], transformed_embeddings[i, 1]))

plt.scatter(projected_candidates[:, 0], projected_candidates[:, 1], color='red', label='Test')
for i, word in enumerate(test_words):
    plt.annotate(word, (projected_candidates[i, 0], projected_candidates[i, 1]))

plt.legend()
plt.show()
