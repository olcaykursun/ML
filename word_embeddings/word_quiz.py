import numpy as np
import random

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
terms_list = {
    'community': 'A group of people living in the same place or having a particular characteristic in common.',
    'laws': 'The system of rules which a particular country or community recognizes as regulating the actions of its members.',
    'conflict': 'A serious disagreement or argument.',
    'common good': 'The benefit or interests of all or most members of a community.',
    'government': 'The governing body of a nation, state, or community.',
    'governor': 'An official appointed to govern a state or region.',
    'equality': 'The state of being equal, especially in status, rights, or opportunities.',
    'fairness': 'The quality of making judgments that are free from discrimination.',
    'responsibility': 'The state or fact of having a duty to deal with something or of having control over someone.',
    'mayor': 'The elected head of a city, town, or other municipality.',
    'conflict resolution': 'The process of resolving a dispute or a conflict by meeting at least some of the needs of each side.'
}

# Get embeddings for terms if they exist in the model
embeddings_list = {}
for term in terms_list:
    try:
        embeddings_list[term] = wv[term]
    except KeyError:
        # Term not in Word2Vec vocabulary
        embeddings_list[term] = None

def get_quiz_choices(term, embeddings_list):
    if embeddings_list[term] is None:
        # If the term embedding is missing, return 3 random terms excluding the main term
        return random.sample([t for t in terms_list.keys() if t != term], 3)
    else:
        # Exclude None values and compute distances
        valid_embeddings = [v for v in embeddings_list.values() if v is not None]
        norms = np.linalg.norm(valid_embeddings, axis=1).reshape(-1,1)        
        valid_embeddings /= norms
        similarity = np.dot(valid_embeddings, embeddings_list[term])
        
        # Sort terms based on computed distances
        valid_terms = [t for t in terms_list.keys() if embeddings_list[t] is not None]
        closest_terms = sorted(zip(valid_terms, similarity), key=lambda x: x[0], reverse=True)
        
        # Get the 3 most similar terms to the given term
        similar_terms = [t for t, _ in closest_terms[1:4]]
        return similar_terms

# Quiz loop
for term, description in terms_list.items():
    print(f"\nDescription: {description}")
    print("\nWhich term is described above?")

    # Get answer choices
    choices = [term] + get_quiz_choices(term, embeddings_list)
    random.shuffle(choices)  # Shuffle the choices to make it unpredictable

    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")

    # Get user's answer
    answer = input("Your choice (1-4): ")

    if choices[int(answer)-1] == term:
        print("Correct!")
    else:
        print(f"Wrong. The correct answer is {term}.")
