import streamlit as st
from fastai.learner import Learner, load_learner
from fastai.vision.core import PILImage

"""
# Daschund Type Classifier

This app classifies images of dogs as one of:

* Short Haired
* Long Haired
* Random Dog

Upload an image of a dog and it will automatically be classified as one of the three types of dogs listed above.
"""

model: Learner = load_learner('model.pkl')

file = st.file_uploader(
  'Select an image of a dog', 
  type = None, 
  accept_multiple_files = False
)

if file is not None:
  img = PILImage.create(file)
  st.image(file, caption = 'Your selection', width = 256)
  
  dog_type, _, probs = model.predict(img)

  f"This is a **{dog_type}** with probabilities:\n"
  max_prob = 0
  for i, t in enumerate(model.dls.vocab):
    prob = round(float(probs[i]), 3)
    max_prob = round(max(max_prob, prob), 3)

    f'\t=> {t}: {prob}'