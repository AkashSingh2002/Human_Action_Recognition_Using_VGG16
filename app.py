import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.models.load_model(r'C:\Users\pk boss\updated_VGG16.h5')

classes = ["sitting","using laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing",
           "eating","fighting","listening_to_music","running",
           "texting"
        ]

st.title("Human Action Recognition App")
st.write("Upload an image and let the model classify the human action!")

# Function to make predictions
def predict_action(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    action_index = np.argmax(model.predict(img_array), axis=1)[0]
    action_index = classes[action_index]
    return action_index

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    action = predict_action(uploaded_file)
    st.write(f"The predicted action is: {action}")
