import streamlit as st
import pickle
import os
import numpy as np
from skimage.transform import resize
from PIL import Image



st.title('Image Classifier Using Machine Learning')
st.markdown(" **Upload Image!!**")
model = pickle.load(open('img_model.p','rb'))
uploaded_file = st.file_uploader("Choose an Image -->",type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image')

    if st.button('Predict'):
        CATEGORIES = ['basketball','cricket bats','ice cream cone']
        st.header('Results: ')
        flat_data = []
        img = np.array(img)
        img_resize = resize(img,(150,150,3))
        flat_data.append(img_resize.flatten())
        flat_data = np.array(flat_data)
        # print(img.shape)
        # plt.imshow(img_resize)
        y_out = model.predict(flat_data)
        y_out = CATEGORIES[y_out[0]]
        st.write(f'Predicted Output :  {y_out}')
        q = model.predict_proba(flat_data)
        for index,item in enumerate(CATEGORIES):
            st.text(f'{item} : {q[0][index]*100} %')
st.caption('Created by @Sangram')
