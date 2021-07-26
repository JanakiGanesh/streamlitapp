#importing the libraries
import streamlit as st 
# import pickle
# from PIL import Image
# from skimage.transform import resize
import numpy as np
import tensorflow as tf
# import time
# import keras
from tensorflow import keras
# import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
st.markdown(
   """
   <style>
   .reportview-container {
       background: url("https://img.freepik.com/free-photo/abstract-grunge-decorative-relief-navy-blue-stucco-wall-texture-wide-angle-rough-colored-background_1258-28311.jpg?size=626&ext=jpg")
      
      
   }
 
   </style>
   """,
   unsafe_allow_html=True
)



# model = tf.keras.models.load_model('best_model2.h5')

st.write("""
         # Face Detection Application Window
         """
         )
st.write("This is a simple image classification web app to predict real or fake images")
image_ = st.file_uploader("Please upload an image file", type=["jpg", "jpeg"])



def import_and_predict(image_data, model):
        # hardik_img = image.load_img(image_data, target_size=(128, 128),color_mode='rgb',grayscale=False)
        size = (128,128) 
        images = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     # # Preprocessing the image
        pp_hardik_img = image.img_to_array(images,data_format=None, dtype=None)
        pp_hardik_img = pp_hardik_img/255
        pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)
        predictions = model.predict_classes(pp_hardik_img)
        return predictions
if image_ is None:

    st.text("Please upload an image file")
else:

    imagee = Image.open(image_)
    st.image(imagee, use_column_width=True)


# if image_ is not None:
col1, col2, col3 = st.beta_columns([1,1,1])    
if col2.button("Click Here to predict"):
   col1, col2, col3 = st.beta_columns([1,1,1]) 
#    st.title("Off center :(") 
   
#    col2.title("Centered! :)") 

#   loaded_model = pickle.load(open('best_model2.h5', 'rb'))
#   test_result = saved_model2.evaluate(test_images, test_labels_reshape)
#   result = loaded_model.predict(test_result)
#   st.write(result)

    # hardik_img = Image.open(image_)
    # hardik_img = image.load_img(image_, target_size=(128, 128),color_mode='rgb',grayscale=False)
#     # # Preprocessing the image
    # pp_hardik_img = image.img_to_array(hardik_img)
    # pp_hardik_img = pp_hardik_img/255
    # pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)
    # imagee = Image.open(image_)
   model = tf.keras.models.load_model('best_model2.h5')
    # st.image(imagee, use_column_width=True)
    # size = (128,128)    
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
        
    # img_reshape = img_resize[np.newaxis,...]

   prediction = import_and_predict(imagee, model)
    # prediction = model.predict_classes(img_reshape)
    

   if (prediction == 1).all(): 
      col2.title('It is REAL!'.format(prediction[0]))
    
   elif (prediction == 0).all():
      col2.title('Alert!! FAKE'.format(1-prediction[0]))


        
    
    
    
        
    
    
    
