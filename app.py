import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
st.title('Face Mask Removal App')


# Create two columns for neat side-by-side file uploading
col1, col2 = st.columns(2)

with col1:
    st.subheader("Facemask Image")
    img1 = st.file_uploader("Upload image with facemask", type=["jpg", "jpeg", "png"], key="upload1")
    if img1:
        st.image(img1, caption="Facemask Image Preview", use_container_width=True)

with col2:
    st.subheader("Mask")
    img2 = st.file_uploader("Upload Mask", type=["jpg", "jpeg", "png"], key="upload2")
    if img2:
        st.image(img2, caption="Mask Preview", use_container_width=True)

st.divider()

models = ('Encoder–Decoder','U-NET') #

selected_model = st.selectbox('Select generator model', models)#
# Enable button only when BOTH images are uploaded
if img1 is not None and img2 is not None:
    if st.button("Process", type="primary"):
        # Save both image byte payloads into session_state
        st.session_state["img1_data"] = img1.getvalue()
        st.session_state["img2_data"] = img2.getvalue()
        st.session_state['model_data']= selected_model #
        
        # Navigate to result page
        st.switch_page("pages/result_page.py")
else:
    st.info("Please upload both images to proceed.")





