import streamlit as st
from ultralytics import YOLO

model = YOLO("best2.pt")

def app():

    st.title("LIVE Webcam Feed")
    run=st.checkbox("Run Webcam")

    while run:
        try:
            results = model.predict(source ="0", show=True)
            st.info(results)
        except:
            pass

#Runnning the app on Streamlit
app()