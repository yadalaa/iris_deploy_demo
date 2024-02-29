import pickle
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
from PIL import Image

picklein = open("model_iris.pkl","rb")
classifier = pickle.load(picklein)

def predictiris_variety(sepal_length,sepal_width,petal_length,petal_width):
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    return prediction

def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://scontent.fbkk10-1.fna.fbcdn.net/v/t39.30808-6/292023018_408956564587890_1658677285667301104_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=efb6e6&_nc_eui2=AeE9xAUZmo93d7CKKCEjvEmFFJIlke2no90UkiWR7aej3Xla21RYtGlAKhqnl0e5cv8omO2IhubcQ9fYTbdo85DB&_nc_ohc=mFF7wMRlhwAAX9L9BsY&_nc_ht=scontent.fbkk10-1.fna&oh=00_AfB3SnZ7alI5vEzO9zUQzqmE4t-eX-1uoLUODrpEtlA82Q&oe=65E61BB0", width=600)

    st.markdown("You are using Streamlit...",unsafe_allow_html=True)
    sepal_length = st.text_input("Enter Sepal Length" ,".")
    sepal_width = st.text_input("Enter Sepal width" ,".")
    petal_length = st.text_input("Enter Petal Length" ,".")
    petal_width = st.text_input("Enter Petal width" ,".")

    result = ""
    if st.button("Click here to Predict"):
        result = predictiris_variety(sepal_length, sepal_width, petal_length, petal_length)
        st.balloons()
    st.success('The output is {}' .format(result))

if __name__ == '__main__':
    Input_Output()