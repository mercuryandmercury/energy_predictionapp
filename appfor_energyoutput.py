
import streamlit as st
import pickle
import pandas as pd
import numpy as np



model=pickle.load(open('rf.pkl','rb'))

print(model)


 
    
def user_input():
    AT =st.sidebar.slider("temperature", 19.65,37.11,0.01)
    EV =st.sidebar.slider("exhaust_vacuum, in cm Hg", 54.30,81.56,0.01)
    AP =st.sidebar.slider("amb_pressure, in millibar",1013.25,1033.30,0.01)
    RH =st.sidebar.slider("r_humidity, in percentage", 73.30,100.16,0.01 )

    user_input_data = {
        'AT':AT,
        'EV':EV,
        'AP':AP,
        'RH':RH
        }

    input_data= pd.DataFrame(user_input_data,index=[0]) 
    return input_data






    your_input_data= user_input()
    st.header('input data')
    st.write(' your_input_data')
    
    
    
    
    
    energy = model.predict(your_input_data)
    st.subheader('energy predicted')
    st.subheader('MW'+str(np.round(energy[0],2)))
   

