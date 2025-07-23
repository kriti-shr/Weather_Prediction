import streamlit as st 
import joblib 
import pandas as pd

# laod the model 
model = joblib.load(r'C:\Python\model_weather.pt')

# function to encode and predcit data 
def predict(data): 
    data=data.dropna()
    df=data.copy()
    df = encode_loc(df)
    df = encode_sea(df)
    df = encode_cloud(df)
    # df = decode_weather(data, df)
    # after encode 
    df['predicted']=model.predict(df)

    return df

def encode_loc(df):
    enc_loc = {'inland':1, 'mountain':2, 'coastal':3}
    df=df.drop('Weather Type',axis=1)
    df['Location']=df['Location'].map(enc_loc)
    return df

def encode_sea(df):
    enc_sea = {'Winter':1, 'Spring':2, 'Summer':3, 'Autumn':4}
    df['Season']=df['Season'].map(enc_sea)
    return df

def encode_cloud(df):
    enc_cloud = {'partly cloudy':1, 'clear':2, 'overcast':3, 'cloudy':4}
    df['Cloud Cover']=df['Cloud Cover'].map(enc_cloud)
    return df
def decode_weather(df):
    enc_wea = {1:'Rainy', 2:'Cloudy',3: 'Sunny', 4:'Snowy'}
    df['predicted']=df['predicted'].map(enc_wea)
    return df

# title of applciation 
st.title("Weather Prediction")
file=st.file_uploader("Upload Your file", type='csv')
try:
    if file is not None:
        data=pd.read_csv(file)
        st.write("first five rows")
        st.write(data.head())
        df_p=predict(data)
        df_p = decode_weather(df_p)

        st.write("Data with prediction")
        st.write(df_p)
    else:
        st.write("Empty file cannot be read")
except Exception as e: 
    st.write(f"Error {e} occured")

finally: 
    st.write("Thank you for using our service ")