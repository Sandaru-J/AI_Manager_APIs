from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
# from langchain import PromptTemplate, LLMChain
import requests
import streamlit as st 

HUGGINGFACEHUB_API_TOKEN='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'

load_dotenv(find_dotenv())

def img2txt(url):
    print('Image to text started')
    image_to_text = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large')

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return(text)

# img2txt('A.JPG')

def txtTrans(text):
    print('Text Translation started')
    API_URL = "https://api-inference.huggingface.co/models/google-t5/t5-small"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": text,
    })
    print(output)
    specified_part = output[0]['translation_text']
    print(specified_part)
    return specified_part

def text2speech(text):
    print('Text to speech started')
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads ={
        'inputs':text
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audo.flac', 'wb') as file:
        file.write(response.content)
    print('success')


# text = 'several people standing in a room with a clock and a white board'

# text=img2txt('D.JPG')
# text =txtTrans(text)
# text2speech(text)

def main():
    st.set_page_config(page_title='img 2 audio story')

    st.header('AI Manager')
    uploaded_file = st.file_uploader('Choose image...', type='jpg')

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
                  file.write(bytes_data)
        st.image(uploaded_file, caption = 'Uploaded Image.', use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        
        
        with st.expander('Text'):
             st.write(scenario)
        text2speech(scenario)
        st.audio('audo.flac')

if __name__ == '__main__':
    main()