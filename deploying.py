import streamlit as st
import joblib  
import time 


st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Broadridge Question Answering')
st.text('What would you like to know?')

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('./models/bert_qa_custom.joblib')
    return model

with st.spinner('Lpadomg Model Into Memory....'):
    model = load_model()

text = st.text_input('Enter your question here..')
if text:
    st.write('Response :')
    with st.spinner('Searching for answers.....'):
        predicion = model.predict(text)        
        st.write(f'amswer: {predicion[0]}')
        st.write(f'title: {predicion[1]}')
        st.write(f'paragraph: {predicion[2]}')

    st.write('')
    
        