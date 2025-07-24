from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate, load_prompt

api_key=st.secrets["HuggingFace_API_Key"]

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    HUGGINGFACEHUB_API_TOKEN=api_key
    
)

# loading the template from json file
template=load_prompt('template.json')

load_dotenv()

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

#user_input=st.text_input('Enter Your Prompt')


# Creating selectbox
paper_input=st.selectbox("Select Research Paper Name",["Attenttion Is All You Need ","BERT: Pre-training of Deep Bidirection Transformers ","GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input=st.selectbox("Select Expanation Style",["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input=st.selectbox("Select Explanation Length",["Short(1-2 paragraphs)","Medium (3-5 paragraphs)","Long (detailed explanation)"])







#Creating Button
if st.button('Summarize'):
    # Creating chain so that we avoid calling invoke multiple times
    chain=template|model

    result=chain.invoke(
        # fill the placeholders
            {
        'paper_input':paper_input,
        'style_input': style_input,
        'length_input': length_input
    }

    )
    #result=model.invoke(prompt)
    st.write(result.content)
    