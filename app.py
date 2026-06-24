import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. UI ka Title aur Setup
st.set_page_config(page_title="My AI Tutor", page_icon="🤖")
st.title("🤖 Agentic Systems AI Tutor")

# 2. API Key Load karna
load_dotenv()
my_groq_api_key = os.getenv("GROQ_API_KEY")

# 3. Database ko Memory mein Save rakhna (Taaki har message par baar-baar load na ho)
@st.cache_resource
def setup_database():
    loader = TextLoader("my_notes.txt")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(document)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store

# Database load ho raha hai
with st.spinner("AI aapke notes padh raha hai..."):
    vector_store = setup_database()

# 4. Chat History Setup karna (Taaki purani baatein screen par dikhti rahein)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purane messages ko screen par dikhana
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User se Sawal Lena
user_input = st.chat_input("Apna sawal kisi bhi bhasha mein poochein (Hindi, English, etc.)...")

if user_input:
    # User ka message screen par dikhana
    st.chat_message("user").markdown(user_input)
    # User ka message memory mein save karna
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AI ka Engine Load karna
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=my_groq_api_key)

    # Database se notes dhoondhna
    milte_julte_notes = vector_store.similarity_search(user_input, k=2)
    context_text = ""
    for doc in milte_julte_notes:
        context_text += doc.page_content + "\n\n"

    # 6. AI ko MULTILINGUAL Instruction dena (Yeh line sabse zaroori hai)
    final_prompt = f"""Tum ek helpful aur smart AI tutor ho. 
    Neeche diye gaye 'Notes' ko padh kar user ke 'Sawal' ka detail mein jawab do. 
    Sabse Zaroori Rule: User jis bhasha (language) mein sawal pooche, tumhe apna jawab 100% usi bhasha mein dena hai.
    
    Notes:
    {context_text}
    
    Sawal: {user_input}
    """

    # AI ka Jawab generate karna aur UI par dikhana
    with st.chat_message("assistant"):
        with st.spinner("Llama 70B soch raha hai..."):
            jawab = llm.invoke(final_prompt)
            st.markdown(jawab.content)
            
    # AI ka jawab memory mein save karna
    st.session_state.messages.append({"role": "assistant", "content": jawab.content})