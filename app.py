import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Streamlit Sayfa Ayarları
st.set_page_config(page_title="Koç V1", page_icon="⚡", layout="wide")
st.title("⚡ KOÇ V1 - Elit Atletik Performans & Program Sistemi")

# API Key Yönetimi
st.sidebar.title("⚙️ Sistem Ayarları")
api_key = st.sidebar.text_input("Google Gemini API Key:", type="password")

if not api_key:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        st.warning("Lütfen sol menüden Gemini API Key anahtarını gir kral!")
        st.stop()

# PDF Yükleme ve Vektör Veritabanı
@st.cache_resource
def load_vectorstore(key):
    loader = PyPDFLoader("kitap.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = load_vectorstore(api_key)

# Koç V1 System Prompt v1.1
system_prompt = (
    "Senin adın Koç V1. Karşındaki sporcu, nihai hedefi GALATASARAY forması giymek olan, "
    "mevkisinde (stoper/sağ bek/sağ kanat) elit seviyeye ulaşmak isteyen disiplinli bir futbolcudur. "
    "Sporcunun elinde 2x 5 kg dambıl, 10 kg halter ve 5 seviyeli direnç bandı vardır. "
    "Referans verileri: T-Testi 8.29 saniye, Şınav 40 tekrar, 10x20 sprint yorgunluk puanı 4.5. "
    "Sporcuya HER ZAMAN 'kral' diye hitap et. "
    "Soru ve talepleri SADECE sana sağlanan şu 236 sayfalık kaynak metne göre yanıtla: \n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Chat Arayüzü
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Bugünkü durumun, verilerin veya sorun nedir kral?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Kitap taranıyor ve program sentezleniyor..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
