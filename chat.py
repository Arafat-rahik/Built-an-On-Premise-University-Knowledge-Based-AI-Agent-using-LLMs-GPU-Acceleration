import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama

# -----------------------------
# Initialize embeddings & DB
# -----------------------------
@st.cache_resource
def init_chroma():
    # Use free MiniLM embeddings
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to Chroma DB
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embed_model
    )
    return db

db = init_chroma()

# -----------------------------
# Initialize retrieval chain
# -----------------------------
@st.cache_resource
def init_qa_chain():
    # Ollama LLM (Mistral model)
    llm = Ollama(
        model="mistral:latest",  # Use the model you pulled via `ollama pull mistral:latest`
        verbose=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 6})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

qa_chain = init_qa_chain()

# -----------------------------
# Greeting Logic (ADDED ONLY)
# -----------------------------
def ask_question(question):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

    if question.lower().strip() in greetings:
        return "Hello! 👋 I'm your AUW knowledge assistant. Ask me anything about staff policies, student programs, alumni info, finance reports, or university services.", ""

# -----------------------------
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 AUW AI Assistant")

st.markdown("""
### 📌 Capabilities of this Assistant:

<ul>
<li>Answer inquiries with reference to the AUW Charter, Academic Bulletin, Staff and Faculty Handbook, and relevant policies related to HR, Procurement, IT, and other administrative areas.</li>
<li>Provide accurate, policy-based information drawn strictly from AUW’s official PDF documents.</li>
</ul>
""", unsafe_allow_html=True)

# Create a form so the user must click Submit
with st.form("question_form"):
    query = st.text_input("Ask something about your documents:")
    submitted = st.form_submit_button("🚀 Submit")

# Process only when user clicks the button
if submitted and query:
    greeting_reply = ask_question(query)

    if greeting_reply:
        st.write("**Answer:**")
        st.write(greeting_reply[0])
    else:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)

            st.write("**Answer:**")
            st.write(result['result'])

            st.write("---")
            st.write("**Source Documents:**")
            for doc in result['source_documents']:
                st.write(doc.page_content)
