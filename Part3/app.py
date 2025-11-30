import streamlit as st
from utils import process_pdf
from rag import get_simple_rag_chain, get_fusion_chain
from router import route_and_execute
from critic import run_critic

# Streamlit init
st.set_page_config(page_title="RAG Fusion Bot")
st.title("Assignments RAG-Fusion Chatbot")


# sidebar settings
with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", ["Standard RAG", "RAG Fusion", "Auto-Routing"])
    
    # PDF uploading logic
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = process_pdf(uploaded_file)
        st.success("PDF Processed")

    st.divider()
    
    # critic button
    if st.button("Run LLM Critic"):
        
        if "last_query" in st.session_state and "vectorstore" in st.session_state:
            run_critic(st.session_state["last_query"], st.session_state["vectorstore"])
            st.info("Check Terminal for Report!")
            
        else:
            st.warning("Ask a question first!")



# chat persistance 
if "messages" not in st.session_state: 
    st.session_state["messages"] = []

for msg in st.session_state["messages"]: 
    st.chat_message(msg["role"]).markdown(msg["content"])


# input handling
if prompt := st.chat_input("Ask away..."):
    
    st.session_state["last_query"] = prompt
    
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        vs = st.session_state.get("vectorstore")
        
        if not vs: 
            st.error("Upload PDF first.")
        else:
            # Mode Switching Logic
            if mode == "Standard RAG":
                resp = get_simple_rag_chain(vs).invoke(prompt)
            elif mode == "RAG Fusion":
                resp = get_fusion_chain(vs).invoke(prompt)
            else: 
                # Auto-Routing (Section 6.1 & 6.2)
                resp, cat = route_and_execute(prompt, vs)
                # Visual cue to show which chain was selected
                resp = f"**[Mode: {cat}]**\n{resp}"
            
            st.markdown(resp)
            st.session_state["messages"].append({"role": "assistant", "content": resp})