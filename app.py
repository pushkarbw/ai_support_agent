import streamlit as st
from rag_utils import (
    split_text, embed_chunks, retrieve_top_chunks, generate_answer,
    crawl_browserstack_faq, crawl_browserstack_ai_terms
)
from PIL import Image
import os
import pytesseract

# Load local files
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_images_kb(folder_path):
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, filename)
            try:
                img = Image.open(path)
                text = pytesseract.image_to_string(img)
                combined_text += f"\nFrom Image: {filename}\n{text}\n"
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return combined_text


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.write("🔄 Loading knowledge base...")
    AI_FAQ = load_file("ai_faqs.txt")
    OPS_GUIDE = load_file("ai_ops_guidelines.txt")
    IMAGE_KB = load_images_kb("knowledge_images") if os.path.exists("knowledge_images") else ""
    FULL_KB = AI_FAQ + "\n" + OPS_GUIDE + "\n" + IMAGE_KB
    chunks = split_text(FULL_KB)
    embeddings = embed_chunks(chunks)
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings

st.title("🧠 BrowserStack AI Support Agent")

with st.form("chat_form"):
    query = st.text_area("Ask your question:", height=100)
    submitted = st.form_submit_button("Submit")

if submitted and query.strip():
    with st.spinner("Thinking..."):
        relevant_chunks = retrieve_top_chunks(query, st.session_state.chunks, st.session_state.embeddings)
        faq_text = crawl_browserstack_faq()
        faq_chunks = split_text(faq_text)
        faq_embeddings = embed_chunks(faq_chunks)
        relevant_chunks1 = retrieve_top_chunks(query, faq_chunks, faq_embeddings)

        ai_terms_text = crawl_browserstack_ai_terms()
        ai_terms_chunks = split_text(ai_terms_text)
        ai_terms_embeddings = embed_chunks(ai_terms_chunks)
        relevant_chunks2 = retrieve_top_chunks(query, ai_terms_chunks, ai_terms_embeddings)

        relevant_chunks = relevant_chunks or []
        relevant_chunks1 = relevant_chunks1 or []
        relevant_chunks2 = relevant_chunks2 or []
        final_relevant_chunks = relevant_chunks + relevant_chunks1 + relevant_chunks2

        if final_relevant_chunks:
            context = "\n\n".join([chunk for score, chunk in final_relevant_chunks])
            final_answer = generate_answer(query, context, st.session_state.chat_history)
            tagged_answer = final_answer + "\ncc @ai-ops"
        else:
            tagged_answer = "❌ Sorry, no clear answer found. cc @ai-ops"

        st.session_state.chat_history.append((query, tagged_answer))
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

# Display chat
st.markdown("### 💬 Chat History")
for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
    with st.expander(f"**Q{i}: {q}**", expanded=True):
        st.markdown(f"**Answer:** {a}")
