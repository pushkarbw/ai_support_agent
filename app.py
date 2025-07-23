import time
import streamlit as st
from rag_utils import (
    split_text, embed_chunks, retrieve_top_chunks, generate_answer,
    crawl_browserstack_faq, crawl_browserstack_ai_terms
)
from PIL import Image
import os
import pytesseract


from datetime import datetime, timedelta



FAQ_CACHE_FILE = "cached_faq_text.txt"
AI_TERMS_CACHE_FILE = "cached_ai_terms_text.txt"
FAQ_TIMESTAMP_FILE = "faq_cache_timestamp.txt"
AI_TERMS_TIMESTAMP_FILE = "ai_terms_cache_timestamp.txt"

def should_refresh_faq_cache():
    if not os.path.exists(FAQ_CACHE_FILE) or os.path.getsize(FAQ_CACHE_FILE) == 0:
        return True
    if not os.path.exists(FAQ_TIMESTAMP_FILE):
        return True
    try:
        with open(FAQ_TIMESTAMP_FILE, "r") as f:
            last_ts = float(f.read().strip())
            return (datetime.now() - datetime.fromtimestamp(last_ts)) > timedelta(days=7)
    except:
        return True

def update_faq_cache_timestamp():
    with open(FAQ_TIMESTAMP_FILE, "w") as f:
        f.write(str(time.time()))


def should_refresh_ai_terms_cache():
    if not os.path.exists(AI_TERMS_CACHE_FILE) or os.path.getsize(AI_TERMS_CACHE_FILE) == 0:
        return True
    if not os.path.exists(AI_TERMS_TIMESTAMP_FILE):
        return True
    try:
        with open(AI_TERMS_TIMESTAMP_FILE, "r") as f:
            last_ts = float(f.read().strip())
            return (datetime.now() - datetime.fromtimestamp(last_ts)) > timedelta(days=7)
    except:
        return True

def update_ai_terms_cache_timestamp():
    with open(AI_TERMS_TIMESTAMP_FILE, "w") as f:
        f.write(str(time.time()))



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

    # Refresh web FAQ Terms weekly
    if should_refresh_faq_cache():
        st.write("🔄 Scraping BrowserStack Public FAQ Pages. Please Wait / DO NOT REFRESH......")
        faq_text = crawl_browserstack_faq()
        with open(FAQ_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(faq_text)
        update_faq_cache_timestamp()
    else:
        faq_text = open(FAQ_CACHE_FILE, encoding="utf-8").read()
    
    # Refresh AI Terms weekly
    if should_refresh_ai_terms_cache():
        st.write("🔄 Scraping BrowserStack AI Terms & Conditions Pages. Please Wait / DO NOT REFRESH....")
        ai_text = crawl_browserstack_ai_terms()
        with open(AI_TERMS_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(ai_text)

        update_ai_terms_cache_timestamp()
    else:
        ai_text = open(AI_TERMS_CACHE_FILE, encoding="utf-8").read()

    AI_FAQ = load_file("ai_faqs.txt")
    OPS_GUIDE = load_file("ai_ops_guidelines.txt")
    IMAGE_KB = load_images_kb("knowledge_images") if os.path.exists("knowledge_images") else ""

    FULL_KB = AI_FAQ + "\n" + OPS_GUIDE + "\n" + IMAGE_KB 

    st.write("🔄 Creating Chunks and Embeddings. Please wait / DO NOT REFRESH.....")
    chunks = split_text(FULL_KB)
    embeddings = embed_chunks(chunks)

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings

    faq_chunks = split_text(faq_text)
    faq_embeddings = embed_chunks(faq_chunks)

    st.session_state.faq_chunks = faq_chunks
    st.session_state.faq_embeddings = faq_embeddings

    ai_terms_chunks = split_text(ai_text)
    ai_terms_embeddings = embed_chunks(ai_terms_chunks)

    st.session_state.ai_terms_chunks = ai_terms_chunks
    st.session_state.ai_terms_embeddings = ai_terms_embeddings
    st.write("✅ Chunks and Embeddings Created")
    st.write("✅ Knowledge Base Loading Complete")
    


st.title("🧠 BrowserStack AI Support Agent")

with st.form("chat_form"):
    query = st.text_area("Ask your question:", height=100)
    submitted = st.form_submit_button("Submit")

if submitted and query.strip():
    with st.spinner("Thinking..."):
        
        
        relevant_chunks = retrieve_top_chunks(query, st.session_state.chunks, st.session_state.embeddings)
        relevant_chunks1 = retrieve_top_chunks(query, st.session_state.faq_chunks , st.session_state.faq_embeddings)
        relevant_chunks2 = retrieve_top_chunks(query, st.session_state.ai_terms_chunks, st.session_state.ai_terms_embeddings)

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
