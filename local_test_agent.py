# local_test_agent.py

import os
from PIL import Image
from dotenv import load_dotenv
import pytesseract
from PIL import Image

import pytesseract
from rag_utils import (
    split_text, embed_chunks, retrieve_top_chunks, generate_answer,
    crawl_browserstack_faq, crawl_browserstack_ai_terms
)

load_dotenv()
# Load static KB
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

AI_FAQ = load_file("ai_faqs.txt")
OPS_GUIDE = load_file("ai_ops_guidelines.txt")

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

# Final KB
print("Loading local knowledge base...")
IMAGE_KB = load_images_kb("knowledge_images") if os.path.exists("knowledge_images") else ""
FULL_KB = AI_FAQ + "\n" + OPS_GUIDE + "\n" + IMAGE_KB

# Precompute embeddings
print("Splitting and embedding KB...")
chunks = split_text(FULL_KB)
print(f"✅ {len(chunks)} chunks created.")
embeddings = embed_chunks(chunks)
print("Embeddings created")

# Loop
chat_history = []

while True:
    query = input("\nAsk your question (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    relevant_chunks = retrieve_top_chunks(query, chunks, embeddings)
    print("\n📄 Relevant Local Chunks:")
    if relevant_chunks:
        for i, (score, chunk) in enumerate(relevant_chunks, 1):
            print(f"\n--- Chunk {i} (Score: {score:.4f}) ---\n{chunk}")

    # Fallback sources
    faq_text = crawl_browserstack_faq()
    faq_chunks = split_text(faq_text)
    faq_embeddings = embed_chunks(faq_chunks)
    relevant_chunks1 = retrieve_top_chunks(query, faq_chunks, faq_embeddings)
    if relevant_chunks1:
        for i, (score, chunk) in enumerate(relevant_chunks1, 1):
            print(f"\n--- FAQ Chunk {i} (Score: {score:.4f}) ---\n{chunk}")

    ai_terms_text = crawl_browserstack_ai_terms()
    ai_terms_chunks = split_text(ai_terms_text)
    ai_terms_embeddings = embed_chunks(ai_terms_chunks)
    relevant_chunks2 = retrieve_top_chunks(query, ai_terms_chunks, ai_terms_embeddings)
    if relevant_chunks2:
        for i, (score, chunk) in enumerate(relevant_chunks2, 1):
            print(f"\n--- AI Terms Chunk {i} (Score: {score:.4f}) ---\n{chunk}")

    # Ensure all are lists, even if empty
    relevant_chunks = relevant_chunks or []
    relevant_chunks1 = relevant_chunks1 or []
    relevant_chunks2 = relevant_chunks2 or []

    final_relevant_chunks = relevant_chunks + relevant_chunks1 + relevant_chunks2

    if final_relevant_chunks:
        final_context = "\n\n".join([chunk for score, chunk in final_relevant_chunks])
        final_answer = generate_answer(query, final_context, chat_history)
        tagged_answer = final_answer + "\n cc @ai-ops"
        print("\n🔍 Answer:\n", tagged_answer.strip(), "\n\n")
    else:
        print("\n❌ Sorry, I couldn’t find a clear answer even from the FAQ & AI Terms. cc @ai-ops.\n")
        continue

    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
