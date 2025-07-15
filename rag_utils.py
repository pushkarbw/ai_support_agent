# rag_utils.py

import openai
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def split_text(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 600:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]

# rag_utils.py

def retrieve_top_chunks(query, chunks, embeddings, top_n=3, similarity_threshold=0.15):
    # ✅ STEP 2: Exact match fallback
    for i, chunk in enumerate(chunks):
        if query.lower() in chunk.lower():
            print(f"\n⚡ Found direct match in Chunk {i}")
            print(f"\n🔹 Direct Match Chunk:\n{chunk[:300]}...\n")
            return [chunk]

    # 🔍 STEP 1: Proceed with normal embedding-based similarity
    query_embedding = get_embedding(query)
    scored_chunks = []


    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        score = cosine_similarity([query_embedding], [emb])[0][0]

        if score >= similarity_threshold:
            scored_chunks.append((score, chunk))
            

    if not scored_chunks:
        print("⚠️ No chunks found above the threshold.")
        return None

    scored_chunks.sort(reverse=True)


    return [chunk for score, chunk in scored_chunks[:top_n]]


def generate_answer(query, context, chat_history=None):
    if chat_history is None:
        chat_history = []

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for BrowserStack queries. Use only the context provided."},
    ]

    # Add chat history into the messages
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    # Add current question + context
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    # Completion call
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content.strip()

