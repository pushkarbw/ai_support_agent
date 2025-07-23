# rag_utils.py

from bs4 import BeautifulSoup
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import re
import requests
from urllib.parse import urljoin
import time


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def split_text(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 300:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        return [0.0] * 1536  # Return dummy vector of correct size
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]

# rag_utils.py
# Crawl BS AI FAQ pages

# Modified: Crawl BS AI FAQ pages recursively and save to file
def crawl_browserstack_faq():
    BASE_DOMAIN = "https://www.browserstack.com"
    ROOT_URL = "https://www.browserstack.com/support/faq/browserstack-ai"
    ALLOWED_PREFIX = ROOT_URL

    visited = set()
    all_text = []

    def scrape_text_from_page(url):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            page_content = []
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                text = tag.get_text(strip=True)
                if text:
                    page_content.append(text)

            return "\n".join(page_content), soup

        except Exception as e:
            return "", None

    def crawl_recursive(url):
        if url in visited or not url.startswith(ALLOWED_PREFIX):
            return

        visited.add(url)
        page_text, soup = scrape_text_from_page(url)

        if page_text:
            all_text.append(f"\n=== URL: {url} ===\n" + page_text)

        if soup:
            for a_tag in soup.find_all("a", href=True):
                next_link = urljoin(BASE_DOMAIN, a_tag["href"])
                if next_link.startswith(ALLOWED_PREFIX) and next_link not in visited:
                    time.sleep(0.5)
                    crawl_recursive(next_link)

    crawl_recursive(ROOT_URL)
    return "\n\n".join(all_text)



# Crawl AI Terms (single page only)
def crawl_browserstack_ai_terms():
    url = "https://www.browserstack.com/terms/ai-terms"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content = []

        # Extract readable text from paragraphs, lists, headings
        for tag in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']):
            text = tag.get_text(strip=True)
            if text:
                content.append(text)

        # Extract table contents
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                if row_text:
                    content.append(row_text)

        return "\n".join(content)

    except Exception as e:
        #print(f"❌ Failed to fetch {url}: {e}")
        return ""



def retrieve_top_chunks(query, chunks, embeddings, top_n=5, similarity_threshold=0.2):
    
    # 🔍 STEP 1: Proceed with normal embedding-based similarity
    query_embedding = get_embedding(query)
    scored_chunks = []


    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        score = cosine_similarity([query_embedding], [emb])[0][0]

        if score >= similarity_threshold:
            scored_chunks.append((score, chunk))
            

    if not scored_chunks:
        #print("⚠️ No chunks found above the threshold.")
        return None

    scored_chunks.sort(reverse=True)
    #print("\n📊 Top relevant chunks with similarity scores:")
    #for score, chunk in scored_chunks[:top_n]:
        #print(f"\nScore: {score:.4f}")

    return scored_chunks[:top_n]



def generate_answer(query, context, chat_history=None):
    if chat_history is None:
        chat_history = []

    messages = [
        {"role": "system", "content": "You are a helpful BrowserStack AI assistant. Use only relevant provided context.Do not mix unrelated content from previous question or answers. Do not hallucinate."},
    ]

    # Add chat history into the messages
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    # Add current question + context
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    # Completion call
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
