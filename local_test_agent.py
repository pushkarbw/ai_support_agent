# local_test_agent.py

import os
from bs4 import BeautifulSoup
import requests
import pytesseract
from PIL import Image
from rag_utils import split_text, embed_chunks, retrieve_top_chunks, generate_answer


# Load local KB files
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

AI_FAQ = load_file("ai_faqs.txt")
OPS_GUIDE = load_file("ai_ops_guidelines.txt")

# Scrape all internal/external articles from AI FAQ page
def crawl_browserstack_articles(start_url, path_starts_with):
    base_url = "https://www.browserstack.com"
    visited_links = set()
    full_text = ""

    try:
        res = requests.get(start_url)
        soup = BeautifulSoup(res.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip irrelevant external links (for AI FAQ only)
            if not href.startswith("/") and not href.startswith(base_url):
                continue

            # Normalize
            if href.startswith(base_url):
                rel_path = href[len(base_url):]
            else:
                rel_path = href

            if rel_path.startswith(path_starts_with) and rel_path not in visited_links:
                visited_links.add(rel_path)
                full_url = base_url + rel_path
                try:
                    article_res = requests.get(full_url)
                    soup2 = BeautifulSoup(article_res.text, "html.parser")
                    full_text += "\n\n--- " + full_url + " ---\n" + soup2.get_text()
                except Exception as e:
                    print(f"❌ Failed to fetch {full_url}: {e}")
    except Exception as e:
        print(f"❌ Error loading base page {start_url}: {e}")
    return full_text

# Crawl external links from AI Terms page too
def fetch_ai_terms_main_page_only():
    url = "https://www.browserstack.com/terms/ai-terms"
    try:
        print(f"Fetching AI Terms page: {url}")
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return f"\n\n--- {url} ---\n{text}"
    except Exception as e:
        print(f"❌ Failed to fetch AI Terms page: {e}")
        return ""



# OCR any image KBs if needed
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

# Load all KBs
print("Loading knowledge base...")
FAQ_WEB = crawl_browserstack_articles("https://www.browserstack.com/support/faq/browserstack-ai", "/support/faq/browserstack-ai/")
AI_TERMS_SITE_TEXT = fetch_ai_terms_main_page_only()
IMAGE_KB = load_images_kb("knowledge_images") if os.path.exists("knowledge_images") else ""

FULL_KB = AI_FAQ + "\n" + OPS_GUIDE + "\n" + FAQ_WEB + "\n" + AI_TERMS_SITE_TEXT + "\n" + IMAGE_KB

print("Knowledge Base ready. Ask your question!")


# Step 1: Split and embed KB
print("📖 Splitting KB into chunks...")
chunks = split_text(FULL_KB)
print(f"✅ {len(chunks)} chunks created.")



print("🔁 Creating embeddings...")
embeddings = embed_chunks(chunks)
print("✅ Embeddings created")

# Step 2: User query loop with history
chat_history = []

while True:
    query = input("\nAsk your question (or type 'exit'): ")
    if query.strip().lower() == "exit":
        break

    relevant_chunks = retrieve_top_chunks(query, chunks, embeddings)

    if not relevant_chunks:
        print("\n❌ Sorry, I couldn’t find a clear answer. cc @ai-ops.")
        continue

    context = "\n\n".join(relevant_chunks)
    answer = generate_answer(query, context, chat_history=chat_history)
    tagged_answer = answer + "\n cc @ai-ops"

    print("\n💬 Answer:\n", tagged_answer)
    chat_history.append((query, tagged_answer))
    # 🧹 Clean history if too long
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]







