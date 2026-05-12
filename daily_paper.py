import arxiv
import google.generativeai as genai
import requests
import os
from datetime import datetime, timedelta, timezone

# --- 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "history.txt"
RETENTION_DAYS = 10.5

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def manage_history():
    now = datetime.now(timezone.utc)
    valid_history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            for line in f:
                if "|" in line:
                    parts = line.strip().split("|")
                    if len(parts) == 2:
                        paper_id, ts_str = parts
                        ts = datetime.fromisoformat(ts_str)
                        if now - ts < timedelta(days=RETENTION_DAYS):
                            valid_history[paper_id] = ts_str
    return valid_history

def save_history(history_dict):
    with open(HISTORY_FILE, "w") as f:
        for paper_id, ts_str in history_dict.items():
            f.write(f"{paper_id}|{ts_str}\n")

def get_papers_from_sources():
    query = 'cat:cs.AI OR cat:cs.LG OR abs:"transformer" OR abs:"llm" OR abs:"diffusion"'
    search = arxiv.Search(query=query, max_results=40, sort_by=arxiv.SortCriterion.SubmittedDate)
    history = manage_history()
    new_papers = []
    for result in search.results():
        paper_id = result.entry_id.split('/')[-1]
        if paper_id not in history:
            new_papers.append({"id": paper_id, "title": result.title, "summary": result.summary, "url": result.pdf_url})
    return new_papers, history

def select_top_3(papers):
    if not papers: return None
    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in papers])
    prompt = f"あなたは技術トレンドに敏感なAIエンジニアです。以下の論文から面白いものを3つ厳選し日本語で出力してください：\n\n{context}"
    response = model.generate_content(prompt)
    return response.text

def main():
    papers, history = get_papers_from_sources()
    report = select_top_3(papers)
    if report:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🚀 **本日の厳選論文**\n\n{report}"})
        now_str = datetime.now(timezone.utc).isoformat()
        for p in papers:
            history[p['id']] = now_str
    save_history(history)

if __name__ == "__main__":
    main()
