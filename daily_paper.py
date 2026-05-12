import os
import sys
import time
from datetime import datetime, timedelta, timezone
import arxiv
from google import genai
import requests

# --- 設定 ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY_2")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") # 新設
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "history.txt"
RETENTION_DAYS = 10.5

if not DISCORD_URL:
    print("Error: Discord URLが設定されていません。")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
arxiv_client = arxiv.Client()

def manage_history():
    now = datetime.now(timezone.utc)
    valid_history = {}
    if not os.path.exists(HISTORY_FILE): return valid_history
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    pid, ts_str = parts
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if now - ts < timedelta(days=RETENTION_DAYS):
                            valid_history[pid] = ts_str
                    except: continue
    return valid_history

def get_papers():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL'
    search = arxiv.Search(query=query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        filtered = [p for p in results if p.published.date() == yesterday]
        if len(filtered) < 4:
            filtered = results[:10]
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url} for r in filtered], "arXiv"
    except:
        res = requests.get("https://huggingface.co/api/daily_papers", timeout=10)
        data = res.json()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf"} for i in data], "Hugging Face"

def call_openrouter(prompt):
    """OpenRouter経由で無料モデルを呼び出す"""
    if not OPENROUTER_KEY: return None
    print("Trying OpenRouter (DeepSeek/Llama)...")
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={
                "model": "google/gemini-2.0-flash-lite-preview-02-05:free", # 無料モデルを指定
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return None

def call_llm(prompt):
    # 1. まずGemini本家を試す
    if client:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                print(f"Calling Google Gemini ({model_name})...")
                res = client.models.generate_content(model=model_name, contents=prompt)
                return res.text
            except Exception as e:
                print(f"Google Gemini Error: {e}")
                time.sleep(10)
    
    # 2. 本家がダメならOpenRouterを試す
    return call_openrouter(prompt)

def main():
    history = manage_history()
    papers_to_check, source_name = get_papers()
    new_papers = [p for p in papers_to_check if p['id'] not in history]
    
    if not new_papers:
        print("No new papers.")
        return

    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"AI専門リサーチャーとして、以下の{source_name}の論文から注目の4本を選び日本語で要約して：\n\n{context}"
    
    report = call_llm(prompt)
    
    if report:
        prefix = "📰 **昨日の厳選AI論文**" if source_name == "arXiv" else "⚠️ **HF抽出AIトレンド**"
        requests.post(DISCORD_URL, json={"content": f"{prefix}\n\n{report}"})
        now_str = datetime.now(timezone.utc).isoformat()
        for p in new_papers: history[p['id']] = now_str
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
    else:
        print("All LLM services failed.")

if __name__ == "__main__":
    main()
