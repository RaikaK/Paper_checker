import os
import sys
import time
from datetime import datetime, timedelta, timezone
import arxiv
from google import genai
import requests

# --- 設定 ---
API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "history.txt"
RETENTION_DAYS = 10.5

if not API_KEY or not DISCORD_URL:
    print("Error: Secretsが設定されていません。")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
arxiv_client = arxiv.Client()

def manage_history():
    now = datetime.now(timezone.utc)
    valid_history = {}
    if not os.path.exists(HISTORY_FILE):
        open(HISTORY_FILE, 'w').close()
        return valid_history
    
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

def get_arxiv_papers():
    query = 'cat:cs.AI OR cat:cs.LG OR abs:"transformer" OR abs:"agent" OR abs:"robot"'
    search = arxiv.Search(query=query, max_results=30, sort_by=arxiv.SortCriterion.SubmittedDate)
    for i in range(3):
        try:
            return list(arxiv_client.results(search))
        except Exception as e:
            print(f"arXiv retry {i+1}/3: {e}")
            time.sleep(15 * (i + 1))
    raise Exception("arXiv is down")

def get_hf_papers_fallback():
    url = "https://huggingface.co/api/daily_papers"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf"} for i in data]
    except:
        return []

def call_gemini_with_retry(prompt):
    """Gemini APIをモデルを切り替えながらリトライ呼び出しする"""
    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        for i in range(3): # 各モデルで3回試行
            try:
                print(f"Calling {model_name} (Attempt {i+1}/3)...")
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API error with {model_name}: {e}")
                if "429" in str(e):
                    wait_time = 60 # 429エラーなら1分待つ
                    print(f"Quota exceeded. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(5)
    raise Exception("All Gemini models failed or quota exhausted.")

def main():
    history = manage_history()
    source_name = "arXiv"
    
    try:
        results = get_arxiv_papers()
        papers_to_check = [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url} for r in results]
    except:
        source_name = "Hugging Face (Fallback)"
        papers_to_check = get_hf_papers_fallback()

    new_papers = [p for p in papers_to_check if p['id'] not in history]
    
    if not new_papers:
        print("新規論文なし")
        return

    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"あなたは世界最高のAI論文ウォッチャーです。以下のリスト（提供元: {source_name}）から特に面白い4本を選び、日本語で要約して：\n\n{context}"
    
    try:
        report = call_gemini_with_retry(prompt)
        prefix = "🚀 **本日の厳選論文**" if source_name == "arXiv" else "⚠️ **arXiv混雑中のためHFから抽出**"
        requests.post(DISCORD_URL, json={"content": f"{prefix}\n\n{report}"})

        # 履歴更新
        now_str = datetime.now(timezone.utc).isoformat()
        for p in new_papers:
            history[p['id']] = now_str
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items():
                f.write(f"{pid}|{ts}\n")
    except Exception as e:
        print(f"Final failure: {e}")

if __name__ == "__main__":
    main()
