import os
import sys
import time
from datetime import datetime, timedelta, timezone
import arxiv
from google import genai
import requests

# --- 設定 ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY_2")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
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
    """arXiv -> Hugging Face 取得ロジック"""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL'
    search = arxiv.Search(query=query, max_results=30, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        filtered = [p for p in results if p.published.date() == yesterday]
        if not filtered: filtered = results[:8]
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url} for r in filtered], "arXiv"
    except:
        print("arXiv 503/429. Using HF fallback.")
        res = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        data = res.json()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf"} for i in data], "Hugging Face"

def call_openrouter(prompt):
    """OpenRouterバックアップ (Llama 3.1 / DeepSeek)"""
    if not OPENROUTER_KEY: return None
    print("Trying OpenRouter (Backup Model)...")
    # 無料で安定しているモデルを優先
    models = ["meta-llama/llama-3.1-8b-instruct:free", "google/gemini-2.0-flash-lite-preview-02-05:free"]
    
    for model in models:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "HTTP-Referer": "https://github.com/RaikaK/Paper_checker"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            res_json = response.json()
            if response.status_code == 200 and 'choices' in res_json:
                return res_json['choices'][0]['message']['content']
            else:
                print(f"OpenRouter ({model}) Failed: {res_json.get('error', 'Unknown Error')}")
        except Exception as e:
            print(f"OpenRouter Request Error: {e}")
    return None

def call_llm(prompt):
    """メインLLMロジック"""
    # 1. Google Gemini 本家
    if client:
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                print(f"Attempting Google Gemini ({model_name})...")
                res = client.models.generate_content(model=model_name, contents=prompt)
                return res.text
            except Exception as e:
                print(f"Google Gemini ({model_name}) Error: {e}")
                if "429" in str(e):
                    print("Quota hit. Sleeping...")
                    time.sleep(30)
    
    # 2. OpenRouter バックアップ
    return call_openrouter(prompt)

def main():
    history = manage_history()
    papers_to_check, source_name = get_papers()
    new_papers = [p for p in papers_to_check if p['id'] not in history]
    
    if not new_papers:
        print("No new papers to notify.")
        return

    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"AI専門リサーチャーとして、以下の{source_name}の論文から注目の4本を選び日本語で要約して。数式がある場合はその意味も軽く触れて：\n\n{context}"
    
    report = call_llm(prompt)
    
    if report:
        prefix = "📰 **昨日の厳選AI論文**" if source_name == "arXiv" else "⚠️ **arXiv混雑中のためHF抽出**"
        # Discord送信
        try:
            res = requests.post(DISCORD_URL, json={"content": f"{prefix}\n\n{report}"}, timeout=15)
            res.raise_for_status()
            
            # 【重要】送信成功時のみ履歴を更新
            now_str = datetime.now(timezone.utc).isoformat()
            for p in new_papers: history[p['id']] = now_str
            with open(HISTORY_FILE, "w") as f:
                for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
            print("Successfully notified and updated history.")
        except Exception as e:
            print(f"Discord Post Error: {e}")
    else:
        print("Critical Error: All LLM services failed. History not updated to allow retry tomorrow.")

if __name__ == "__main__":
    main()
