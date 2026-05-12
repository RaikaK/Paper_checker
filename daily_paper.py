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
    """arXiv (前日のAI関連) -> ダメなら HF"""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    # カテゴリをAI/機械学習に厳選
    query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL'
    search = arxiv.Search(query=query, max_results=30, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        filtered = [p for p in results if p.published.date() == yesterday]
        if not filtered: filtered = results[:6]
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url} for r in filtered], "arXiv"
    except:
        res = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf"} for i in res.json()], "Hugging Face"

def call_openrouter(prompt):
    """ユーザー提供の最新モデルIDリストによるバックアップ"""
    if not OPENROUTER_KEY: return None
    print("Starting OpenRouter sequence with verified IDs...")
    
    # 提示されたリンクと名称に基づく正確なモデルIDリスト (2026年版)
    free_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b:free",
        "qwen/qwen3-coder-480b-a35b:free",
        "openai/gpt-oss-120b:free",
        "mistralai/pixtral-12b:free" # 予備
    ]
    
    for model_id in free_models:
        try:
            print(f"Trying: {model_id}")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "HTTP-Referer": "https://github.com/RaikaK/Paper_checker",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=45
            )
            res_json = response.json()
            if response.status_code == 200 and 'choices' in res_json:
                print(f"Success! Model used: {model_id}")
                return res_json['choices'][0]['message']['content']
            else:
                msg = res_json.get('error', {}).get('message', 'Unknown Error')
                print(f"Failed {model_id}: {msg}")
        except Exception as e:
            print(f"Request error: {e}")
    return None

def call_llm(prompt):
    # 1. Google Gemini (本家)
    if client:
        # モデル名の指定をより標準的な形式に変更
        for m in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                print(f"Attempting Google {m}...")
                res = client.models.generate_content(model=m, contents=prompt)
                return res.text
            except Exception as e:
                print(f"Google {m} rejected: {e}")
    
    # 2. OpenRouter (提供された最新無料モデル群)
    return call_openrouter(prompt)

def main():
    history = manage_history()
    papers_to_check, source_name = get_papers()
    new_papers = [p for p in papers_to_check if p['id'] not in history]
    
    if not new_papers:
        print("No new papers found since yesterday.")
        return

    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"AIリサーチャーとして、以下の{source_name}の最新AI論文から注目の4本を選び、日本語で要約して。重要な数式があればその意味も添えて：\n\n{context}"
    
    report = call_llm(prompt)
    
    if report:
        try:
            prefix = "📰 **昨日の厳選AI論文**" if source_name == "arXiv" else "⚠️ **HF抽出AIトレンド**"
            requests.post(DISCORD_URL, json={"content": f"{prefix}\n\n{report}"}, timeout=15).raise_for_status()
            
            # 成功時のみ履歴保存
            now_str = datetime.now(timezone.utc).isoformat()
            for p in new_papers: history[p['id']] = now_str
            with open(HISTORY_FILE, "w") as f:
                for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
            print("Successfully completed.")
        except Exception as e:
            print(f"Discord notify error: {e}")
    else:
        print("Fatal: All LLM routes failed. Check OpenRouter Key or Model IDs.")

if __name__ == "__main__":
    main()
