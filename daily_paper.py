import os
import sys
from datetime import datetime, timedelta, timezone
import arxiv
from google import genai
import requests

# --- 設定 ---
API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "history.txt"
RETENTION_DAYS = 10.5 # 1週間半

# 起動チェック
if not API_KEY or not DISCORD_URL:
    print("Error: Secretsが設定されていません。")
    sys.exit(1)

# 2026年最新SDKクライアント
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

def main():
    history = manage_history()
    
    # 検索（M1の研究にも役立つAI・機械学習系）
    query = 'cat:cs.AI OR cat:cs.LG OR abs:"transformer" OR abs:"agent"'
    search = arxiv.Search(query=query, max_results=40, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    new_papers = []
    for result in arxiv_client.results(search):
        pid = result.entry_id.split('/')[-1]
        if pid not in history:
            new_papers.append({"id": pid, "title": result.title, "summary": result.summary, "url": result.pdf_url})
    
    if not new_papers:
        print("新規論文なし")
        return

    # プロンプト（最新論文を目利きさせる）
    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"あなたは技術トレンドに敏感なAIエンジニアです。以下の論文から特に面白い4本を選び日本語で要約して：\n\n{context}"
    
    # 2026年標準モデルを使用
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    
    # Discord通知
    requests.post(DISCORD_URL, json={"content": f"🚀 **本日の厳選論文**\n\n{response.text}"})

    # 履歴更新（1.5週間分）
    now_str = datetime.now(timezone.utc).isoformat()
    for p in new_papers:
        history[p['id']] = now_str
    
    with open(HISTORY_FILE, "w") as f:
        for pid, ts in history.items():
            f.write(f"{pid}|{ts}\n")

if __name__ == "__main__":
    main()
