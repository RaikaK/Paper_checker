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
RETENTION_DAYS = 10.5

if not API_KEY or not DISCORD_URL:
    print("Error: APIキーまたはDiscord URLが設定されていません。")
    sys.exit(1)

# 最新のSDKクライアント (2026年仕様)
client = genai.Client(api_key=API_KEY)
# arXivクライアントの作成
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
                    except ValueError:
                        continue
    return valid_history

def main():
    history = manage_history()
    
    # 検索条件
    query = 'cat:cs.AI OR cat:cs.LG OR abs:"transformer" OR abs:"llm" OR abs:"agent"'
    search = arxiv.Search(
        query=query,
        max_results=40,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    new_papers = []
    # arxiv_client.results(search) を使うのが最新の書き方です
    for result in arxiv_client.results(search):
        pid = result.entry_id.split('/')[-1]
        if pid not in history:
            new_papers.append({
                "id": pid,
                "title": result.title,
                "summary": result.summary,
                "url": result.pdf_url
            })
    
    if not new_papers:
        print("新規論文はありませんでした。")
        return

    # AIへのプロンプト
    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"""
    あなたは技術トレンドに敏感なAIエンジニアです。
    以下の論文リストから、特に興味深く、SNS等で話題になりそうな4本を選び、日本語で解説してください。
    
    【タイトル】（日本語訳）
    【注目理由】（1行）
    【概要】（専門用語を交えつつ3行で）
    【URL】
    ------------------
    {context}
    """
    
    # 最新モデル gemini-2.0-flash を使用
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    report = response.text

    # Discord通知
    requests.post(DISCORD_URL, json={"content": f"🚀 **本日の厳選論文 (4本)**\n\n{report}"})

    # 履歴保存
    now_str = datetime.now(timezone.utc).isoformat()
    for p in new_papers:
        history[p['id']] = now_str
    
    with open(HISTORY_FILE, "w") as f:
        for pid, ts in history.items():
            f.write(f"{pid}|{ts}\n")

if __name__ == "__main__":
    main()
