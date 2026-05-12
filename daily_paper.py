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

def get_papers():
    """arXiv (前日のAI関連) -> ダメなら Hugging Face (AIトレンド)"""
    # 前日の日付（UTC）を取得
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    
    # 主要なAIカテゴリーに限定
    # cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CV (Computer Vision), cs.CL (NLP)
    query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL'
    search = arxiv.Search(query=query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    try:
        print(f"Fetching papers published on {yesterday} from arXiv...")
        results = list(arxiv_client.results(search))
        
        # 前日に公開されたものだけを抽出
        filtered_papers = []
        for r in results:
            if r.published.date() == yesterday:
                filtered_papers.append({
                    "id": r.entry_id.split('/')[-1],
                    "title": r.title,
                    "summary": r.summary,
                    "url": r.pdf_url
                })
        
        # 該当が少ない場合は少し範囲を広げる（空だと困るため）
        if len(filtered_papers) < 4:
            print("Few papers found for yesterday, taking latest AI papers.")
            filtered_papers = [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url} for r in results[:10]]
            
        return filtered_papers, "arXiv"
        
    except Exception as e:
        print(f"arXiv Access Error: {e}. Switching to HF (AI Trends)...")
        # Hugging Face Daily Papersはほぼ100% AI/ML関連です
        url = "https://huggingface.co/api/daily_papers"
        res = requests.get(url, timeout=10)
        data = res.json()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf"} for i in data], "Hugging Face"

def call_gemini(prompt):
    for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
        try:
            print(f"Requesting to {model_name}...")
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except Exception as e:
            print(f"Gemini error ({model_name}): {e}")
            time.sleep(30)
    return None

def main():
    history = manage_history()
    papers_to_check, source_name = get_papers()

    # 未読の論文のみ抽出
    new_papers = [p for p in papers_to_check if p['id'] not in history]
    
    if not new_papers:
        print("新規のAI論文はありませんでした。")
        return

    context = "\n\n".join([f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}" for p in new_papers])
    prompt = f"あなたはAI専門のリサーチャーです。以下の{source_name}の最新AI論文から特に注目すべき4本を選び、日本語で要約して：\n\n{context}"
    
    report = call_gemini(prompt)
    
    if report:
        try:
            prefix = "📰 **昨日の厳選AI論文**" if source_name == "arXiv" else "⚠️ **HFより抽出された最新AIトレンド**"
            res = requests.post(DISCORD_URL, json={"content": f"{prefix}\n\n{report}"}, timeout=10)
            res.raise_for_status()
            
            # 成功時のみ履歴保存
            now_str = datetime.now(timezone.utc).isoformat()
            for p in new_papers:
                history[p['id']] = now_str
            with open(HISTORY_FILE, "w") as f:
                for pid, ts in history.items():
                    f.write(f"{pid}|{ts}\n")
            print("Successfully posted and updated history.")
        except Exception as e:
            print(f"Discord Post Error: {e}")
    else:
        print("Failed to generate report.")

if __name__ == "__main__":
    main()
