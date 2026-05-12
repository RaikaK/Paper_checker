import os
import sys
import time
import json
import re
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
    if os.path.exists(HISTORY_FILE):
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

def get_hf_papers():
    """Hugging Face Daily Papers (メインソース) を取得"""
    try:
        url = "https://huggingface.co/api/daily_papers"
        res = requests.get(url, timeout=15)
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf", "source": "Hugging Face"} for i in res.json()]
    except:
        return []

def get_arxiv_papers():
    """主要学会・有力企業関連のarXiv論文を取得"""
    # 主要学会と企業のキーワードを組み込んだクエリ
    # CVPR, NeurIPS, ICLR, ICML, ACL, EMNLP, Google, Meta, OpenAI, NVIDIA, DeepMind, Microsoft
    keywords = '(CVPR OR NeurIPS OR ICLR OR ICML OR ACL OR Google OR Meta OR OpenAI OR NVIDIA OR DeepMind)'
    query = f'({keywords}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)'
    
    search = arxiv.Search(query=query, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url, "source": "arXiv (Top Tier Search)"} for r in results]
    except:
        return []

def call_llm(prompt):
    free_models = [
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b-it:free",
        "qwen/qwen3-coder:free"
    ]
    if client:
        try:
            res = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            return res.text
        except: pass
    if OPENROUTER_KEY:
        for model_id in free_models[1:]:
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                    json={"model": model_id, "messages": [{"role": "user", "content": prompt}]},
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
            except: continue
    return None

def parse_json_from_text(text):
    try:
        json_str = re.search(r'\[.*\]', text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return None

def main():
    history = manage_history()
    
    # HFとarXivの両方から取得
    hf_papers = get_hf_papers()
    arxiv_papers = get_arxiv_papers()
    
    all_papers = hf_papers + arxiv_papers
    new_papers = [p for p in all_papers if p['id'] not in history]
    
    if not new_papers:
        print("No new papers."); return

    context = json.dumps(new_papers, ensure_ascii=False)
    prompt = f"""
    あなたは最先端のAIトレンドを追う研究者です。以下の論文リストから【4本】を厳選してJSON形式で出力してください。

    【制約事項】
    1. 最低2本は source が 'Hugging Face' のものを選んでください。
    2. arXiv のものは、大手企業や主要学会に関連がありそうなものを優先してください。
    3. JSON配列形式のみを出力してください。

    [
      {{
        "title": "日本語訳タイトル",
        "url": "URL",
        "source": "提供元(Hugging Face または arXiv)",
        "summary": "3行要約(技術的内容)",
        "tags": ["分野タグ1", "タグ2"],
        "layman_point": "一般人が興味を持ちそうなポイント",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]
    データ: {context}
    """
    
    raw_report = call_llm(prompt)
    selected_papers = parse_json_from_text(raw_report)
    
    if selected_papers:
        today_str = datetime.now().strftime('%Y-%m-%d')
        # 1. リストの送信
        header_msg = f"📅 **{today_str} 厳選AIニュース**\n\n"
        for i, p in enumerate(selected_papers, 1):
            header_msg += f"{i}. {p['title']} ({p['source']})\n"
        requests.post(DISCORD_URL, json={"content": header_msg})
        time.sleep(1)

        # 2. 詳細の送信
        for p in selected_papers:
            msg = (
                f"📄 **{p['title']}**\n"
                f"🔗 URL: {p['url']}\n"
                f"🏢 Source: {p['source']}\n"
                f"📝 要約: {p['summary']}\n"
                f"🏷️ タグ: {', '.join(p['tags'])}\n"
                f"💡 ポイント: {p['layman_point']}\n"
                f"⭐ 興味深さ: {p['interest_score']}/10 | 🛠️ 汎用性: {p['applicability_score']}/10\n"
                f"--------------------------------------------"
            )
            requests.post(DISCORD_URL, json={"content": msg})
            time.sleep(1)
        
        # 履歴更新
        now_str = datetime.now(timezone.utc).isoformat()
        for p in new_papers: history[p['id']] = now_str
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
    else:
        print("Failed to process LLM output.")

if __name__ == "__main__":
    main()
