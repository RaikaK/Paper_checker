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
    try:
        url = "https://huggingface.co/api/daily_papers"
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf", "source": "Hugging Face"} for i in res.json()]
    except: return []

def get_arxiv_papers():
    keywords = '(CVPR OR NeurIPS OR ICLR OR ICML OR ACL OR Google OR Meta OR OpenAI OR NVIDIA OR DeepMind OR Microsoft)'
    query = f'({keywords}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)'
    search = arxiv.Search(query=query, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url, "source": "arXiv (Top Tier Search)"} for r in results]
    except: return []

def call_llm(prompt):
    models = ["google/gemini-2.0-flash-001", "meta-llama/llama-3.3-70b-instruct:free", "google/gemma-4-31b-it:free", "qwen/qwen3-coder:free"]
    if client:
        try:
            res = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            return res.text
        except: pass
    if OPENROUTER_KEY:
        for m_id in models[1:]:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                    json={"model": m_id, "messages": [{"role": "user", "content": prompt}]},
                    timeout=60
                )
                if resp.status_code == 200: return resp.json()['choices'][0]['message']['content']
            except: continue
    return None

def parse_json_from_text(text):
    if not text: return None
    try:
        # 文字列のクリーンアップ（SyntaxErrorの原因になりやすい改行等を除去）
        cleaned = text.strip()
        # JSON配列の開始 [ と終了 ] を探す
        start = cleaned.find('[')
        end = cleaned.rfind(']') + 1
        if start != -1 and end != 0:
            json_str = cleaned[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Error: {e}")
    return None

def main():
    history = manage_history()
    hf = get_hf_papers()
    ar = get_arxiv_papers()
    new_papers = [p for p in (hf + ar) if p['id'] not in history]
    
    if not new_papers:
        print("No new papers."); return

    prompt = f"""
    最先端AIリサーチの専門家として、以下の論文リストから【4本】厳選し、指定のJSON形式で出力してください。
    【条件】
    1. sourceが'Hugging Face'のものを最低2本含める。
    2. arXivは大手企業や主要学会のものを優先。
    3. 解説不要、純粋なJSON配列のみ。

    [
      {{
        "title": "日本語訳タイトル",
        "url": "URL",
        "source": "提供元",
        "summary": "3行要約(技術詳細)",
        "tags": ["タグ1", "タグ2"],
        "layman_point": "一般向け注目点",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]
    データ: {json.dumps(new_papers, ensure_ascii=False)}
    """
    
    report_text = call_llm(prompt)
    selected = parse_json_from_text(report_text)
    
    if selected:
        today = datetime.now().strftime('%Y-%m-%d')
        # 1. リスト送信
        header = f"📅 **{today} 厳選AIニュース**\n"
        for i, p in enumerate(selected, 1):
            header += f"{i}. {p['title']} ({p['source']})\n"
        requests.post(DISCORD_URL, json={"content": header})
        time.sleep(1)

        # 2. 詳細送信
        for p in selected:
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
        
        # 履歴保存
        now_s = datetime.now(timezone.utc).isoformat()
        for p in new_papers: history[p['id']] = now_s
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
    else:
        print("Failed to parse LLM output.")
        if report_text: print(f"Raw response: {report_text[:300]}")

if __name__ == "__main__":
    main()
