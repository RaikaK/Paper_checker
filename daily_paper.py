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
    print("Error: DISCORD_WEBHOOK_URL is missing.")
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
        print("Fetching from Hugging Face...")
        res = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        res.raise_for_status()
        return [{"id": i['paper']['id'], "title": i['paper']['title'], "summary": i['paper'].get('summary', ''), "url": f"https://arxiv.org/pdf/{i['paper']['id']}.pdf", "source": "Hugging Face"} for i in res.json()]
    except Exception as e:
        print(f"HF Fetch Error: {e}")
        return []

def get_arxiv_papers():
    print("Fetching from arXiv...")
    keywords = '(CVPR OR NeurIPS OR ICLR OR ICML OR ACL OR Google OR Meta OR OpenAI OR NVIDIA OR DeepMind OR Microsoft)'
    query = f'({keywords}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)'
    search = arxiv.Search(query=query, max_results=20, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        return [{"id": r.entry_id.split('/')[-1], "title": r.title, "summary": r.summary, "url": r.pdf_url, "source": "arXiv (Top Tier)"} for r in results]
    except Exception as e:
        print(f"arXiv Fetch Error: {e}")
        return []

def call_llm(prompt):
    # 2026年最新の無料モデルリスト
    models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b-it:free",
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-120b:free",
        "deepseek/deepseek-r1:free"
    ]
    
    # 1. Google Gemini (本家)
    if client:
        for m_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                print(f"Trying Google Gemini ({m_name})...")
                res = client.models.generate_content(model=m_name, contents=prompt)
                if res.text: return res.text
            except Exception as e:
                print(f"Gemini {m_name} failed: {e}")

    # 2. OpenRouter (バックアップ)
    if OPENROUTER_KEY:
        for m_id in models:
            try:
                print(f"Trying OpenRouter Model: {m_id}")
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "HTTP-Referer": "https://github.com/RaikaK/Paper_checker"},
                    json={"model": m_id, "messages": [{"role": "user", "content": prompt}]},
                    timeout=60
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if 'choices' in result:
                        content = result['choices'][0]['message']['content']
                        if content: return content
                else:
                    print(f"OpenRouter {m_id} error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"OpenRouter {m_id} request failed: {e}")
                continue
    return None

def parse_json_from_text(text):
    if not text: return None
    try:
        # JSONの開始と終了を探す
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw text that failed to parse: \n{text}")
    return None

def main():
    history = manage_history()
    hf = get_hf_papers()
    ar = get_arxiv_papers()
    
    # 未知の論文のみ
    all_papers = hf + ar
    new_papers = [p for p in all_papers if p['id'] not in history]
    
    if not new_papers:
        print("No new papers found since last run."); return

    print(f"Processing {len(new_papers)} new papers...")

    prompt = f"""
    あなたはAIリサーチの専門家です。以下の論文リストから【4本】を厳選し、指定のJSON配列形式でのみ出力してください。
    
    【厳守ルール】
    1. sourceが'Hugging Face'のものを【必ず2本以上】選んでください。
    2. arXivは大手企業や有名学会のものを優先してください。
    3. JSON以外の説明、挨拶、コードブロック(```json等)は一切不要です。
    4. 日本語で出力してください。

    [
      {{
        "title": "タイトル(和訳)",
        "url": "URL",
        "source": "提供元",
        "summary": "3行程度の技術要約",
        "tags": ["タグ1", "タグ2"],
        "layman_point": "一般人が興味を持ちそうな点",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]

    データ: {json.dumps(new_papers, ensure_ascii=False)}
    """
    
    report_text = call_llm(prompt)
    if not report_text:
        print("CRITICAL: All LLM models failed to return any response.")
        return

    selected = parse_json_from_text(report_text)
    
    if selected:
        today = datetime.now().strftime('%Y-%m-%d')
        # リスト送信
        header = f"📅 **{today} 厳選AI論文リスト**\n"
        for i, p in enumerate(selected, 1):
            header += f"{i}. {p['title']} ({p['source']})\n"
        requests.post(DISCORD_URL, json={"content": header})
        
        time.sleep(1)

        # 詳細送信
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
        print("All processes finished successfully.")
    else:
        print("Failed to parse the response as JSON.")

if __name__ == "__main__":
    main()
