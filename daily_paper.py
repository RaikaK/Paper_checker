import os
import sys
import time
import json
import re
import traceback
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

def extract_github_url(text):
    """テキスト内からGitHubのURLを探して返す"""
    if not text:
        return ""
    match = re.search(r"https?://github\.com/[^\s,\]\)]+", text)
    return match.group(0) if match else ""

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
        papers = []
        for i in res.json():
            p = i['paper']
            pub_date = p.get('publishedAt', datetime.now().isoformat())[:10]
            papers.append({
                "id": p['id'],
                "title": p['title'],
                "summary": p.get('summary', ''),
                "pdf_url": f"https://arxiv.org/pdf/{p['id']}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/{p['id']}",
                "hf_url": f"https://huggingface.co/papers/{p['id']}",
                "source": "Hugging Face",
                "published_date": pub_date,
                "github_url": "",
                "journal_ref": "",
                "comment": ""
            })
        return papers
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
        papers = []
        for r in results:
            pid = r.entry_id.split('/')[-1]
            git_url = extract_github_url(r.comment) or extract_github_url(r.summary)
            papers.append({
                "id": pid,
                "title": r.title,
                "summary": r.summary,
                "pdf_url": r.pdf_url,
                "arxiv_url": f"https://arxiv.org/abs/{pid}",
                "hf_url": "",
                "source": "arXiv (Top Tier)",
                "published_date": r.published.strftime('%Y-%m-%d'),
                "github_url": git_url,
                "journal_ref": r.journal_ref or "",
                "comment": r.comment or ""
            })
        return papers
    except Exception as e:
        print(f"arXiv Fetch Error: {e}")
        return []

def call_llm(prompt):
    models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b-it:free",
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-120b:free",
        "deepseek/deepseek-r1:free"
    ]
    
    if client:
        for m_name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
            try:
                print(f"Trying Google Gemini ({m_name})...")
                res = client.models.generate_content(model=m_name, contents=prompt)
                if res.text: return res.text
            except Exception as e:
                print(f"Gemini {m_name} failed: {e}")

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
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON Parse Error: {e}")
    return None

def main():
    history = manage_history()
    hf = get_hf_papers()
    ar = get_arxiv_papers()
    
    all_papers = hf + ar
    new_papers = [p for p in all_papers if p['id'] not in history]
    
    if not new_papers:
        today = datetime.now().strftime('%Y-%m-%d')
        no_paper_msg = f"📅 **{today} 厳選AI論文リスト**\n本日新しく公表された（未読の）論文はありませんでした。"
        requests.post(DISCORD_URL, json={"content": no_paper_msg})
        print("No new papers found since last run. Notification sent to Discord.")
        return

    print(f"Processing {len(new_papers)} new papers...")

    prompt = f"""
    あなたはAIリサーチの専門家です。以下の論文リストから厳選し、指定のJSON配列形式でのみ出力してください。
    原則として【4本】を選出しますが、データ内に「Survey論文（サーベイ、レビュー、包括的な解説論文）」が含まれている場合は、それらを最優先で追加し【最大5本】まで選出枠を広げてください。
    
    【厳守ルール】
    1. タイトルや要約に 'survey', 'review', 'overview', 'comprehensive study' などの文言が含まれる「Survey論文」があれば、通常の選定（4本）に加えて1枠追加し、合計最大5本として必ず出力に含めてください。Survey論文がない場合は、通常通り4本にしてください。
    2. sourceが'Hugging Face'のものを【必ず2本以上】選んでください。
    3. arXivは大手企業や有名学会のものを優先してください。
    4. JSON以外の説明、挨拶、コードブロック(```json等)は一切不要です。
    5. 出力テキストは指定がない限り【日本語】で行ってください。

    【査読判定のヒント】
    データ内の 'journal_ref' や 'comment' に学会名や「Accepted to ○○」といった記述がある場合は「査読あり（学会名）」、特に記載がなくプレプリント状態の場合は「Preprint（査読未定）」と判定してください。

    [
      {{
        "title_ja": "タイトル(日本語和訳)",
        "title_en": "元の英語タイトル",
        "pdf_url": "論文PDFのURL",
        "arxiv_url": "arXivの概要ページのURL",
        "hf_url": "Hugging Faceの論文ページのURL (データ内にあればそのまま、なければ空文字)",
        "github_url": "ソースコードのURL(あれば優先、なければ空文字)",
        "published_date": "公開日(YYYY-MM-DD)",
        "peer_review_status": "査読ステータス（例：『査読あり（CVPR 2026）』または『Preprint（査読未定）』）",
        "source": "提供元",
        "summary": "技術的な要約(日本語で3行程度)",
        "tags": ["タグ1", "タグ2"],
        "layman_point": "専門外の人でも凄さがわかるポイント(日本語)",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]

    データ: {json.dumps(new_papers, ensure_ascii=False)}
    """
    
    report_text = call_llm(prompt)
    if not report_text:
        err_msg = "⚠️ **致命的エラー**: すべてのLLMモデルからの応答取得に失敗したため、本日の選出処理を中断しました。"
        print(err_msg)
        requests.post(DISCORD_URL, json={"content": err_msg})
        return

    selected = parse_json_from_text(report_text)
    
    if selected:
        today = datetime.now().strftime('%Y-%m-%d')
        header = f"📅 **{today} 厳選AI論文リスト**\n"
        for i, p in enumerate(selected, 1):
            header += f"{i}. {p['title_ja']} / {p['title_en']} ({p['source']})\n"
        requests.post(DISCORD_URL, json={"content": header})
        
        time.sleep(1)

        for p in selected:
            git_info = f"💻 GitHub: {p['github_url']}\n" if p.get('github_url') else ""
            hf_info = f"🤗 Hugging Face: {p['hf_url']}\n" if p.get('hf_url') else ""
            
            msg = (
                f"📄 **{p['title_ja']}**\n"
                f"🔤 原題: {p['title_en']}\n"
                f"📅 公開日: {p.get('published_date', '不明')} | 🛡️ 査読: {p.get('peer_review_status', 'Preprint（査読未定）')}\n"
                f"🔗 arXiv Abs: {p.get('arxiv_url', '不明')}\n"
                f"📕 PDF: {p.get('pdf_url', '不明')}\n"
                f"{hf_info}"
                f"{git_info}"
                f"🏢 Source: {p['source']}\n"
                f"📝 要約: {p['summary']}\n"
                f"🏷️ タグ: {', '.join(p['tags'])}\n"
                f"💡 ポイント: {p['layman_point']}\n"
                f"⭐ 興味深さ: {p['interest_score']}/10 | 🛠️ 汎用性: {p['applicability_score']}/10\n"
                f"--------------------------------------------"
            )
            requests.post(DISCORD_URL, json={"content": msg})
            time.sleep(1)
        
        now_s = datetime.now(timezone.utc).isoformat()
        for p in new_papers: history[p['id']] = now_s
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
        print("All processes finished successfully.")
    else:
        parse_err = "⚠️ **構文エラー**: LLMから応答はありましたが、JSON形式へのパースに失敗しました。出力を確認してください。"
        print(parse_err)
        requests.post(DISCORD_URL, json={"content": parse_err})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # エラーの最終行（例外の型と原因メッセージ）だけを抽出
        error_reason = "".join(traceback.format_exception_only(type(e), e)).strip()
        
        # 特殊文字やクォーテーションの連続によるパースエラーを完全に防ぐ安全な結合
        msg_parts = [
            "🚨 **プログラム実行エラーが発生しました**",
            "```",
            error_reason,
            "```"
        ]
        error_msg = "\n".join(msg_parts)
        print(error_msg)
        
        if DISCORD_URL:
            try:
                requests.post(DISCORD_URL, json={"content": error_msg})
            except Exception as discord_err:
                print(f"Failed to send error notification to Discord: {discord_err}")
