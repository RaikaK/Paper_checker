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
import argparse

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

def get_hf_papers(target_date_str=None):
    """
    指定された日付（なければ今日）のHugging Face Daily Papersを取得し、
    Upvotes（いいね数）が多い順にソートして返す。
    """
    try:
        url = "https://huggingface.co/api/daily_papers"
        if target_date_str:
            url += f"?date={target_date_str}"
            print(f"Fetching from Hugging Face Daily for {target_date_str}...")
        else:
            print("Fetching from Hugging Face Daily (Today)...")
            
        res = requests.get(url, timeout=15)
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
                "comment": "",
                "upvotes": p.get('upvotes', 0) # Upvote数を確保
            })
            
        # Upvotesの降順（多い順）でソート
        papers.sort(key=lambda x: x['upvotes'], reverse=True)
        return papers
    except Exception as e:
        print(f"HF Fetch Error: {e}")
        return []

def get_hf_ranking_papers(history):
    try:
        print("Fetching ranking papers from Hugging Face...")
        res = requests.get("https://huggingface.co/api/papers?sort=upvotes&limit=100", timeout=15)
        res.raise_for_status()
        
        now = datetime.now(timezone.utc)
        weekly_papers = []
        monthly_papers = []
        
        for item in res.json():
            pid = item.get('id')
            if not pid: continue
            
            pub_str = item.get('publishedAt') or item.get('date')
            if not pub_str: continue
            
            try:
                date_part = pub_str[:10]
                pub_date = datetime.strptime(date_part, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except: continue
            
            days_ago = (now - pub_date).days
            
            p_data = {
                "id": pid,
                "title": item.get('title', ''),
                "summary": item.get('summary', ''),
                "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/{pid}",
                "hf_url": f"https://huggingface.co/papers/{pid}",
                "source": "Hugging Face (Ranking)",
                "published_date": date_part,
                "github_url": "",
                "journal_ref": "",
                "comment": "",
                "upvotes": item.get('upvotes', 0)
            }
            
            if days_ago <= 7:
                weekly_papers.append(p_data)
            if days_ago <= 30:
                monthly_papers.append(p_data)
        
        weekly_papers.sort(key=lambda x: x['upvotes'], reverse=True)
        monthly_papers.sort(key=lambda x: x['upvotes'], reverse=True)
        
        selected_backup = []
        
        for p in weekly_papers[:10]:
            if p['id'] not in history:
                selected_backup.append(p)
                if len(selected_backup) >= 2: break
                
        if len(selected_backup) < 2:
            for p in monthly_papers[:15]:
                if p['id'] not in history and p['id'] not in [x['id'] for x in selected_backup]:
                    selected_backup.append(p)
                    if len(selected_backup) >= 2: break
                    
        return selected_backup
    except Exception as e:
        print(f"HF Ranking Fetch Error: {e}")
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

def main(target_date=None):
    history = manage_history()
    
    # ターゲット日付が指定されていればHFに渡し、なければNone（今日）
    hf = get_hf_papers(target_date)
    ar = get_arxiv_papers()
    
    all_papers = hf + ar
    new_papers = [p for p in all_papers if p['id'] not in history]
    
    is_fallback = False
    if not new_papers:
        print("No new daily papers found. Checking Hugging Face rankings as fallback...")
        new_papers = get_hf_ranking_papers(history)
        if new_papers:
            is_fallback = True
            print(f"Found {len(new_papers)} ranking papers for fallback.")
        else:
            today = target_date if target_date else datetime.now().strftime('%Y-%m-%d')
            no_paper_msg = f"📅 **{today} 厳選AI論文リスト**\n本日新しく公表された論文、および過去のランキング（週間10位/月間15位以内）に対象となる未読論文はありませんでした。"
            requests.post(DISCORD_URL, json={"content": no_paper_msg})
            print("No new papers and no ranking papers found. Notification sent to Discord.")
            return

    print(f"Processing {len(new_papers)} papers...")

    if is_fallback:
        selection_rule = f"""リストにある【すべて（{len(new_papers)}本）】を必ず解析し、指定のJSON配列形式でのみ出力してください。
    これは新着がない場合の過去のランキングからの補填処理です。選出数の変更（絞り込みや追加）はせず、与えられたデータをそのまま和訳・要約してください。"""
    else:
        selection_rule = """原則として【4本】を選出しますが、データ内に「Survey論文（サーベイ、レビュー、包括的な解説論文）」が含まれている場合は、それらを最優先で追加し【最大5本】まで選出枠を広げてください。
    
    【厳守ルール】
    1. タイトルや要約に 'survey', 'review', 'overview', 'comprehensive study' などの文言が含まれる「Survey論文」があれば、通常の選定（4本）に加えて1枠追加し、合計最大5本として必ず出力に含めてください。Survey論文がない場合は、通常通り4本にしてください。
    2. sourceが'Hugging Face'のものを【必ず2本以上】選んでください。なお、データはHugging FaceのUpvotes（いいね数）が多い順に並んでいます。
    3. arXivは大手企業や有名学会のものを優先してください。"""

    prompt = f"""
    あなたはAIリサーチの専門家です。以下の論文リストから厳選し、指定のJSON配列形式でのみ出力してください。
    
    {selection_rule}
    4. JSON以外の説明、挨拶、コードブロック(```json等)は一切不要です。
    5. 出力テキストは指定がない限り【日本語】で行ってください。

    【査読判定のヒント】
    データ内の 'journal_ref' や 'comment' に学会名や「Accepted to ○○」といった記述がある場合は「査読あり（学会名）」、特に記載がなくプレプリント状態の場合は「Preprint（査読未定）」と判定してください。

    [
      {{
        "id": "データ内の id をそのまま正確に転記してください。ここが空欄になるとシステムが壊れます",
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
        today = target_date if target_date else datetime.now().strftime('%Y-%m-%d')
        header = f"📅 **{today} 厳選AI論文リスト**\n"
        for i, p in enumerate(selected, 1):
            header += f"{i}. {p.get('title_ja', '無題')} / {p.get('title_en', 'Untitled')} ({p.get('source', 'Unknown')})\n"
        requests.post(DISCORD_URL, json={"content": header})
        
        time.sleep(1)

        for p in selected:
            git_info = f"💻 GitHub: {p['github_url']}\n" if p.get('github_url') else ""
            hf_info = f"🤗 Hugging Face: {p['hf_url']}\n" if p.get('hf_url') else ""
            
            msg = (
                f"📄 **{p.get('title_ja', '無題')}**\n"
                f"🔤 原題: {p.get('title_en', 'Untitled')}\n"
                f"📅 公開日: {p.get('published_date', '不明')} | 🛡️ 査読: {p.get('peer_review_status', 'Preprint（査読未定）')}\n"
                f"🔗 arXiv Abs: {p.get('arxiv_url', '不明')}\n"
                f"📕 PDF: {p.get('pdf_url', '不明')}\n"
                f"{hf_info}"
                f"{git_info}"
                f"🏢 Source: {p.get('source', 'Unknown')}\n"
                f"📝 要約: {p.get('summary', '要約なし短評')}\n"
                f"🏷️ タグ: {', '.join(p.get('tags', []))}\n"
                f"💡 ポイント: {p.get('layman_point', 'なし')}\n"
                f"⭐ 興味深さ: {p.get('interest_score', 0)}/10 | 🛠️ 汎用性: {p.get('applicability_score', 0)}/10\n"
                f"--------------------------------------------"
            )
            requests.post(DISCORD_URL, json={"content": msg})
            time.sleep(1)
        
        now_s = datetime.now(timezone.utc).isoformat()
        for p in selected:
            if p.get('id'):
                history[p['id']] = now_s
                
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): f.write(f"{pid}|{ts}\n")
        print("All processes finished successfully.")
    else:
        parse_err = "⚠️ **構文エラー**: LLMから応答はありましたが、JSON形式へのパースに失敗しました。出力を確認してください。"
        print(parse_err)
        requests.post(DISCORD_URL, json={"content": parse_err})

if __name__ == "__main__":
    try:
        # コマンドライン引数から日付（YYYY-MM-DD）を取得。なければNone
        parser = argparse.ArgumentParser(description="Daily Paper Checker")
        parser.add_argument("date", nargs="?", default=None, help="Target date in YYYY-MM-DD format (optional)")
        args = parser.parse_args()
        
        # もし日付のフォーマットチェックをしたい場合はここで validate しても良いです
        if args.date and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
            print("Error: Date format must be YYYY-MM-DD")
            sys.exit(1)

        main(target_date=args.date)
        
    except Exception as e:
        error_reason = "".join(traceback.format_exception_only(type(e), e)).strip()
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
