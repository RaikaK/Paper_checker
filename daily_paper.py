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

def clean_id(raw_id):
    """arXiv IDの末尾にあるバージョン情報(v1, v2など)を削除してIDを統一する"""
    if not raw_id:
        return ""
    return re.sub(r'v\d+$', '', raw_id.strip())

def is_benchmark_paper(title, summary):
    """ベンチマークやデータセット系の論文を判定して除外する"""
    blacklist = ['benchmark', 'dataset', 'evaluation', 'leaderboard', 'benchmarks', 'datasets']
    text = f"{title} {summary}".lower()
    return any(word in text for word in blacklist)

def is_survey_paper(title, summary):
    """Survey（包括的な解説・レビュー）論文かどうかを判定する"""
    keywords = ['survey', 'review', 'overview', 'comprehensive study']
    text = f"{title} {summary}".lower()
    return any(word in text for word in keywords)

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
                                valid_history[clean_id(pid)] = ts_str
                        except: continue
    return valid_history

def get_hf_papers(target_date_str=None):
    try:
        url = "https://huggingface.co/api/daily_papers"
        if target_date_str:
            url += f"?date={target_date_str}"
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        
        papers = []
        for i in res.json():
            p = i['paper']
            pid = clean_id(p['id'])
            title = p.get('title', '')
            summary = p.get('summary', '')
            
            if is_benchmark_paper(title, summary):
                continue
                
            pub_date = p.get('publishedAt', datetime.now().isoformat())[:10]
            papers.append({
                "id": pid,
                "title": title,
                "summary": summary,
                "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
                "arxiv_url": f"https://arxiv.org/abs/{pid}",
                "hf_url": f"https://huggingface.co/papers/{pid}",
                "source": "Hugging Face",
                "published_date": pub_date,
                "github_url": "",
                "journal_ref": "",
                "comment": "",
                "upvotes": p.get('upvotes', 0)
            })
            
        papers.sort(key=lambda x: x['upvotes'], reverse=True)
        return papers
    except Exception as e:
        print(f"HF Fetch Error: {e}")
        return []

def get_hf_monthly_backup(history, count=3, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = []
    try:
        print(f"Fetching monthly ranking papers (up to {count} candidates) from Hugging Face...")
        res = requests.get("https://huggingface.co/api/papers?sort=upvotes&limit=100", timeout=15)
        res.raise_for_status()
        
        now = datetime.now(timezone.utc)
        monthly_papers = []
        
        for item in res.json():
            pid = clean_id(item.get('id'))
            if not pid or pid in history or pid in exclude_ids: continue
            
            title = item.get('title', '')
            summary = item.get('summary', '')
            if is_benchmark_paper(title, summary): continue
            
            pub_str = item.get('publishedAt') or item.get('date')
            if not pub_str: continue
            
            try:
                date_part = pub_str[:10]
                pub_date = datetime.strptime(date_part, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except: continue
            
            if (now - pub_date).days <= 30:
                monthly_papers.append({
                    "id": pid,
                    "title": title,
                    "summary": summary,
                    "pdf_url": f"https://arxiv.org/pdf/{pid}.pdf",
                    "arxiv_url": f"https://arxiv.org/abs/{pid}",
                    "hf_url": f"https://huggingface.co/papers/{pid}",
                    "source": "Hugging Face (Monthly Top)",
                    "published_date": date_part,
                    "github_url": "",
                    "journal_ref": "",
                    "comment": "",
                    "upvotes": item.get('upvotes', 0)
                })
        
        monthly_papers.sort(key=lambda x: x['upvotes'], reverse=True)
        return monthly_papers[:count]
    except Exception as e:
        print(f"HF Monthly Backup Fetch Error: {e}")
        return []

def get_arxiv_papers():
    print("Fetching from arXiv...")
    keywords = '(CVPR OR NeurIPS OR ICLR OR ICML OR ACL OR Google OR Meta OR OpenAI OR NVIDIA OR DeepMind OR Microsoft)'
    query = f'({keywords}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)'
    search = arxiv.Search(query=query, max_results=30, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        results = list(arxiv_client.results(search))
        papers = []
        for r in results:
            pid = clean_id(r.entry_id.split('/')[-1])
            if is_benchmark_paper(r.title, r.summary):
                continue
                
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
            except: continue
    return None

def parse_json_from_text(text):
    if not text: return None
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"JSON Parse Error: {e}")
    return None

def main(target_date=None):
    history = manage_history()
    
    hf_all = get_hf_papers(target_date)
    ar_all = get_arxiv_papers()
    
    hf_new = [p for p in hf_all if p['id'] not in history]
    ar_new = [p for p in ar_all if p['id'] not in history]
    
    final_papers = []
    survey_paper = None
    
    # 1. 完全新着なしモード：月間ランキングから未読3本を取得
    if not hf_new and not ar_new:
        print("No new papers found. Fetching Monthly Top papers...")
        # Surveyを探すために少し多めの15本を候補として取得
        backup_candidates = get_hf_monthly_backup(history, count=15)
        if backup_candidates:
            # 通常枠として上位3本を採択
            final_papers = backup_candidates[:3]
            # 4本目以降にSurvey論文があれば1本だけ追加枠として確保
            for p in backup_candidates[3:]:
                if is_survey_paper(p['title'], p['summary']):
                    survey_paper = p
                    break
        else:
            today = target_date if target_date else datetime.now().strftime('%Y-%m-%d')
            no_paper_msg = f"📅 **{today} 厳選AI論文リスト**\n新しい論文、および月間ランキングに対象となる未読論文はありませんでした。"
            requests.post(DISCORD_URL, json={"content": no_paper_msg})
            return
    else:
        # 2. 新着ありモード：基本HF 2本 + arXiv 2本 = 計4本
        hf_needed = 2
        ar_needed = 2
        
        if len(hf_new) < hf_needed:
            hf_needed = len(hf_new)
            ar_needed = 4 - hf_needed
        elif len(ar_new) < ar_needed:
            ar_needed = len(ar_new)
            hf_needed = 4 - ar_needed
            
        final_papers.extend(hf_new[:hf_needed])
        final_papers.extend(ar_new[:ar_needed])
        
        if len(final_papers) < 4:
            deficit = 4 - len(final_papers)
            exclude = [p['id'] for p in final_papers]
            backup = get_hf_monthly_backup(history, count=deficit, exclude_ids=exclude)
            final_papers.extend(backup)
            
        # --- Survey論文の追加枠チェック ---
        # まだ選ばれていない（final_papersに入っていない）残りの新着論文からSurveyを探す
        chosen_ids = [p['id'] for p in final_papers]
        remaining_new_papers = [p for p in (hf_new + ar_new) if p['id'] not in chosen_ids]
        
        for p in remaining_new_papers:
            if is_survey_paper(p['title'], p['summary']):
                survey_paper = p
                break

    # Survey論文が見つかっていれば、+1枠として末尾に追加
    if survey_paper:
        print(f"Found a survey paper! Adding as extra slot: {survey_paper['title']}")
        survey_paper['source'] += " (Survey枠)"
        final_papers.append(survey_paper)

    print(f"Processing {len(final_papers)} papers with LLM...")

    llm_input_data = []
    for idx, p in enumerate(final_papers):
        llm_input_data.append({
            "index": idx,
            "title": p['title'],
            "summary": p['summary'],
            "source": p['source'],
            "comment": p['comment'],
            "journal_ref": p['journal_ref']
        })

    prompt = f"""
    あなたはAIリサーチの専門家です。提示されたすべての論文について、指定のJSON配列形式でのみ翻訳・解説を出力してください。
    選別の必要はありません。提供された【すべてのインデックス】を網羅してください。JSON以外の文章は一切不要です。

    [
      {{
        "index": 0,
        "title_ja": "タイトル(日本語和訳)",
        "title_en": "元の英語タイトル",
        "peer_review_status": "査読ステータス（『査読あり（学会名）』または『Preprint（査読未定）』）",
        "summary": "技術的な要約(日本語で3行程度)",
        "tags": ["タグ1", "タグ2"],
        "layman_point": "専門外の人でも凄さがわかるポイント(日本語)",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]

    データ: {json.dumps(llm_input_data, ensure_ascii=False)}
    """
    
    report_text = call_llm(prompt)
    if not report_text:
        requests.post(DISCORD_URL, json={"content": "⚠️ **致命的エラー**: AIモデルからの応答取得に失敗しました。"})
        return

    selected_results = parse_json_from_text(report_text)
    
    if selected_results:
        today = target_date if target_date else datetime.now().strftime('%Y-%m-%d')
        
        valid_selected = []
        for item in selected_results:
            idx = item.get('index')
            if idx is not None and idx < len(final_papers):
                valid_selected.append((final_papers[idx], item))
        
        # ヘッダー送信
        header = f"📅 **{today} 厳選AI論文リスト**\n"
        for i, (orig, ai) in enumerate(valid_selected, 1):
            header += f"{i}. {ai.get('title_ja', orig['title'])} ({orig['source']})\n"
        requests.post(DISCORD_URL, json={"content": header})
        time.sleep(1)

        # 各論文の詳細送信
        now_s = datetime.now(timezone.utc).isoformat()
        for orig, ai in valid_selected:
            git_info = f"💻 GitHub: {orig['github_url']}\n" if orig['github_url'] else ""
            hf_info = f"🤗 Hugging Face: {orig['hf_url']}\n" if orig['hf_url'] else ""
            
            msg = (
                f"--------------------------------------------"
                f"📄 **{ai.get('title_ja', orig['title'])}**\n"
                f"🔤 原題: {ai.get('title_en', orig['title'])}\n"
                f"📅 公開日: {orig['published_date']} | 🛡️ 査読: {ai.get('peer_review_status', 'Preprint（査読未定）')}\n"
                f"🔗 arXiv Abs: {orig['arxiv_url']}\n"
                f"📕 PDF: {orig['pdf_url']}\n"
                f"{hf_info}{git_info}"
                f"🏢 Source: {orig['source']}\n"
                f"📝 要約: {ai.get('summary', '要約エラー')}\n"
                f"🏷️ タグ: {', '.join(ai.get('tags', []))}\n"
                f"💡 ポイント: {ai.get('layman_point', 'なし')}\n"
                f"⭐ 興味深さ: {ai.get('interest_score', 0)}/10 | 🛠️ 汎用性: {ai.get('applicability_score', 0)}/10\n"
            )
            requests.post(DISCORD_URL, json={"content": msg})
            
            # 【確実な履歴保存】送信が確定した論文のみ履歴に登録
            history[orig['id']] = now_s
            time.sleep(1)
        
        with open(HISTORY_FILE, "w") as f:
            for pid, ts in history.items(): 
                f.write(f"{pid}|{ts}\n")
        print("All processes finished successfully.")
    else:
        requests.post(DISCORD_URL, json={"content": "⚠️ **構文エラー**: AIからのレスポンスを正しく解析できませんでした。"})

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("date", nargs="?", default=None)
        args = parser.parse_args()
        
        if args.date and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
            sys.exit(1)

        main(target_date=args.date)
    except Exception as e:
        error_reason = "".join(traceback.format_exception_only(type(e), e)).strip()
        
        # バックティック「`」の連続によるパースエラーを根絶するため、文字コードから生成
        ticks = chr(96) * 3
        
        msg_parts = [
            "🚨 **プログラム実行エラーが発生しました**",
            ticks,
            error_reason,
            ticks
        ]
        error_msg = "\n".join(msg_parts)
        
        if DISCORD_URL:
            try: 
                requests.post(DISCORD_URL, json={"content": error_msg})
            except: 
                pass
