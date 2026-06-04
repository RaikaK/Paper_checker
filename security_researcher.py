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
MAX_POSTS = 2             # 毎日最大何本投稿するか (1または2)
GEMINI_KEY = os.getenv("GEMINI_API_KEY_2")
DISCORD_URL = os.getenv("SECURITY_WEBHOOK_URL")  # ご指定の環境変数名

HISTORY_FILE = "history_security.txt"  # ゲームAI版と混ざらないように別ファイル名にしています
RETENTION_DAYS = 30       # 履歴保持期間
OUTPUT_DIR = "outputs"
ARCHIVE_DIR = "archives"

if not DISCORD_URL:
    print("Error: SECURITY_WEBHOOK_URL is missing.")
    sys.exit(1)

# ディレクトリの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# クライアント初期化
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
arxiv_client = arxiv.Client()

# セキュリティ分野で注目度の高い組織・ベンダー・テック企業
SECURITY_TECH_KEYWORDS = [
    'google', 'microsoft', 'meta', 'apple', 'amazon', 'openai', 'nvidia', 'anthropic',
    'palo alto', 'crowdstrike', 'fireeye', 'mandiant', 'fortinet', 'checkpoint', 'cisco', 
    'cloudflare', 'kaspersky', 'mitre', 'nsa', 'nist'
]

def clean_id(raw_id):
    if not raw_id: return ""
    return re.sub(r'v\d+$', '', raw_id.strip())

def is_benchmark_paper(title, summary):
    blacklist = ['benchmark', 'dataset', 'evaluation', 'leaderboard', 'benchmarks', 'datasets']
    text = f"{title} {summary}".lower()
    return any(word in text for word in blacklist)

def extract_github_url(text):
    if not text: return ""
    match = re.search(r"https?://github\.com/[^\s,\]\)]+", text)
    return match.group(0) if match else ""

def check_security_tech(summary, comment):
    text = f"{summary} {comment}".lower()
    return any(company in text for company in SECURITY_TECH_KEYWORDS)

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

def get_security_papers(history):
    print("Fetching Cryptography and Security papers from arXiv...")
    # cs.CR (Cryptography and Security) をダイレクトにターゲットにします
    query = "cat:cs.CR"
    search = arxiv.Search(query=query, max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    try:
        results = list(arxiv_client.results(search))
        now = datetime.now(timezone.utc)
        
        # タイムスタンプの基準を設定
        one_day_ago = now - timedelta(hours=36) # 時差を考慮し約1.5日前までを前日枠とする
        one_month_ago = now - timedelta(days=30)
        
        yesterday_papers = []
        older_papers = []
        
        for r in results:
            pid = clean_id(r.entry_id.split('/')[-1])
            
            # 共通フィルタリング（既読・1ヶ月以上古い・ベンチマークを除外）
            if pid in history: continue
            if r.published < one_month_ago: continue
            if is_benchmark_paper(r.title, r.summary): continue
                
            git_url = extract_github_url(r.comment) or extract_github_url(r.summary)
            is_big_tech = check_security_tech(r.summary, r.comment)
            
            # スコアリング（PoCのGitHubコードありは最優先、主要組織は次点）
            score = 0
            if git_url: score += 10
            if is_big_tech: score += 5
            
            paper_data = {
                "id": pid,
                "title": r.title,
                "summary": r.summary,
                "pdf_url": r.pdf_url,
                "arxiv_url": f"https://arxiv.org/abs/{pid}",
                "source": "arXiv (cs.CR)",
                "published_date": r.published.strftime('%Y-%m-%d'),
                "github_url": git_url,
                "is_big_tech": is_big_tech,
                "score": score
            }
            
            # 前日（直近）の論文か、それより古い（1ヶ月以内）かでグループ分け
            if r.published >= one_day_ago:
                yesterday_papers.append(paper_data)
            else:
                older_papers.append(paper_data)
        
        # それぞれスコア順にソート
        yesterday_papers.sort(key=lambda x: x['score'], reverse=True)
        older_papers.sort(key=lambda x: x['score'], reverse=True)
        
        # 前日分を最優先し、足りない枠（MAX_POSTS）を1ヶ月以内の過去分から補填
        final_candidates = yesterday_papers + older_papers
        return final_candidates
        
    except Exception as e:
        print(f"arXiv Fetch Error: {e}")
        return []

def call_llm(prompt):
    """gemini-3.1-flash-lite のみを使用"""
    TARGET_MODEL = 'gemini-3.1-flash-lite'
    
    if client:
        try:
            print(f"Trying Google Gemini ({TARGET_MODEL})...")
            res = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
            if res.text: 
                return res.text, f"Gemini ({TARGET_MODEL})"
        except Exception as e:
            print(f"Gemini {TARGET_MODEL} failed: {e}")
            
    return None, None

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

def main():
    history = manage_history()
    candidates = get_security_papers(history)
    
    if not candidates:
        print("配信対象（直近1ヶ月以内）の新しいセキュリティー論文はありませんでした。")
        return
        
    # 上位件数を抽出（前日分があればそれが最優先で入る）
    final_papers = candidates[:MAX_POSTS]
    print(f"Processing {len(final_papers)} papers with LLM...")

    llm_input_data = []
    for idx, p in enumerate(final_papers):
        llm_input_data.append({
            "index": idx,
            "title": p['title'],
            "summary": p['summary'],
            "source": p['source']
        })

    prompt = f"""
    高度なサイバーセキュリティ・暗号理論の専門家として振る舞ってください。提示されたすべての論文について、指定のJSON配列形式でのみ翻訳・解説を出力してください。
    提供された【すべてのインデックス】を必ず網羅してください。JSON以外の解説テキスト文は一切不要です。

    [
      {{
        "index": 0,
        "title_ja": "タイトル(日本語和訳)",
        "title_en": "元の英語タイトル",
        "summary": "技術的な要約(日本語で3行程度。脆弱性、攻撃手法、防御機構、暗号アルゴリズム等の詳細を含めること)",
        "tags": ["タグ1", "タグ2"],
        "layman_point": "この研究が実世界のセキュリティやプライバシーにどう影響するのか、専門外の人でも凄さや重要性がわかるポイント(日本語)",
        "interest_score": 10,
        "applicability_score": 5
      }}
    ]

    データ: {json.dumps(llm_input_data, ensure_ascii=False)}
    """
    
    report_text, used_model = call_llm(prompt)
    if not report_text:
        requests.post(DISCORD_URL, json={"content": "⚠️ **エラー**: gemini-3.1-flash-lite からの解説取得に失敗しました。"})
        return

    selected_results = parse_json_from_text(report_text)
    
    if selected_results:
        today = datetime.now().strftime('%Y-%m-%d')
        valid_selected = []
        for item in selected_results:
            idx = item.get('index')
            if idx is not None and idx < len(final_papers):
                valid_selected.append((final_papers[idx], item))
        
        # ヘッダー通知
        header = f"🛡️ **【Security】本日({today})の厳選論文** (AI: {used_model})\n"
        for i, (orig, ai) in enumerate(valid_selected, 1):
            header += f"{i}. {ai.get('title_ja', orig['title'])}\n"
        requests.post(DISCORD_URL, json={"content": header})
        time.sleep(1)

        # 各論文の詳細を送信
        now_s = datetime.now(timezone.utc).isoformat()
        for orig, ai in valid_selected:
            git_info = f"💻 GitHub (PoC/Code): {orig['github_url']}\n" if orig['github_url'] else ""
            tag_tech = "✨ 注目のセキュリティ機関/テック企業論文\n" if orig['is_big_tech'] else ""
            
            msg = (
                f"--------------------------------------------\n"
                f"📄 **{ai.get('title_ja', orig['title'])}**\n"
                f"🔤 原題: {ai.get('title_en', orig['title'])}\n"
                f"📅 公開日: {orig['published_date']}\n"
                f"🔗 arXiv Abs: {orig['arxiv_url']}\n"
                f"📕 PDF: {orig['pdf_url']}\n"
                f"{git_info}{tag_tech}"
                f"📝 要約: {ai.get('summary', '要約エラー')}\n"
                f"🏷️ タグ: {', '.join(ai.get('tags', []))}\n"
                f"💡 セキュリティ視点ポイント: {ai.get('layman_point', 'なし')}\n"
                f"⭐ 興味深さ: {ai.get('interest_score', 0)}/10 | 🛠️ 汎用性: {ai.get('applicability_score', 0)}/10\n"
            )
            requests.post(DISCORD_URL, json={"content": msg})
            
            # 履歴に記録（セキュリティ用に別ファイルに保存されます）
            history[orig['id']] = now_s
            with open(HISTORY_FILE, "w") as f:
                for pid, ts in history.items(): 
                    f.write(f"{pid}|{ts}\n")
            time.sleep(1)
            
        print("All processes finished successfully.")
    else:
        requests.post(DISCORD_URL, json={"content": "⚠️ **構文エラー**: AIからのレスポンスを正しく解析できませんでした。"})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_reason = "".join(traceback.format_exception_only(type(e), e)).strip()
        ticks = chr(96) * 3
        error_msg = f"🚨 **プログラム実行エラーが発生しました**\n{ticks}\n{error_reason}\n{ticks}"
        if DISCORD_URL:
            try: requests.post(DISCORD_URL, json={"content": error_msg})
            except: pass
