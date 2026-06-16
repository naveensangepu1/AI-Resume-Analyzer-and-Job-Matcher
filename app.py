import gradio as gr
import os, re, webbrowser, threading, time
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util

#Skills & Abbreviations
SKILLS = [
    "python","java","c++","c","javascript","typescript","go","rust","php","kotlin","swift",
    "html","css","react","angular","vue","node","express","django","flask","fastapi",
    "sql","mysql","postgresql","mongodb","sqlite","redis","oracle",
    "machine learning","deep learning","artificial intelligence","nlp","computer vision",
    "tensorflow","pytorch","keras","scikit learn","opencv",
    "data analysis","pandas","numpy","data visualization","matplotlib","power bi","tableau",
    "aws","azure","gcp","cloud","docker","kubernetes","ci cd","jenkins","linux",
    "git","github","agile","scrum","rest api","api development","automation",
    "testing","unit testing","debugging","data structures","oop","dbms","operating systems",
]
ABBR = {"ml":"machine learning","ai":"artificial intelligence","dl":"deep learning",
        "cv":"computer vision","js":"javascript","ts":"typescript","k8s":"kubernetes",
        "dsa":"data structures","oops":"oop","ci":"ci cd","cd":"ci cd"}

model = SentenceTransformer('all-MiniLM-L6-v2')

#Core Functions
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return " ".join(p.extract_text() or "" for p in PdfReader(file).pages).lower()
    if ext == 'docx':
        return "\n".join(p.text for p in Document(file).paragraphs).lower()
    return ""

def extract_skills(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    words = text.split()
    found = {ABBR[w] for w in words if w in ABBR}
    for s in SKILLS:
        if (s in text) if " " in s else (s in words): found.add(s)
    return found

def content_score(r, j):
    sim = util.pytorch_cos_sim(model.encode(r, convert_to_tensor=True),
                               model.encode(j, convert_to_tensor=True)).item()
    return round((max(0.0, min(1.0, sim)) ** 0.5) * 100)

def skill_score(matched, job_skills):
    return round(len(matched) / len(job_skills) * 100) if job_skills else 100

def suggestion(missing, cscore):
    if not missing and cscore >= 70: return "Excellent match! This resume aligns well."
    parts = []
    if missing:
        shown = sorted(missing)[:5]
        extra = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
        parts.append(f"Consider adding: {', '.join(shown)}{extra}.")
    if cscore < 50: parts.append("Try rephrasing your summary to mirror the job description.")
    return " ".join(parts) or "Good overall match."

# ── HTML Rendering ───────────────────────────────────────────────────
def score_class(s): return "score-high" if s>=70 else "score-medium" if s>=40 else "score-low"

def make_tags(skills, cls):
    return "".join(f'<span class="tag {cls}">{s}</span>' for s in skills)

def build_card(rank, r, total):
    cls  = score_class(r["final_score"])
    rank_label = f'<span class="rank-badge">#{rank}</span> ' if total > 1 else ""
    mtags = make_tags(r["matched"], "tag-match") or '<span class="tag tag-empty">None found</span>'
    xtags = make_tags(r["missing"], "tag-missing") or '<span class="tag tag-match">None — all covered!</span>'
    return f"""
    <div class="resume-card">
      <div class="resume-header">
        <span class="resume-name">{rank_label}📄 {r['name']}</span>
        <span class="score-badge {cls}">{r['final_score']}%</span>
      </div>
      <div class="score-row"><span class="score-label">Skill Match</span>
        <div class="bar-bg"><div class="bar-fill {score_class(r['skill_score'])}-fill" style="width:{r['skill_score']}%"></div></div>
        <span class="score-value">{r['skill_score']}%</span></div>
      <div class="score-row"><span class="score-label">Content Match</span>
        <div class="bar-bg"><div class="bar-fill {score_class(r['content_score'])}-fill" style="width:{r['content_score']}%"></div></div>
        <span class="score-value">{r['content_score']}%</span></div>
      <div class="skills-section"><div class="skills-title">✅ MATCHED SKILLS</div><div class="tags">{mtags}</div></div>
      <div class="skills-section"><div class="skills-title">⚠️ MISSING SKILLS</div><div class="tags">{xtags}</div></div>
      <div class="suggestion">💡 {r['suggestion']}</div>
    </div>"""

def build_html(results):
    valid = [r for r in results if not r.get("error")]
    summary = ""
    if len(valid) > 1:
        b = valid[0]
        summary = (f"<div class='summary-bar'>📊 Analyzed {len(results)} resume(s) — "
                   f"top match: <strong>{b['name']}</strong> "
                   f"<span class='score-badge {score_class(b['final_score'])}' "
                   f"style='font-size:13px;padding:2px 10px'>{b['final_score']}%</span></div>")
    cards = []
    for rank, r in enumerate(results, 1):
        if r.get("error"):
            cards.append(f'<div class="resume-card error-card"><div class="resume-header">'
                         f'<span class="resume-name">📄 {r["name"]}</span>'
                         f'<span class="score-badge score-low">ERROR</span></div>'
                         f'<p style="color:#f87171">{r["error"]}</p></div>')
        else:
            cards.append(build_card(rank, r, len(valid)))
    return summary + "".join(cards)

#Main
def analyze(files, jd):
    if not files: return "<div class='resume-card'><p>Please upload at least one resume.</p></div>"
    if not jd or not jd.strip(): return "<div class='resume-card'><p>Please paste a job description.</p></div>"
    jskills = extract_skills(jd)
    results = []
    for f in files:
        try:
            txt     = extract_text(f)
            rskills = extract_skills(txt)
            matched = sorted(rskills & jskills)
            missing = sorted(jskills - rskills)
            cs      = content_score(txt, jd)
            ss      = skill_score(set(matched), jskills)
            results.append({"name": os.path.basename(f.name), "final_score": round(0.5*cs+0.5*ss),
                             "content_score": cs, "skill_score": ss,
                             "matched": matched, "missing": missing,
                             "suggestion": suggestion(missing, cs), "error": None})
        except Exception as e:
            results.append({"name": os.path.basename(f.name), "error": str(e)})
    results.sort(key=lambda r: r.get("final_score", -1), reverse=True)
    return build_html(results)

#CSS
CSS = """
#submit-btn button { background-color:#2ECC71!important;color:white!important;font-size:16px!important;
    padding:10px 20px!important;border-radius:10px!important;border:none!important;font-weight:700!important; }
.summary-bar { background:rgba(59,130,246,0.12);border:1px solid #3b82f6;border-radius:10px;
    padding:10px 16px;margin-bottom:16px;font-size:14px;color:#cbd5e1; }
.resume-card { background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border:1px solid #334155;
    border-radius:16px;padding:20px;margin-bottom:16px; }
.error-card { border-color:#dc2626; }
.resume-header { display:flex;justify-content:space-between;align-items:center;margin-bottom:14px; }
.resume-name { font-size:17px;font-weight:700;color:#e2e8f0; }
.rank-badge { font-size:13px;color:#fbbf24;font-weight:800; }
.score-badge { font-size:20px;font-weight:800;padding:6px 16px;border-radius:999px;color:white; }
.score-high { background:#16a34a; } .score-medium { background:#ca8a04; } .score-low { background:#dc2626; }
.score-high-fill { background:#16a34a; } .score-medium-fill { background:#ca8a04; } .score-low-fill { background:#dc2626; }
.score-row { display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px;color:#cbd5e1; }
.score-label { width:100px; }
.bar-bg { flex:1;background:#334155;border-radius:8px;height:10px;overflow:hidden; }
.bar-fill { height:100%;border-radius:8px;transition:width 0.6s ease; }
.score-value { width:42px;text-align:right; }
.skills-section { margin-top:14px; }
.skills-title { font-size:12px;font-weight:800;letter-spacing:0.05em;color:#94a3b8;margin-bottom:6px; }
.tags { display:flex;flex-wrap:wrap;gap:6px; }
.tag { padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600; }
.tag-match { background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid #16a34a; }
.tag-missing { background:rgba(239,68,68,0.15);color:#f87171;border:1px solid #dc2626; }
.tag-empty { background:rgba(148,163,184,0.15);color:#94a3b8;border:1px solid #475569; }
.suggestion { margin-top:14px;padding:10px 14px;background:rgba(59,130,246,0.1);
    border-left:3px solid #3b82f6;border-radius:8px;font-size:13px;color:#cbd5e1; }
"""

#UI
with gr.Blocks(title="Bulk Resume Scorer", css=CSS) as demo:
    gr.Markdown("<h1 style='text-align:center;color:#2E86C1'>📊 Resume Analyzer & Job Matcher</h1>"
                "<p style='text-align:center;font-size:16px;color:#117864'>Upload resumes and match them with job descriptions using NLP & AI.</p>")
    with gr.Row():
        resumes = gr.File(label="Upload Resumes", file_types=['.pdf','.docx'], file_count="multiple")
        job_desc = gr.Textbox(label="Job Description", lines=8, placeholder="Paste job description here...")
    submit = gr.Button("Analyze Resumes", elem_id="submit-btn")
    output = gr.HTML()
    submit.click(fn=analyze, inputs=[resumes, job_desc], outputs=[output])

if __name__ == "__main__":
    threading.Thread(target=lambda: (time.sleep(3), webbrowser.open("http://127.0.0.1:7860")), daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=7860)
