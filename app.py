#!/usr/bin/env python3
"""
선행연구 분석기 — Academic Literature Review Analyzer
PDF 논문을 분석하고, 인용 네트워크·논쟁 지형을 시각화하며, AI 채팅으로 이해를 심화합니다.

사용법:
    ANTHROPIC_API_KEY=your_key python3 app.py
    브라우저에서 http://localhost:8000 접속
"""

import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Optional, AsyncGenerator

import fitz  # PyMuPDF
import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ───────────────────────── App & Client Setup ─────────────────────────

app = FastAPI(title="선행연구 분석기")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

def get_client(key: str = "") -> anthropic.AsyncAnthropic:
    k = key or api_key
    if not k:
        raise HTTPException(status_code=400, detail="API 키가 설정되지 않았습니다. 화면 상단에서 Anthropic API 키를 입력하세요.")
    return anthropic.AsyncAnthropic(api_key=k)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PAPERS_FILE = DATA_DIR / "papers.json"
ANALYSIS_FILE = DATA_DIR / "analysis.json"

# ───────────────────────── Data Helpers ─────────────────────────

def load_papers() -> dict:
    if PAPERS_FILE.exists():
        return json.loads(PAPERS_FILE.read_text(encoding="utf-8"))
    return {}

def save_papers(papers: dict):
    PAPERS_FILE.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")

def load_analysis() -> dict:
    if ANALYSIS_FILE.exists():
        return json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
    return {}

def save_analysis(analysis: dict):
    ANALYSIS_FILE.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

# ───────────────────────── PDF Extraction ─────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    text = "\n".join(pages)
    # 토큰 한도를 고려해 최대 120,000자로 제한
    return text[:120_000]

# ───────────────────────── JSON Extraction ─────────────────────────

def extract_json(text: str) -> dict:
    """Claude 응답에서 JSON 추출 (마크다운 코드블록 처리 포함)"""
    text = text.strip()
    # 마크다운 코드블록 제거
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    # JSON 파싱
    return json.loads(text)

# ───────────────────────── Claude Analysis Prompts ─────────────────────────

PAPER_ANALYSIS_PROMPT = """당신은 학술 논문 분석 전문가입니다. 아래 논문 텍스트를 읽고 정해진 JSON 형식으로만 응답하세요.

논문 텍스트:
{text}

다음 JSON 구조로만 응답하세요 (다른 텍스트 없이 JSON만):
{{
  "title": "논문의 정확한 제목",
  "authors": ["저자1", "저자2"],
  "year": 출판연도(숫자, 모르면 null),
  "venue": "저널 또는 학회명",
  "main_thesis": "이 논문의 핵심 주장을 1~2문장으로",
  "key_arguments": [
    {{
      "argument": "논증 내용",
      "evidence": "이 논증을 뒷받침하는 근거/증거",
      "significance": "왜 이 논증이 중요한가"
    }}
  ],
  "methodology": "연구 방법론 (실험, 설문, 문헌 고찰 등)",
  "conclusion": "논문이 도달한 최종 결론",
  "contributions": ["학문적 기여 1", "학문적 기여 2"],
  "limitations": ["한계점 1"],
  "keywords": ["핵심어1", "핵심어2", "핵심어3"],
  "cited_titles": ["참고문헌에 등장하는 논문 제목들"],
  "simple_explanation": {{
    "main_idea": "초등학생도 이해할 수 있는 핵심 아이디어 (예: '이 논문은 ~를 밝혔어요')",
    "why_matters": "왜 이 연구가 세상에 중요한지 쉬운 말로",
    "analogy": "핵심 개념을 이해하기 위한 친숙한 비유 (예: 마치 ~처럼)"
  }}
}}"""

CROSS_ANALYSIS_PROMPT = """당신은 학술 문헌을 조망하는 연구 전문가입니다. 아래 논문 요약들을 읽고 논문들 사이의 관계를 분석하여 JSON으로만 응답하세요.

분석 대상 논문들:
{papers_summary}

다음 JSON 구조로만 응답하세요:
{{
  "citation_relationships": [
    {{
      "citing_id": "인용하는 논문의 ID",
      "cited_id": "인용되는 논문의 ID",
      "relationship_type": "지지 또는 반박 또는 확장 또는 검토",
      "description": "이 두 논문의 관계를 한 문장으로"
    }}
  ],
  "debates": [
    {{
      "topic": "논쟁 주제 (간결하게)",
      "description": "이 논쟁이 무엇에 관한 것인지 설명",
      "positions": [
        {{
          "stance": "입장 이름 (예: '강한 인과론')",
          "description": "이 입장이 주장하는 바",
          "paper_ids": ["이 입장을 취하는 논문의 ID"],
          "key_arguments": ["이 입장의 핵심 논거"]
        }}
      ]
    }}
  ],
  "schools_of_thought": [
    {{
      "name": "학파/관점 이름",
      "description": "이 학파의 핵심 입장과 특징",
      "paper_ids": ["해당 논문 ID들"],
      "key_ideas": ["핵심 아이디어"]
    }}
  ],
  "intellectual_progression": "이 분야에서 아이디어가 어떻게 발전해왔는지 서술 (3~5문장)",
  "key_controversies": ["아직 해결되지 않은 주요 쟁점 1", "쟁점 2"],
  "consensus_points": ["학자들이 대체로 동의하는 사항 1", "합의 사항 2"],
  "research_gaps": ["이 논문들이 채우지 못한 연구 공백 1"],
  "landscape_summary": "이 연구 분야 전체의 지형을 요약하는 4~6문장 서술"
}}"""

# ───────────────────────── Analysis Functions ─────────────────────────

async def analyze_paper(text: str, paper_id: str, key: str = "") -> dict:
    prompt = PAPER_ANALYSIS_PROMPT.format(text=text)
    c = get_client(key)
    message = await c.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    data = extract_json(message.content[0].text)
    data["id"] = paper_id
    return data


async def run_cross_analysis(papers: dict, key: str = "") -> dict:
    summary_parts = []
    for pid, p in papers.items():
        args = " | ".join(
            a.get("argument", "") for a in p.get("key_arguments", [])
        )
        summary_parts.append(
            f"ID: {pid}\n"
            f"제목: {p.get('title', '(제목 없음)')}\n"
            f"저자: {', '.join(p.get('authors', []))}\n"
            f"연도: {p.get('year', 'n/a')}\n"
            f"핵심 주장: {p.get('main_thesis', '')}\n"
            f"주요 논증: {args}\n"
            f"키워드: {', '.join(p.get('keywords', []))}\n"
            f"인용 논문들: {', '.join(p.get('cited_titles', []))}\n"
            "---"
        )
    papers_summary = "\n".join(summary_parts)
    prompt = CROSS_ANALYSIS_PROMPT.format(papers_summary=papers_summary)
    c = get_client(key)
    message = await c.messages.create(
        model="claude-opus-4-6",
        max_tokens=6144,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_json(message.content[0].text)

# ───────────────────────── Chat System Prompt ─────────────────────────

def build_system_prompt(papers: dict, analysis: dict) -> str:
    papers_ctx = ""
    for p in papers.values():
        args = "; ".join(
            a.get("argument", "") for a in p.get("key_arguments", [])
        )
        papers_ctx += (
            f"\n### [{p['id']}] {p.get('title', '(제목 없음)')} ({p.get('year', '')})\n"
            f"저자: {', '.join(p.get('authors', []))}\n"
            f"핵심 주장: {p.get('main_thesis', '')}\n"
            f"주요 논증: {args}\n"
            f"결론: {p.get('conclusion', '')}\n"
            f"키워드: {', '.join(p.get('keywords', []))}\n"
            f"쉬운 설명: {p.get('simple_explanation', {}).get('main_idea', '')}\n"
        )

    landscape = analysis.get("landscape_summary", "")
    debates_json = json.dumps(analysis.get("debates", []), ensure_ascii=False, indent=2)
    controversies = "\n".join(f"- {c}" for c in analysis.get("key_controversies", []))

    return f"""당신은 아래 학술 논문들을 철저히 분석하고 학습한 전문 연구 보조 AI입니다.
사용자가 논문을 깊이 이해하고, 논쟁을 탐구하고, 논문 간 관계를 파악할 수 있도록 도와주세요.

## 분석된 논문들:
{papers_ctx}

## 연구 지형 요약:
{landscape}

## 주요 논쟁:
{debates_json}

## 핵심 쟁점:
{controversies}

## 응답 지침:
- 논문 내용에 충실하고 정확하게 답변하세요.
- 주장할 때는 구체적인 논문 제목 또는 ID를 인용하세요.
- 논문 간 연결고리를 적극적으로 언급하세요.
- 쉬운 설명 요청 시 비유·일상 언어를 사용하세요.
- 한국어로 답변하되, 학술 용어는 영어 병기도 괜찮습니다.
- 확실하지 않은 내용은 솔직하게 불확실하다고 말하세요."""

# ───────────────────────── API Endpoints ─────────────────────────

@app.post("/api/upload")
async def upload_papers(files: List[UploadFile] = File(...), x_api_key: str = Header(default="")):
    key = x_api_key or api_key
    get_client(key)  # validates key early

    papers = load_papers()
    results = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"error": f"{file.filename}은 PDF 파일이 아닙니다."})
            continue

        content = await file.read()
        paper_id = str(uuid.uuid4())[:8]

        try:
            text = extract_pdf_text(content)
            paper_data = await analyze_paper(text, paper_id, key)
            paper_data["filename"] = file.filename
            papers[paper_id] = paper_data
            results.append({"id": paper_id, "title": paper_data.get("title", file.filename)})
        except Exception as e:
            results.append({"error": str(e), "filename": file.filename})

    save_papers(papers)

    if len(papers) >= 2:
        try:
            analysis = await run_cross_analysis(papers, key)
            save_analysis(analysis)
        except Exception as e:
            print(f"[교차 분석 오류] {e}")

    return {"results": results, "total_papers": len(papers)}


@app.get("/api/papers")
async def get_papers():
    return load_papers()


@app.get("/api/analysis")
async def get_analysis():
    return load_analysis()


@app.post("/api/reanalyze")
async def reanalyze(x_api_key: str = Header(default="")):
    key = x_api_key or api_key
    papers = load_papers()
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="교차 분석에는 2편 이상의 논문이 필요합니다.")
    analysis = await run_cross_analysis(papers, key)
    save_analysis(analysis)
    return analysis


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    papers = load_papers()
    if paper_id not in papers:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    del papers[paper_id]
    save_papers(papers)
    # 남은 논문이 2편 미만이면 분석 초기화
    if len(papers) < 2:
        save_analysis({})
    return {"success": True}


class ChatRequest(BaseModel):
    messages: list
    api_key: str = ""


@app.post("/api/chat")
async def chat(request: ChatRequest, x_api_key: str = Header(default="")):
    key = request.api_key or x_api_key or api_key
    c = get_client(key)

    papers = load_papers()
    analysis = load_analysis()
    system = build_system_prompt(papers, analysis)

    async def generate() -> AsyncGenerator[str, None]:
        async with c.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=request.messages,
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ───────────────────────── Frontend HTML ─────────────────────────

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>선행연구 분석기</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
  .paper-card { transition: box-shadow 0.2s ease, transform 0.2s ease; }
  .paper-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(99,102,241,0.12); }
  .node circle { cursor: pointer; transition: opacity 0.2s; }
  .node circle:hover { opacity: 0.8; }
  .loading-dot { animation: bounce 1.2s infinite; }
  .loading-dot:nth-child(2) { animation-delay: 0.2s; }
  .loading-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }
  .fade-in { animation: fadeIn 0.35s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #f1f5f9; }
  ::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 4px; }
  .prose p { margin-bottom: 0.6em; line-height: 1.7; }
  .prose ul { list-style: disc; padding-left: 1.4em; margin-bottom: 0.6em; }
  .prose ol { list-style: decimal; padding-left: 1.4em; margin-bottom: 0.6em; }
  .prose h1,.prose h2,.prose h3 { font-weight: 700; margin: 0.8em 0 0.4em; }
  .prose strong { font-weight: 600; }
  .prose code { background:#ede9fe; color:#5b21b6; padding:1px 4px; border-radius:3px; font-size:0.85em; }
  .tab-active { border-bottom: 2px solid #6366f1; color: #6366f1 !important; }
  .upload-zone { border: 2px dashed #c7d2fe; transition: all 0.2s; }
  .upload-zone.drag-over { border-color: #6366f1; background: #eef2ff; }
  .link-line { stroke-opacity: 0.65; }
  .tooltip-box { pointer-events: none; transition: opacity 0.15s; }
  .sidebar-item { transition: background 0.15s, border-color 0.15s; }
  .sidebar-item.active { background: #eef2ff; border-color: #a5b4fc; }
  .argument-num { min-width: 22px; height: 22px; }
</style>
</head>
<body class="bg-slate-50 text-gray-800 h-screen flex flex-col overflow-hidden">

<!-- ── Header ── -->
<header class="bg-white border-b border-gray-200 px-5 py-3 flex items-center justify-between z-40 flex-shrink-0">
  <div class="flex items-center gap-3">
    <div class="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center shadow">
      <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
      </svg>
    </div>
    <div>
      <h1 class="text-base font-bold text-gray-900 leading-tight">선행연구 분석기</h1>
      <p class="text-xs text-gray-400">Academic Literature Review Analyzer</p>
    </div>
  </div>
  <div class="flex items-center gap-2">
    <span id="paperCount" class="text-sm text-gray-400 mr-1">논문 0편</span>
    <button id="reanalyzeBtn" onclick="reanalyze()" style="display:none" class="border border-indigo-200 text-indigo-600 text-sm px-3 py-1.5 rounded-lg hover:bg-indigo-50 transition-colors font-medium">
      ↻ 재분석
    </button>
  </div>
</header>

<!-- ── Global Drop Overlay ── -->
<div id="dropOverlay" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(99,102,241,0.92);align-items:center;justify-content:center;flex-direction:column;gap:16px;">
  <svg style="width:80px;height:80px;color:#fff;opacity:0.9" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
  </svg>
  <p style="color:#fff;font-size:1.5rem;font-weight:700;">PDF를 여기에 놓으세요</p>
  <p style="color:rgba(255,255,255,0.75);font-size:0.9rem;">여러 파일을 한 번에 분석합니다</p>
</div>

<!-- ── Upload Progress Toast ── -->
<div id="uploadToast" style="display:none;position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:8888;background:#1e1b4b;color:#fff;border-radius:14px;padding:14px 22px;box-shadow:0 8px 32px rgba(0,0,0,0.3);min-width:280px;align-items:center;gap:12px;">
  <div style="display:flex;gap:5px;align-items:center;flex-shrink:0">
    <div class="w-2 h-2 bg-indigo-300 rounded-full loading-dot"></div>
    <div class="w-2 h-2 bg-indigo-300 rounded-full loading-dot"></div>
    <div class="w-2 h-2 bg-indigo-300 rounded-full loading-dot"></div>
  </div>
  <span id="toastText" style="font-size:0.875rem;font-weight:500;">분석 중...</span>
</div>

<!-- ── Main 3-column Layout ── -->
<div class="flex flex-1 min-h-0">

  <!-- Left Sidebar -->
  <aside class="w-60 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
    <div class="px-4 py-3 border-b border-gray-100">
      <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider">논문 목록</p>
    </div>
    <div id="sidebarList" class="flex-1 overflow-y-auto p-3 space-y-1">
      <div class="text-center py-10 text-gray-300">
        <svg class="w-10 h-10 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
        <p class="text-sm">논문을 추가하세요</p>
      </div>
    </div>
    <!-- Drop hint at bottom of sidebar -->
    <div class="p-3 border-t border-gray-100 flex-shrink-0">
      <label style="display:flex;flex-direction:column;align-items:center;gap:6px;border:2px dashed #c7d2fe;border-radius:12px;padding:12px 8px;cursor:pointer;background:#f5f3ff;transition:background 0.2s" onmouseover="this.style.background='#ede9fe'" onmouseout="this.style.background='#f5f3ff'">
        <input type="file" id="fileInput" accept=".pdf" multiple style="display:none" onchange="handleFileInput(event)">
        <svg style="width:22px;height:22px;color:#818cf8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
        </svg>
        <span style="font-size:0.72rem;font-weight:600;color:#6366f1;text-align:center;line-height:1.4">PDF 드래그 또는 클릭하여 추가</span>
      </label>
    </div>
  </aside>

  <!-- Center: Tabs + Content -->
  <main class="flex-1 flex flex-col min-w-0 overflow-hidden">
    <!-- Tab Bar -->
    <div class="bg-white border-b border-gray-200 px-5 flex-shrink-0">
      <div class="flex gap-5">
        <button onclick="switchTab('overview')" id="tab-overview"
          class="tab-active py-3 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent tab-btn transition-colors">
          📊 연구 지형
        </button>
        <button onclick="switchTab('network')" id="tab-network"
          class="py-3 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent tab-btn transition-colors">
          🔗 인용 네트워크
        </button>
        <button onclick="switchTab('debates')" id="tab-debates"
          class="py-3 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent tab-btn transition-colors">
          ⚡ 논쟁 &amp; 입장
        </button>
        <button onclick="switchTab('papers')" id="tab-papers"
          class="py-3 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent tab-btn transition-colors">
          📄 논문 상세
        </button>
      </div>
    </div>

    <!-- Tab Panels -->
    <div class="flex-1 overflow-y-auto">

      <!-- ① Overview -->
      <div id="panel-overview" class="tab-panel p-6 space-y-5" style="display:block">
        <div id="overviewEmpty" class="text-center py-20 text-gray-300">
          <svg class="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          <p class="text-lg font-medium">논문을 업로드하면 분석이 시작됩니다</p>
          <p class="text-sm mt-1">2편 이상 업로드 시 연구 지형 분석이 자동 실행됩니다</p>
        </div>
        <div id="overviewContent" style="display:none" class="space-y-5">
          <div class="bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl p-6 text-white shadow-lg">
            <div class="text-xs font-bold uppercase tracking-widest text-indigo-200 mb-2">🗺️ 연구 지형 요약</div>
            <p id="landscapeSummary" class="leading-relaxed text-indigo-50"></p>
          </div>
          <div id="statsRow" class="grid grid-cols-3 gap-4"></div>
          <div id="progressionCard" class="bg-white border border-gray-200 rounded-2xl p-5">
            <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">📈 아이디어 발전 흐름</div>
            <p id="progressionText" class="text-gray-700 leading-relaxed text-sm"></p>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="bg-white border border-gray-200 rounded-2xl p-5">
              <div class="text-xs font-bold uppercase tracking-widest text-orange-500 mb-3">⚠️ 주요 쟁점</div>
              <ul id="controversiesList" class="space-y-2"></ul>
            </div>
            <div class="bg-white border border-gray-200 rounded-2xl p-5">
              <div class="text-xs font-bold uppercase tracking-widest text-green-600 mb-3">✅ 합의된 사항</div>
              <ul id="consensusList" class="space-y-2"></ul>
            </div>
            <div class="bg-white border border-gray-200 rounded-2xl p-5">
              <div class="text-xs font-bold uppercase tracking-widest text-blue-600 mb-3">🔍 연구 공백</div>
              <ul id="gapsList" class="space-y-2"></ul>
            </div>
          </div>
        </div>
      </div>

      <!-- ② Citation Network -->
      <div id="panel-network" class="tab-panel p-6 space-y-4" style="display:none">
        <div id="networkEmpty" class="text-center py-20 text-gray-300">
          <p class="text-lg font-medium">2편 이상의 논문을 업로드하면 인용 네트워크가 나타납니다</p>
        </div>
        <div id="networkContainer" style="display:none" class="space-y-4">
          <div class="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm" style="height:560px; position:relative;">
            <div class="absolute top-3 right-3 z-10 bg-white/90 rounded-xl px-3 py-2 text-xs text-gray-500 border border-gray-200 shadow-sm">
              드래그로 이동 · 스크롤로 확대/축소 · 노드 클릭 시 상세 이동
            </div>
            <div id="legendBox" class="absolute top-3 left-3 z-10 bg-white/90 rounded-xl px-3 py-2 border border-gray-200 shadow-sm space-y-1"></div>
            <svg id="networkSvg" width="100%" height="100%"></svg>
          </div>
          <div class="bg-white border border-gray-200 rounded-2xl p-5">
            <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">인용 관계 목록</div>
            <div id="citationList" class="space-y-2"></div>
          </div>
        </div>
      </div>

      <!-- ③ Debates -->
      <div id="panel-debates" class="tab-panel p-6 space-y-6" style="display:none">
        <div id="debatesEmpty" class="text-center py-20 text-gray-300">
          <p class="text-lg font-medium">2편 이상의 논문을 업로드하면 논쟁 분석이 나타납니다</p>
        </div>
        <div id="debatesContent" style="display:none" class="space-y-6"></div>
      </div>

      <!-- ④ Papers -->
      <div id="panel-papers" class="tab-panel p-6" style="display:none">
        <div id="papersEmpty" class="text-center py-20 text-gray-300">
          <p class="text-lg font-medium">논문을 추가하면 상세 분석이 표시됩니다</p>
        </div>
        <div id="papersGrid" style="display:none" class="grid grid-cols-1 xl:grid-cols-2 gap-6"></div>
      </div>

    </div>
  </main>

  <!-- Right: Chat Panel -->
  <aside class="w-[22rem] bg-white border-l border-gray-200 flex flex-col flex-shrink-0">
    <div class="px-4 py-3 border-b border-gray-200 flex-shrink-0">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
          <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/>
            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z"/>
          </svg>
        </div>
        <div>
          <p class="text-sm font-bold text-gray-900">AI 연구 비서</p>
          <p class="text-xs text-gray-400">논문에 관해 무엇이든 물어보세요</p>
        </div>
      </div>
    </div>

    <div id="chatMessages" class="flex-1 overflow-y-auto p-4 space-y-4">
      <!-- Welcome -->
      <div class="fade-in">
        <div class="flex items-start gap-2">
          <div class="w-6 h-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex-shrink-0 flex items-center justify-center mt-0.5">
            <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/>
            </svg>
          </div>
          <div class="bg-slate-100 rounded-2xl rounded-tl-none px-4 py-3 max-w-full">
            <p class="text-sm text-gray-700 leading-relaxed">안녕하세요! 📚 저는 업로드된 모든 논문을 학습한 AI 연구 비서입니다.<br><br>논문 내용, 논증 구조, 논쟁, 쉬운 설명 등 무엇이든 물어보세요!</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick buttons -->
    <div class="px-4 pt-3 flex flex-wrap gap-1.5 flex-shrink-0">
      <button onclick="quickChat('이 논문들의 핵심 차이점과 공통점을 비교해줘')"
        class="text-xs bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-colors">핵심 비교</button>
      <button onclick="quickChat('모든 논문을 초등학생도 이해할 수 있게 쉽게 설명해줘')"
        class="text-xs bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-colors">쉬운 설명</button>
      <button onclick="quickChat('이 분야의 가장 중요한 논쟁점이 무엇이고, 각 논문은 어떤 입장인지 알려줘')"
        class="text-xs bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-colors">주요 논쟁</button>
      <button onclick="quickChat('이 논문들이 발견하지 못한 연구 공백은 무엇인가? 앞으로 어떤 연구가 필요한가?')"
        class="text-xs bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-full hover:bg-indigo-100 transition-colors">연구 공백</button>
    </div>

    <!-- Input -->
    <div class="p-4 flex-shrink-0">
      <div class="flex gap-2">
        <textarea id="chatInput" rows="2" placeholder="논문에 관해 질문하세요... (Enter 전송, Shift+Enter 줄바꿈)"
          class="flex-1 border border-gray-200 rounded-xl px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-transparent"
          onkeydown="onChatKey(event)"></textarea>
        <button onclick="sendChat()" id="sendBtn"
          class="bg-indigo-600 text-white rounded-xl px-3 hover:bg-indigo-700 transition-colors flex items-center justify-center flex-shrink-0">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
          </svg>
        </button>
      </div>
    </div>
  </aside>
</div>

<!-- D3 Tooltip -->
<div id="d3Tooltip" class="tooltip-box fixed opacity-0 bg-gray-900 text-white text-xs rounded-xl px-3 py-2 shadow-xl max-w-xs z-50 pointer-events-none"></div>

<script>
// ────────── State ──────────
let papers = {};
let analysis = {};
let chatHistory = [];
let selectedFiles = [];
let currentTab = 'overview';

// ────────── Tab ──────────
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-panel').forEach(el => { el.style.display = 'none'; });
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('tab-active'));
  document.getElementById('panel-' + tab).style.display = 'block';
  document.getElementById('tab-' + tab).classList.add('tab-active');
  if (tab === 'network' && Object.keys(papers).length >= 2) renderNetwork();
}

// ────────── Global Drag & Drop ──────────
let dragCounter = 0;

document.addEventListener('dragenter', e => {
  e.preventDefault();
  dragCounter++;
  if (dragCounter === 1) document.getElementById('dropOverlay').style.display = 'flex';
});
document.addEventListener('dragleave', e => {
  dragCounter--;
  if (dragCounter === 0) document.getElementById('dropOverlay').style.display = 'none';
});
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  dragCounter = 0;
  document.getElementById('dropOverlay').style.display = 'none';
  const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
  if (files.length) doUpload(files);
});

function handleFileInput(e) {
  const files = Array.from(e.target.files);
  if (files.length) doUpload(files);
  e.target.value = '';
}

function showToast(msg) {
  const t = document.getElementById('uploadToast');
  document.getElementById('toastText').textContent = msg;
  t.style.display = 'flex';
}
function hideToast() { document.getElementById('uploadToast').style.display = 'none'; }

async function doUpload(files) {
  showToast(files.length + '편 분석 중... (논문당 약 1~2분 소요)');
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '업로드 실패');
    showToast('✓ 분석 완료!');
    await loadData();
    setTimeout(hideToast, 2000);
  } catch (e) {
    showToast('오류: ' + e.message);
    setTimeout(hideToast, 4000);
  }
}

// ────────── Data ──────────
async function loadData() {
  try {
    const [pr, ar] = await Promise.all([
      fetch('/api/papers').then(r => r.json()),
      fetch('/api/analysis').then(r => r.json()),
    ]);
    papers = pr;
    analysis = ar;
  } catch(e) { console.error(e); }
  renderSidebar();
  renderOverview();
  renderDebates();
  renderPapersGrid();
  updateHeader();
  if (currentTab === 'network' && Object.keys(papers).length >= 2) renderNetwork();
}

function updateHeader() {
  const n = Object.keys(papers).length;
  document.getElementById('paperCount').textContent = '논문 ' + n + '편';
  document.getElementById('reanalyzeBtn').style.display = n < 2 ? 'none' : 'inline-flex';
}

async function reanalyze() {
  const btn = document.getElementById('reanalyzeBtn');
  btn.textContent = '분석 중...'; btn.disabled = true;
  try { await fetch('/api/reanalyze', { method: 'POST' }); await loadData(); }
  catch(e) { alert('재분석 오류: ' + e.message); }
  finally { btn.textContent = '↻ 재분석'; btn.disabled = false; }
}

// ────────── Sidebar ──────────
function renderSidebar() {
  const el = document.getElementById('sidebarList');
  const arr = Object.values(papers);
  if (!arr.length) {
    el.innerHTML = '<div class="text-center py-10 text-gray-300 text-sm">논문을 추가하세요</div>';
    return;
  }
  el.innerHTML = arr.map(p => `
    <div class="sidebar-item border border-transparent rounded-xl p-3 cursor-pointer hover:bg-indigo-50 hover:border-indigo-100"
         onclick="jumpToPaper('${p.id}')">
      <div class="flex items-start justify-between gap-1">
        <p class="text-xs font-semibold text-gray-800 leading-snug line-clamp-2 flex-1">${p.title || p.filename}</p>
        <button onclick="event.stopPropagation(); deletePaper('${p.id}')"
          class="text-gray-300 hover:text-red-400 flex-shrink-0 ml-1 transition-colors">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>
      <p class="text-xs text-gray-400 mt-1 truncate">${(p.authors||[]).join(', ')} ${p.year ? '('+p.year+')' : ''}</p>
      <div class="flex flex-wrap gap-1 mt-1.5">
        ${(p.keywords||[]).slice(0,3).map(k =>
          '<span class="text-xs bg-indigo-50 text-indigo-600 rounded-full px-1.5 py-0.5">' + k + '</span>'
        ).join('')}
      </div>
    </div>
  `).join('');
}

function jumpToPaper(id) {
  switchTab('papers');
  setTimeout(() => {
    const el = document.getElementById('pc-' + id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 80);
}

async function deletePaper(id) {
  if (!confirm('이 논문을 삭제하시겠습니까?')) return;
  await fetch('/api/papers/' + id, { method: 'DELETE' });
  await loadData();
}

// ────────── Overview ──────────
function renderOverview() {
  const arr = Object.values(papers);
  document.getElementById('overviewEmpty').style.display = arr.length > 0 ? 'none' : 'block';
  document.getElementById('overviewContent').style.display = arr.length === 0 ? 'none' : 'block';
  if (!arr.length) return;

  document.getElementById('landscapeSummary').textContent =
    analysis.landscape_summary || '2편 이상의 논문을 업로드하면 연구 지형이 분석됩니다.';

  const debates = analysis.debates || [];
  const schools = analysis.schools_of_thought || [];
  document.getElementById('statsRow').innerHTML = [
    ['분석된 논문', arr.length, 'indigo'],
    ['주요 논쟁', debates.length, 'purple'],
    ['학파 / 관점', schools.length, 'blue'],
  ].map(([label, val, color]) =>
    `<div class="bg-white border border-gray-200 rounded-2xl p-5 text-center shadow-sm">
      <div class="text-3xl font-bold text-${color}-600">${val}</div>
      <div class="text-sm text-gray-400 mt-1">${label}</div>
    </div>`
  ).join('');

  document.getElementById('progressionText').textContent =
    analysis.intellectual_progression || '—';

  const mkList = (items, icon, cls) => items.length
    ? items.map(i => `<li class="flex items-start gap-2 text-sm text-gray-700"><span class="${cls} mt-0.5 flex-shrink-0">${icon}</span>${i}</li>`).join('')
    : '<li class="text-sm text-gray-400">—</li>';

  document.getElementById('controversiesList').innerHTML =
    mkList(analysis.key_controversies||[], '!', 'text-orange-400 font-bold');
  document.getElementById('consensusList').innerHTML =
    mkList(analysis.consensus_points||[], '✓', 'text-green-500 font-bold');
  document.getElementById('gapsList').innerHTML =
    mkList(analysis.research_gaps||[], '→', 'text-blue-400 font-bold');
}

// ────────── Debates ──────────
function renderDebates() {
  const debates = analysis.debates || [];
  const schools = analysis.schools_of_thought || [];
  const el = document.getElementById('debatesContent');
  const empty = document.getElementById('debatesEmpty');

  const hasContent = debates.length > 0 || schools.length > 0;
  empty.style.display = hasContent ? 'none' : 'block';
  el.style.display = !hasContent ? 'none' : 'block';
  if (!hasContent) return;

  const posColors = [
    { bg:'bg-blue-50', border:'border-blue-200', title:'text-blue-800', badge:'bg-blue-100 text-blue-700' },
    { bg:'bg-purple-50', border:'border-purple-200', title:'text-purple-800', badge:'bg-purple-100 text-purple-700' },
    { bg:'bg-green-50', border:'border-green-200', title:'text-green-800', badge:'bg-green-100 text-green-700' },
    { bg:'bg-orange-50', border:'border-orange-200', title:'text-orange-800', badge:'bg-orange-100 text-orange-700' },
    { bg:'bg-red-50', border:'border-red-200', title:'text-red-800', badge:'bg-red-100 text-red-700' },
  ];

  el.innerHTML = debates.map((d, di) => `
    <div class="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
      <div class="bg-gradient-to-r from-indigo-600 to-purple-700 px-6 py-4">
        <h3 class="font-bold text-white text-base">논쟁 ${di + 1}: ${d.topic}</h3>
        <p class="text-indigo-200 text-sm mt-1">${d.description || ''}</p>
      </div>
      <div class="p-5 grid grid-cols-1 md:grid-cols-${Math.min((d.positions||[]).length, 3)} gap-4">
        ${(d.positions||[]).map((pos, pi) => {
          const c = posColors[pi % posColors.length];
          const pNames = (pos.paper_ids||[]).map(pid => {
            const pp = papers[pid];
            return pp ? `<span class="text-xs bg-gray-100 text-gray-600 rounded-full px-2 py-0.5">${(pp.title||pid).slice(0,25)}…</span>` : '';
          }).join('');
          return `
            <div class="${c.bg} border ${c.border} rounded-xl p-4">
              <div class="font-bold text-sm ${c.title} mb-2">${pos.stance}</div>
              <p class="text-sm text-gray-700 mb-3 leading-relaxed">${pos.description}</p>
              ${(pos.key_arguments||[]).length ? `
                <div class="text-xs text-gray-500 font-semibold mb-1">핵심 논거</div>
                <ul class="space-y-1">
                  ${pos.key_arguments.map(a => `<li class="text-xs text-gray-600 flex items-start gap-1"><span class="mt-0.5">•</span>${a}</li>`).join('')}
                </ul>` : ''}
              ${pNames ? `<div class="mt-3 flex flex-wrap gap-1">${pNames}</div>` : ''}
            </div>`;
        }).join('')}
      </div>
    </div>
  `).join('');

  if (schools.length) {
    el.innerHTML += `
      <div class="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
        <div class="bg-gradient-to-r from-blue-600 to-cyan-600 px-6 py-4">
          <h3 class="font-bold text-white text-base">학파 및 관점</h3>
        </div>
        <div class="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
          ${schools.map(s => `
            <div class="border border-gray-200 rounded-xl p-4">
              <h4 class="font-bold text-gray-900 mb-2">${s.name}</h4>
              <p class="text-sm text-gray-600 mb-3 leading-relaxed">${s.description}</p>
              ${(s.key_ideas||[]).length ? `
                <ul class="space-y-1">
                  ${s.key_ideas.map(i => `<li class="text-xs text-gray-500 flex items-start gap-1"><span>→</span>${i}</li>`).join('')}
                </ul>` : ''}
            </div>`).join('')}
        </div>
      </div>`;
  }
}

// ────────── Papers Grid ──────────
function renderPapersGrid() {
  const arr = Object.values(papers);
  document.getElementById('papersEmpty').style.display = arr.length > 0 ? 'none' : 'block';
  document.getElementById('papersGrid').style.display = arr.length === 0 ? 'none' : 'grid';
  if (!arr.length) return;

  document.getElementById('papersGrid').innerHTML = arr.map(p => `
    <div id="pc-${p.id}" class="paper-card bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
      <!-- Header -->
      <div class="p-5 border-b border-gray-100">
        <div class="flex items-start justify-between gap-2">
          <div class="flex-1">
            <h3 class="font-bold text-gray-900 leading-snug">${p.title || p.filename}</h3>
            <p class="text-xs text-gray-400 mt-1">${(p.authors||[]).join(', ')}${p.year ? ' · ' + p.year : ''}${p.venue ? ' · ' + p.venue : ''}</p>
          </div>
          ${p.year ? `<span class="text-3xl font-black text-indigo-100 flex-shrink-0">'${String(p.year).slice(-2)}</span>` : ''}
        </div>
        <div class="flex flex-wrap gap-1 mt-3">
          ${(p.keywords||[]).map(k => `<span class="text-xs bg-indigo-50 text-indigo-700 rounded-full px-2 py-0.5">${k}</span>`).join('')}
        </div>
      </div>

      <!-- Main Thesis -->
      <div class="px-5 py-4 border-b border-gray-100">
        <div class="text-xs font-bold uppercase tracking-widest text-indigo-500 mb-2">핵심 주장</div>
        <p class="text-gray-900 font-semibold leading-relaxed text-sm">${p.main_thesis || '—'}</p>
      </div>

      <!-- Simple Explanation -->
      <div class="px-5 py-4 border-b border-gray-100 bg-amber-50/70">
        <div class="text-xs font-bold uppercase tracking-widest text-amber-600 mb-2">🌟 초등학생 버전</div>
        <p class="text-sm text-gray-700 leading-relaxed">${p.simple_explanation?.main_idea || '—'}</p>
        ${p.simple_explanation?.analogy ? `
          <div class="mt-3 bg-amber-100 rounded-xl px-4 py-3">
            <p class="text-xs text-amber-700 font-semibold mb-1">💡 비유하자면...</p>
            <p class="text-sm text-amber-900">${p.simple_explanation.analogy}</p>
          </div>` : ''}
        ${p.simple_explanation?.why_matters ? `
          <p class="text-xs text-amber-700 mt-2 leading-relaxed"><strong>왜 중요한가:</strong> ${p.simple_explanation.why_matters}</p>` : ''}
      </div>

      <!-- Key Arguments -->
      ${(p.key_arguments||[]).length ? `
        <div class="px-5 py-4 border-b border-gray-100">
          <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">주요 논증</div>
          <div class="space-y-3">
            ${p.key_arguments.map((a, i) => `
              <div class="flex gap-3 items-start">
                <span class="argument-num w-6 h-6 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold flex items-center justify-center flex-shrink-0">${i+1}</span>
                <div>
                  <p class="text-sm font-semibold text-gray-800">${a.argument}</p>
                  ${a.evidence ? `<p class="text-xs text-gray-500 mt-0.5">근거: ${a.evidence}</p>` : ''}
                  ${a.significance ? `<p class="text-xs text-indigo-500 mt-0.5">의의: ${a.significance}</p>` : ''}
                </div>
              </div>`).join('')}
          </div>
        </div>` : ''}

      <!-- Conclusion -->
      <div class="px-5 py-4 border-b border-gray-100">
        <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-2">결론</div>
        <p class="text-sm text-gray-700 leading-relaxed">${p.conclusion || '—'}</p>
      </div>

      <!-- Methodology + Contributions -->
      <div class="px-5 py-4 grid grid-cols-2 gap-4">
        ${p.methodology ? `
          <div>
            <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-1">방법론</div>
            <p class="text-xs text-gray-600 leading-relaxed">${p.methodology}</p>
          </div>` : ''}
        ${(p.contributions||[]).length ? `
          <div>
            <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-1">학문적 기여</div>
            <ul class="space-y-0.5">
              ${p.contributions.map(c => `<li class="text-xs text-gray-600">• ${c}</li>`).join('')}
            </ul>
          </div>` : ''}
      </div>
      ${(p.limitations||[]).length ? `
        <div class="px-5 pb-4">
          <div class="bg-gray-50 rounded-xl px-4 py-3">
            <div class="text-xs font-bold uppercase tracking-widest text-gray-400 mb-1">한계점</div>
            ${p.limitations.map(l => `<p class="text-xs text-gray-500">• ${l}</p>`).join('')}
          </div>
        </div>` : ''}
    </div>
  `).join('');
}

// ────────── Citation Network (D3) ──────────
function renderNetwork() {
  const arr = Object.values(papers);
  document.getElementById('networkEmpty').style.display = arr.length >= 2 ? 'none' : 'block';
  document.getElementById('networkContainer').style.display = arr.length < 2 ? 'none' : 'block';
  if (arr.length < 2) return;

  const nodes = arr.map(p => ({
    id: p.id, title: p.title || p.filename,
    year: p.year, authors: (p.authors||[]).join(', '),
    main_thesis: p.main_thesis || '',
  }));

  const rels = analysis.citation_relationships || [];
  const links = rels
    .filter(r => papers[r.citing_id] && papers[r.cited_id])
    .map(r => ({ source: r.citing_id, target: r.cited_id, type: r.relationship_type, desc: r.description }));

  const inDeg = {};
  nodes.forEach(n => inDeg[n.id] = 0);
  links.forEach(l => { if (inDeg[l.target] !== undefined) inDeg[l.target]++; });

  const typeColor = { '지지':'#22c55e', '반박':'#ef4444', '확장':'#6366f1', '검토':'#f59e0b' };
  const defaultColor = '#6366f1';

  // Year color scale
  const years = nodes.map(n => n.year).filter(Boolean);
  const minY = Math.min(...years), maxY = Math.max(...years);
  const nodeColor = n => (n.year && years.length > 1) ? d3.interpolateCool((n.year - minY) / (maxY - minY || 1)) : '#818cf8';

  const svg = d3.select('#networkSvg');
  svg.selectAll('*').remove();
  const W = svg.node().clientWidth, H = svg.node().clientHeight;

  // Defs: arrowheads per type
  const defs = svg.append('defs');
  Object.entries(typeColor).forEach(([type, color]) => {
    defs.append('marker').attr('id', 'arrow-' + type)
      .attr('viewBox', '-0 -5 10 10').attr('refX', 28).attr('refY', 0)
      .attr('orient', 'auto').attr('markerWidth', 7).attr('markerHeight', 7)
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', color);
  });
  defs.append('marker').attr('id', 'arrow-default')
    .attr('viewBox', '-0 -5 10 10').attr('refX', 28).attr('refY', 0)
    .attr('orient', 'auto').attr('markerWidth', 7).attr('markerHeight', 7)
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', defaultColor);

  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => g.attr('transform', e.transform)));

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(180))
    .force('charge', d3.forceManyBody().strength(-500))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide().radius(d => 26 + inDeg[d.id] * 10));

  const link = g.append('g').selectAll('line').data(links).join('line')
    .attr('class', 'link-line')
    .attr('stroke', d => typeColor[d.type] || defaultColor)
    .attr('stroke-width', 2)
    .attr('marker-end', d => `url(#arrow-${typeColor[d.type] ? d.type : 'default'})`);

  const linkLabel = g.append('g').selectAll('text').data(links).join('text')
    .attr('text-anchor', 'middle').attr('font-size', '9px').attr('font-weight', '600')
    .attr('fill', d => typeColor[d.type] || defaultColor)
    .text(d => d.type || '');

  const tooltip = d3.select('#d3Tooltip');
  const node = g.append('g').selectAll('g').data(nodes).join('g')
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('mouseover', (e, d) => {
      tooltip.style('opacity', 1)
        .html(`<div class="font-bold mb-1 leading-snug">${d.title}</div>
               <div class="text-gray-400">${d.authors}${d.year ? ' · ' + d.year : ''}</div>
               <div class="mt-1.5 text-gray-200 leading-snug">${d.main_thesis.slice(0, 100)}${d.main_thesis.length > 100 ? '…' : ''}</div>
               <div class="mt-1 text-indigo-300 text-xs">클릭하여 논문 상세 보기</div>`)
        .style('left', (e.pageX + 14) + 'px').style('top', (e.pageY - 10) + 'px');
    })
    .on('mouseout', () => tooltip.style('opacity', 0))
    .on('click', (e, d) => jumpToPaper(d.id));

  node.append('circle')
    .attr('r', d => 20 + inDeg[d.id] * 9)
    .attr('fill', nodeColor)
    .attr('stroke', '#fff')
    .attr('stroke-width', 3)
    .style('filter', 'drop-shadow(0 3px 6px rgba(0,0,0,0.18))');

  node.append('text')
    .attr('text-anchor', 'middle').attr('dy', '-0.1em')
    .attr('font-size', '9px').attr('font-weight', '700').attr('fill', '#fff')
    .text(d => (d.title || '').slice(0, 8) + ((d.title||'').length > 8 ? '…' : ''));

  node.append('text')
    .attr('text-anchor', 'middle').attr('dy', '1.1em')
    .attr('font-size', '8px').attr('fill', 'rgba(255,255,255,0.8)')
    .text(d => d.year || '');

  node.append('text')
    .attr('text-anchor', 'middle')
    .attr('y', d => (24 + inDeg[d.id] * 9) + 14)
    .attr('font-size', '8px').attr('fill', '#4b5563')
    .text(d => (d.title || '').slice(0, 12) + ((d.title||'').length > 12 ? '…' : ''));

  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    linkLabel.attr('x', d => (d.source.x + d.target.x) / 2)
             .attr('y', d => (d.source.y + d.target.y) / 2 - 4);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Legend
  const legEl = document.getElementById('legendBox');
  legEl.innerHTML = Object.entries(typeColor).map(([type, color]) =>
    `<div class="flex items-center gap-2">
      <div class="w-5 h-0.5" style="background:${color}"></div>
      <span class="text-xs text-gray-600">${type}</span>
    </div>`
  ).join('');

  // Citation list
  const clEl = document.getElementById('citationList');
  if (!rels.length) { clEl.innerHTML = '<p class="text-sm text-gray-400">분석된 인용 관계가 없습니다</p>'; return; }
  const tc = { '지지':'text-green-600', '반박':'text-red-600', '확장':'text-indigo-600', '검토':'text-yellow-600' };
  clEl.innerHTML = rels.map(r => {
    const a = papers[r.citing_id], b = papers[r.cited_id];
    if (!a || !b) return '';
    return `<div class="flex items-center gap-2 text-xs bg-slate-50 rounded-lg px-3 py-2">
      <span class="font-medium text-gray-700 truncate max-w-[130px]">${(a.title||'').slice(0,25)}…</span>
      <span class="font-bold ${tc[r.relationship_type]||'text-indigo-600'} flex-shrink-0">→ ${r.relationship_type}</span>
      <span class="text-gray-500 truncate max-w-[130px]">${(b.title||'').slice(0,25)}…</span>
      ${r.desc ? `<span class="text-gray-400 ml-auto hidden xl:block">${r.desc.slice(0,40)}…</span>` : ''}
    </div>`;
  }).join('');
}

// ────────── Chat ──────────
async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMsg('user', msg);
  chatHistory.push({ role: 'user', content: msg });
  const loadId = addLoading();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: chatHistory }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let full = '';
    removeLoading(loadId);
    const msgEl = addMsg('assistant', '');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value);
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const d = JSON.parse(line.slice(6));
            if (d.text) { full += d.text; msgEl.querySelector('.msg-body').innerHTML = marked.parse(full); }
          } catch {}
        }
      }
      document.getElementById('chatMessages').scrollTop = 999999;
    }
    chatHistory.push({ role: 'assistant', content: full });
  } catch(e) {
    removeLoading(loadId);
    addMsg('assistant', '오류: ' + e.message);
  }
}

function quickChat(msg) {
  document.getElementById('chatInput').value = msg;
  sendChat();
}

function onChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}

function addMsg(role, content) {
  const el = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'fade-in';
  if (role === 'user') {
    div.innerHTML = `<div class="flex justify-end">
      <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none px-4 py-3 max-w-[85%] text-sm leading-relaxed">${content}</div>
    </div>`;
  } else {
    div.innerHTML = `<div class="flex items-start gap-2">
      <div class="w-6 h-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex-shrink-0 flex items-center justify-center mt-0.5">
        <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
          <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/>
        </svg>
      </div>
      <div class="bg-slate-100 rounded-2xl rounded-tl-none px-4 py-3 max-w-full flex-1 min-w-0">
        <div class="prose text-sm text-gray-700 msg-body">${content ? marked.parse(content) : ''}</div>
      </div>
    </div>`;
  }
  el.appendChild(div);
  el.scrollTop = 999999;
  return div;
}

function addLoading() {
  const id = 'loading-' + Date.now();
  const el = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.id = id; div.className = 'fade-in';
  div.innerHTML = `<div class="flex items-start gap-2">
    <div class="w-6 h-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex-shrink-0 flex items-center justify-center">
      <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
        <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/>
      </svg>
    </div>
    <div class="bg-slate-100 rounded-2xl rounded-tl-none px-4 py-3">
      <div class="flex gap-1 items-center">
        <div class="w-2 h-2 bg-gray-400 rounded-full loading-dot"></div>
        <div class="w-2 h-2 bg-gray-400 rounded-full loading-dot"></div>
        <div class="w-2 h-2 bg-gray-400 rounded-full loading-dot"></div>
      </div>
    </div>
  </div>`;
  el.appendChild(div);
  el.scrollTop = 999999;
  return id;
}

function removeLoading(id) { const el = document.getElementById(id); if (el) el.remove(); }

// ────────── Init ──────────
loadData();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)


# ───────────────────────── Run ─────────────────────────

if __name__ == "__main__":
    import uvicorn

    if not api_key:
        print("⚠️  경고: ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export ANTHROPIC_API_KEY='your-key-here'  후 재실행하세요.\n")

    print("=" * 55)
    print("  선행연구 분석기 (Literature Review Analyzer)")
    print("=" * 55)
    print("  브라우저에서 접속: http://localhost:8000")
    print("  종료: Ctrl+C")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
