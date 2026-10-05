# literature-review-analyzer

A single-file FastAPI application for reading a set of academic papers together. You upload PDF files, the server extracts their text and asks Claude for a structured analysis of each paper (thesis, arguments and evidence, method, conclusion, contributions, limitations, keywords, cited titles, and a plain-language explanation). Once two or more papers are loaded, a second pass asks Claude to map the relationships between them: citation links, debates with named positions, schools of thought, how the ideas progressed, open controversies, consensus points and research gaps. A browser front end (embedded in the same Python file) shows the papers, a D3 force-directed network of the citation relationships, the debate map, and a streaming chat whose system prompt contains all of the analyses so you can ask questions across the whole set. All prompts and the interface are in Korean; the analysis JSON fields are English keys with Korean values.

## Features

- Upload one or many PDFs at once; each file is analysed independently and stored under a short random id.
- Per-paper structured analysis with a fixed JSON schema (title, authors, year, venue, main thesis, key arguments with evidence and significance, methodology, conclusion, contributions, limitations, keywords, cited titles, simple explanation with analogy).
- Automatic cross-paper analysis when at least two papers are present: citation relationships typed as support, rebuttal, extension or review; debates with positions and the papers holding each position; schools of thought; intellectual progression; key controversies; consensus; research gaps; landscape summary.
- Four tabs in the UI: Overview (landscape summary, progression, controversies, consensus, gaps), Papers (cards with the structured fields and the simple explanation), Network (D3 v7 force graph of citation relationships), Debates (positions and arguments per debate).
- Chat with server-sent-event streaming; the system prompt is rebuilt from the stored analyses on every request, and the model is told to cite paper titles or ids and to answer in Korean.
- Re-run the cross analysis on demand; delete papers (the cross analysis is cleared when fewer than two remain).
- Persistence in two JSON files, no database.

## How it works

```
browser (Tailwind + D3 + marked, embedded HTML)
   |  POST /api/upload (multipart PDFs)
   v
FastAPI (app.py)
   |-- PyMuPDF: text of all pages, truncated to 120,000 characters
   |-- Claude (claude-opus-4-6, max_tokens 4096): PAPER_ANALYSIS_PROMPT -> JSON  -> data/papers.json
   |-- if >= 2 papers: Claude (claude-opus-4-6, max_tokens 6144): CROSS_ANALYSIS_PROMPT -> JSON -> data/analysis.json
   |
   |  POST /api/chat (messages[])
   |-- build_system_prompt(papers, analysis) -> Claude (claude-sonnet-4-6, max_tokens 2048), streamed as SSE
```

Claude is asked to reply with JSON only; `extract_json` strips an optional Markdown code fence and parses the result. A paper whose analysis fails (for example unparseable JSON or a non-PDF upload) is reported in the upload response and skipped; the others are kept.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Embedded single-page UI |
| POST | `/api/upload` | Multipart `files[]`; analyses each PDF, then runs cross analysis if 2+ papers |
| GET | `/api/papers` | All stored paper analyses |
| GET | `/api/analysis` | Stored cross analysis |
| POST | `/api/reanalyze` | Re-run cross analysis (400 if fewer than 2 papers) |
| DELETE | `/api/papers/{id}` | Remove a paper |
| POST | `/api/chat` | Body `{messages: [...], api_key?: ""}`; returns `text/event-stream` with `data: {"text": ...}` chunks and `data: [DONE]` |

The server also accepts an `X-API-Key` header (upload, reanalyze, chat) or `api_key` in the chat body as a per-request override of the environment key. The bundled front end does not send either, so in practice the environment variable is required.

## Requirements

- Python 3.9 or newer.
- Packages in `requirements.txt`: `fastapi`, `uvicorn[standard]`, `anthropic`, `pymupdf`, `python-multipart`.
- An Anthropic API key with access to the models named in `app.py` (`claude-opus-4-6` for analysis, `claude-sonnet-4-6` for chat). Change the `model=` arguments if you want cheaper or different models.
- Internet access from the browser for the CDN assets (Tailwind, D3, marked).

## Installation and running

```bash
git clone <repo-url>
cd literature-review-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."     # placeholder, use your own key
python3 app.py
# open http://localhost:8000
```

Or use the helper script, which checks that the key is set, installs the requirements with `pip3`, and starts the server:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
./run.sh
```

Alternative start with uvicorn directly (for example to change the port):

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | environment variable | none | Required. Never commit it. |
| Host / port | `uvicorn.run(...)` at the bottom of `app.py` | `0.0.0.0:8000` | Binds to all interfaces by default; see limitations |
| Analysis model and max tokens | `analyze_paper`, `run_cross_analysis` in `app.py` | `claude-opus-4-6`, 4096 / 6144 | |
| Chat model and max tokens | `chat` in `app.py` | `claude-sonnet-4-6`, 2048 | |
| Text cap per PDF | `extract_pdf_text` | 120,000 characters | Longer papers are truncated |
| Storage | `DATA_DIR = Path("data")` | `./data/papers.json`, `./data/analysis.json` | Relative to the working directory |
| CORS | `CORSMiddleware` | allow all origins | |

There is no config file. Prompts are the module-level strings `PAPER_ANALYSIS_PROMPT`, `CROSS_ANALYSIS_PROMPT` and the function `build_system_prompt`; edit them to change the output schema or the response language.

## Project structure

```
.
├── app.py             # FastAPI app, prompts, Claude calls, and the embedded HTML/JS front end
├── requirements.txt
├── run.sh             # key check + pip install + python3 app.py
├── data/              # created at run time: papers.json, analysis.json (not meant to be committed)
├── LICENSE
└── README.md
```

## Usage examples

Upload from the command line:

```bash
curl -s -X POST http://localhost:8000/api/upload \
  -F "files=@paper1.pdf" -F "files=@paper2.pdf"
```

Fetch the cross analysis:

```bash
curl -s http://localhost:8000/api/analysis | python3 -m json.tool
```

Ask a question over the loaded papers (streamed):

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"두 논문의 핵심 쟁점을 비교해줘"}]}'
```

Typical browser workflow: open the page, drop PDFs on the upload area, wait for the analysis to finish (one Claude call per paper plus one cross call), browse the Network and Debates tabs, then use the chat panel to ask follow-up questions that reference specific papers.

## Data sources, licensing and attribution

- No datasets are bundled. Papers you upload are read locally with PyMuPDF; their extracted text is sent to the Anthropic API for analysis and is stored, in summarised form, in `data/*.json`. Check the licence of the papers and your institution's rules before uploading.
- Front-end libraries loaded from CDNs at run time: Tailwind CSS (MIT), D3 v7 (ISC), marked (MIT).
- Server dependencies: FastAPI (MIT), uvicorn (BSD), anthropic SDK (MIT), PyMuPDF (AGPL-3.0), python-multipart (Apache-2.0).

## Known limitations

- Text extraction is plain `page.get_text()`; scanned PDFs without a text layer produce empty input and a meaningless analysis. There is no OCR.
- Papers longer than 120,000 characters are truncated, which can drop reference lists and therefore `cited_titles`.
- Citation relationships are inferred by the model from titles and summaries, not from parsed bibliographies, so links can be missing or wrong.
- Each upload triggers full re-analysis of the cross map, which costs one large Opus call per upload batch; there is no caching or batching of paper analyses.
- Results are Korean-language by prompt design. Changing the language requires editing the prompts.
- Storage is two JSON files with no locking; concurrent uploads from several clients can overwrite each other.
- The server listens on `0.0.0.0` with permissive CORS and no authentication. Run it on a trusted machine or change the host to `127.0.0.1` before exposing it.
- Model names are hardcoded; if a model id is retired the app fails at the first analysis call.
- The UI has no per-session key entry, so the key must be present in the server environment.

## License

MIT. See `LICENSE`.
