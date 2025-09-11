
<h1 align="center">My API Project · PolitenessLayers integration</h1>

<p align="center">
  <em>Serve your notebook logic as a clean HTTP API, powered by FastAPI, and integrate with <code>PolitenessLayers</code>.</em><br>
  使用 FastAPI 暴露你的 API，并无缝接入 <code>PolitenessLayers</code>（从 GitHub 安装）。
</p>

---

## ✨ Features
- **Notebook → Package**: Code auto-extracted from <code>Untitled.ipynb</code> into <code>myapi/api.py</code>.
- **FastAPI app**: Minimal <code>app.py</code> with auto-wrapping of zero-arg functions into <code>GET /&lt;func&gt;</code> endpoints.
- **PolitenessLayers**: Installed directly via <code>pip</code> from GitHub (see <code>requirements.txt</code>).
- **One command run**: <code>uvicorn app:app --host 0.0.0.0 --port 8000</code> and you're live.
- **English-only comments** in scaffolding to keep things consistent.

## 📦 Structure
```
my_api_project/
├── app.py                 # FastAPI entrypoint (auto-wrapping)
├── myapi/
│   ├── __init__.py
│   └── api.py            # Code extracted from your notebook (comments are in English)
├── requirements.txt       # Includes FastAPI + PolitenessLayers (from GitHub)
├── .gitignore
└── README.md
```

## 🚀 Quickstart
```bash
# 1) (Optional) Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies (includes FastAPI, Uvicorn, and PolitenessLayers from GitHub)
pip install -r requirements.txt

# 3) Run the API
uvicorn app:app --host 0.0.0.0 --port 8000

# 4) Open the Swagger UI
# http://localhost:8000/docs
```

> 💡 Tip (中文): 如果你在 <code>myapi/api.py</code> 已经定义了 <code>app = FastAPI(...)</code>，框架会自动复用它；否则会自动将**无必填参数**的函数暴露为 <code>GET</code> 接口。

## 🤝 Using <code>PolitenessLayers</code> in your API
The dependency is installed from GitHub:
```text
git+https://github.com/keyu-wang-2002/PolitenessLayers.git#egg=politenesslayers
```
A minimal usage pattern inside <code>myapi/api.py</code> could look like:
```python
# Example only — adapt to the actual API of PolitenessLayers
try:
    import politenesslayers  # or the correct import path defined by the repo
except ImportError:
    politenesslayers = None

def analyze_politeness(text: str = "Could you please review my PR?"):
    """Call PolitenessLayers to score/annotate politeness for a single text."""
    if politenesslayers is None:
        return {"error": "PolitenessLayers not available"}
    # TODO: replace with the real API from the repo
    # features = politenesslayers.extract_features([text])
    # score = politenesslayers.score(features)
    # return {"text": text, "features": features, "score": score}
    return {"message": "Wire up PolitenessLayers here."}
```
Then you can expose it via HTTP by adding a lightweight FastAPI route in <code>app.py</code> (if you prefer explicit endpoints):
```python
from fastapi import FastAPI
from pydantic import BaseModel
from myapi.api import analyze_politeness

app = FastAPI(title="My API (PolitenessLayers)")

class TextIn(BaseModel):
    text: str

@app.post("/analyze_politeness")
async def analyze(payload: TextIn):
    return analyze_politeness(payload.text)
```

## 🧪 Local smoke test
```bash
curl -X POST http://localhost:8000/analyze_politeness   -H "Content-Type: application/json"   -d '{"text":"Would you mind taking a quick look?"}'
```

## 📚 API Docs
- Interactive docs: <code>/docs</code>
- OpenAPI schema: <code>/openapi.json</code>

## 🧭 Publish to GitHub
```bash
git init
git add .
git commit -m "Initial commit: export API + PolitenessLayers integration"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO>.git
git push -u origin main
```

## 🧱 Notes
- All scaffolding comments are written in **English**.
- If the import path for <code>PolitenessLayers</code> differs (e.g., <code>from politeness_layers import ...</code>), update the import accordingly.
- If the upstream repo requires extra dependencies (e.g., <code>torch</code>, <code>transformers</code>), add them to <code>requirements.txt</code>.
- For production, consider: logging, CORS, request validation via Pydantic models, and tests.
