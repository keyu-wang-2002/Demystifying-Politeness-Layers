# Minimal FastAPI wrapper around functions found in myapi/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import inspect
import types
import importlib

app = FastAPI(title="My API (auto-wrapped)")

# Import user code
mod = importlib.import_module("myapi.api")

# Expose simple GET endpoints for functions with no required args
for name, obj in vars(mod).items():
    if callable(obj) and isinstance(obj, (types.FunctionType,)):
        sig = inspect.signature(obj)
        # Only export zero-arg functions as GET for safety
        cond = all(p.default != inspect._empty or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in sig.parameters.values()) or len(sig.parameters)==0
        if cond:
            route = f"/{name}"
            def make_handler(func):
                async def handler():
                    res = func()
                    return {"result": res}
                return handler
            app.get(route)(make_handler(obj))