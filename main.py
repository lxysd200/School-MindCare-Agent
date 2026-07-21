from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from api.router import router
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="MindBridge Python", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境请替换为具体前端域名，如 ["http://localhost:3000"]
        allow_credentials=True,
        allow_methods=["*"],  # 包含 OPTIONS, POST, GET 等
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def no_cache_frontend_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store"
        return response
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)