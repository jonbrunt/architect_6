from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.agent_supervisor import graph

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the graph
# ===================FILL IN: Add Routes========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)