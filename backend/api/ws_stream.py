import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from .routes_train import RUNS

router = APIRouter()

@router.websocket("/train/stream")
async def stream(ws: WebSocket, run_id: str = Query(...)):
    await ws.accept()
    if run_id not in RUNS:
        await ws.close(code=1008)
        return
    run = RUNS[run_id]
    queue: asyncio.Queue = asyncio.Queue()
    run["queues"].add(queue)

    await ws.send_json({"run_id": run_id, "event": "connected", "status": run["status"]})
    # if already in error, surface last_error immediately
    if run.get("status") == "error" and run.get("last_error"):
        await ws.send_json({"run_id": run_id, "event": "error", "message": run["last_error"][:2000]})

    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        run["queues"].discard(queue)
