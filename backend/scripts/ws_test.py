import asyncio, json, sys
import websockets

async def main(run_id: str):
    url = f"ws://localhost:8000/api/train/stream?run_id={run_id}"
    async with websockets.connect(url) as ws:
        print("Connected to", url)
        while True:
            msg = await ws.recv()
            print(msg)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ws_test.py <run_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
