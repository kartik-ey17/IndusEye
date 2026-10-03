# SentinelAI operator dashboard

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` when FastAPI is not at `http://127.0.0.1:8000`. The dashboard uses browser `getUserMedia`, sends compressed still frames to FastAPI, and only displays the processed face-blurred image returned by the Python service.
