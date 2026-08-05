# 🌐 Authentix Deployment Guide: Render, Railway & Vercel

This guide explains how to deploy the **Authentix** platform to production using **Render** / **Railway** for the FastAPI + AI Backend and **Vercel** for the React frontend.

---

## 🏗️ Architecture Overview

| Component | Technology | Platform |
| :--- | :--- | :--- |
| **Frontend** | React 19 + Vite | **Vercel** |
| **Backend & AI Engine** | FastAPI + PyTorch + FFmpeg | **Render** or **Railway** (Docker) |
| **Database** | PostgreSQL | **Render Postgres** or **Railway Postgres** |

---

## 1. 🐍 Backend & AI Models Deployment (Render / Railway)

### Option A: Deploying on Render (Recommended)

Render provides native Docker container hosting with built-in PostgreSQL databases.

#### Step 1: Create a PostgreSQL Database on Render
1. Go to [Render Dashboard](https://dashboard.render.com) and click **New +** -> **PostgreSQL**.
2. Name: `authentix-db`
3. Region: Select your closest region (e.g. Oregon / Frankfurt).
4. Click **Create Database**.
5. Copy the **Internal Database URL** (e.g., `postgres://user:password@dpg-xxx:5432/authentix_db`).
   *Note: Modify the prefix to `postgresql+asyncpg://user:password@dpg-xxx:5432/authentix_db` for SQLAlchemy async driver.*

#### Step 2: Deploy Backend Web Service via Docker
1. Click **New +** -> **Web Service**.
2. Connect your GitHub repository (`Authentix`).
3. Select **Docker** as the Runtime (Render detects the root `Dockerfile`).
4. Set Instance Type to **Standard** (2 GB RAM or higher recommended for deep learning inference).
5. Add the following **Environment Variables**:

| Variable | Recommended Value |
| :--- | :--- |
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@dpg-xxx:5432/authentix_db` |
| `SECRET_KEY` | Generate with `openssl rand -hex 32` |
| `ALGORITHM` | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` |
| `CORS_ORIGINS` | `https://your-frontend.vercel.app,http://localhost:5173` |

6. Click **Create Web Service**.
7. Render will automatically build the `Dockerfile`, install FFmpeg/PyTorch, apply Alembic database migrations, and launch Uvicorn on port `$PORT`.
8. Copy your backend URL: `https://authentix-backend.onrender.com`.

---

### Option B: Deploying on Railway

1. Go to [Railway.app](https://railway.app) and create a **New Project**.
2. Select **Provision PostgreSQL** to add a database.
3. Click **+ New** -> **GitHub Repo** -> Select `Authentix`.
4. Railway auto-detects the root `Dockerfile`.
5. Under **Variables**, add:
   - `DATABASE_URL`: `${{Postgres.DATABASE_URL}}` (Prefix with `postgresql+asyncpg://`)
   - `SECRET_KEY`: `<your-random-32-byte-hex>`
   - `CORS_ORIGINS`: `https://your-frontend.vercel.app`
6. Click **Deploy**. Generate a public domain under Settings -> Networking.

---

## 2. ⚡ Frontend Deployment (Vercel)

Vercel provides instant global CDN deployment for Vite React applications.

### Step 1: Import Project to Vercel
1. Go to [Vercel Dashboard](https://vercel.com/new) and click **Import Project**.
2. Connect your GitHub repository (`Authentix`).

### Step 2: Configure Project Settings
- **Framework Preset**: `Vite`
- **Root Directory**: Select `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### Step 3: Add Environment Variables
In the Vercel deployment form, add:

| Environment Variable | Value |
| :--- | :--- |
| `VITE_API_BASE_URL` | `https://authentix-backend.onrender.com` (Your deployed Render/Railway API URL) |

### Step 4: Deploy
Click **Deploy**. Vercel will build the React app and deploy it. Client-side SPA routes (`/dashboard`, `/history`, `/analysis/:id`) are handled automatically by `frontend/vercel.json`.

---

## 🧪 Post-Deployment Verification

1. Open your Vercel URL: `https://authentix.vercel.app`.
2. Register a new user account.
3. Upload a sample video file to test the end-to-end multimodal AI detection pipeline.
4. Verify that video frames, audio waveforms, score calibration, reasoning claims, and PDF report downloads function properly.
