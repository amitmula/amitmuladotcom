# Summary of Recent Work

*   **Goal:** Integrated AI responses (from Ollama) into your terminal-like portfolio website.
*   **Backend:** Created a Node.js/Express.js backend (`server.js`, `package.json`) to act as a proxy for Ollama, listening on port 3000.
*   **Frontend:** Modified `index.html` to include an `ai` command, which sends prompts to the backend and streams AI responses into the terminal.
*   **Dockerization:**
    *   Created `Dockerfile.backend` for the Node.js service.
    *   Updated `nginx.conf` to configure Nginx as a reverse proxy, directing `/api/` requests to the `backend` service (using dynamic resolution to prevent startup errors).
    *   Modified `index.html` to use the relative `/api/chat` endpoint.
    *   Updated `docker-compose.yaml` to orchestrate both the `frontend` (Nginx) and `backend` (Node.js) services.
*   **Troubleshooting & Fixes:**
    *   Resolved Nginx `host not found` error by using Nginx `resolver` and a variable for `proxy_pass` in `nginx.conf`.
    *   Fixed backend container's inability to connect to Ollama on the host by updating `server.js` to use `http://host.docker.internal:11434` and adding `extra_hosts` to the `backend` service in `docker-compose.yaml`.
*   **Next Steps:** The plan is to implement Retrieval-Augmented Generation (RAG) to personalize AI responses using your data.
