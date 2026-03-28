# Update system
sudo apt update && sudo apt upgrade -y


# ------------------- Docker -------------------
# Install Docker
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
# Apply docker group (re-login)
newgrp docker
# Verify Docker works
docker run --rm hello-world

# ------------------- Qdrant -------------------
# Create data directory
sudo mkdir -p /var/lib/qdrant/storage
sudo chown -R $USER:$USER /var/lib/qdrant
# Create systemd service
sudo tee /etc/systemd/system/qdrant.service << 'EOF'
[Unit]
Description=Qdrant Vector Database
After=docker.service
Requires=docker.service
[Service]
Type=simple
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker stop qdrant
ExecStartPre=-/usr/bin/docker rm qdrant
ExecStart=/usr/bin/docker run --rm \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /var/lib/qdrant/storage:/qdrant/storage:z \
  qdrant/qdrant:latest
ExecStop=/usr/bin/docker stop qdrant
[Install]
WantedBy=multi-user.target
EOF
# Start Qdrant
sudo systemctl daemon-reload
sudo systemctl enable qdrant
sudo systemctl start qdrant
# Verify Qdrant works
sleep 5
curl http://localhost:6333/collections

# ------------------- Ollama -------------------
# Create data directory
sudo mkdir -p /var/lib/ollama
# Create systemd service (NO --gpus flag)
sudo tee /etc/systemd/system/ollama.service << 'EOF'
[Unit]
Description=Ollama LLM Server
After=docker.service
Requires=docker.service
[Service]
Type=simple
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker stop ollama
ExecStartPre=-/usr/bin/docker rm ollama
ExecStart=/usr/bin/docker run --rm \
  --name ollama \
  -p 11434:11434 \
  -v /var/lib/ollama:/root/.ollama \
  ollama/ollama:latest
ExecStop=/usr/bin/docker stop ollama
[Install]
WantedBy=multi-user.target
EOF
# Start Ollama
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
# Wait and pull model (smaller model recommended for CPU)
sleep 5
docker exec ollama ollama pull gemma3:4b
# Verify
curl http://localhost:11434/api/tags

# ------------------- Python -------------------
sudo apt install -y python3 python3-pip python3-venv
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# ------------------- Node -------------------
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# ------------------- RAG App directory -------------------
sudo mkdir -p /opt/rag-app
sudo chown -R $USER:$USER /opt/rag-app

# ------------------- scp RAG App -------------------
# On LOCAL machine
# gcloud compute scp --recurse \
#   ./server.py \
#   ./pyproject.toml \
#   ./uv.lock \
#   ./.env \
#   ./frontend/dist \
#   ./rag \
#   rag:/opt/rag-app/ \
#   --zone=us-central1-f \
#   --project=cs6120-471600

# -------------------- update .env -------------------
# On RAG instance
# SERVER_HOST=0.0.0.0
# SERVER_PORT=5500
# SERVER_STATIC_DIR=/opt/rag-app/dist

# QDRANT_HOST=localhost
# QDRANT_PORT=6333

# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=gemma3:4b

# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.0-flash-exp

# --------------------- rag-app.service -------------------
sudo tee /etc/systemd/system/rag-app.service << 'EOF'
[Unit]
Description=RAG Chat Application
After=network.target qdrant.service ollama.service
Wants=qdrant.service ollama.service

[Service]
Type=simple
User=csg
WorkingDirectory=/opt/rag-app
Environment="PATH=/opt/rag-app/venv/bin"
EnvironmentFile=/opt/rag-app/.env
ExecStart=/opt/rag-app/venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable rag-app
sudo systemctl start rag-app

# -------------------- Verify -------------------
# Check all services
sudo systemctl status qdrant
sudo systemctl status ollama
sudo systemctl status rag-app

# Check logs if there are issues
sudo journalctl -u rag-app -f


# ------ Test RAG App ------
# Stop
gcloud compute instances stop rag \
  --zone=us-central1-f \
  --project=cs6120-471600
# Wait a minute, then start
gcloud compute instances start rag \
  --zone=us-central1-f \
  --project=cs6120-471600

# Connect to VM
gcloud compute ssh --zone "us-central1-f" "rag" --project "cs6120-471600"
