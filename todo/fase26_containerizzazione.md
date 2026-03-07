# Fase 26 — Deployment Locale & Cloud (Docker, ONNX Runtime, K8s GPU Scheduling)

> In produzione non si installano dipendenze a mano su ogni server.
> Docker multi-stage per immagini leggere, K8s per scalare su GPU.

---

## 1. Docker Multi-Stage Build

### Dockerfile

```dockerfile
# ===== Stage 1: Build =====
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y cmake g++ git
WORKDIR /build
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# ===== Stage 2: Runtime =====
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Solo le librerie runtime, niente compilatori
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/theseriusproject /app/theseriusproject
COPY --from=builder /build/build/lib*.so /app/lib/

WORKDIR /app
EXPOSE 8080
ENV LD_LIBRARY_PATH=/app/lib
ENTRYPOINT ["./theseriusproject", "--server", "--port", "8080"]
```

### Dimensioni risultanti

| Stage | Contenuto | Dimensione |
|---|---|---|
| Builder | CUDA SDK + CMake + sorgenti | ~8 GB |
| **Runtime** | Solo eseguibile + CUDA runtime | **~2 GB** |

---

## 2. Docker Compose (sviluppo locale)

```yaml
version: '3.8'
services:
  llm-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models      # pesi del modello
      - ./data:/data           # dataset
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/models/model.safetensors
      - NUM_THREADS=8
```

---

## 3. Kubernetes con GPU

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm
  template:
    spec:
      containers:
      - name: llm
        image: registry.example.com/theseriusproject:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1        # 1 GPU per pod
            memory: "32Gi"
          requests:
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: model-weights
          mountPath: /models
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: llm
```

### NVIDIA Device Plugin

```bash
# Installa il plugin K8s per GPU NVIDIA
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verifica che le GPU siano visibili
kubectl describe nodes | grep nvidia.com/gpu
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `Dockerfile` | Multi-stage build (builder + runtime) |
| `docker-compose.yml` | Setup locale con GPU |
| `k8s/deployment.yaml` | Deployment K8s |
| `k8s/service.yaml` | Service + LoadBalancer |
| `k8s/pvc.yaml` | PersistentVolumeClaim per i pesi |
| `scripts/build_docker.sh` | Script di build automatizzato |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_docker_build` | Build multi-stage completa senza errori |
| `test_docker_run` | Container avvia e risponde su :8080 |
| `test_gpu_access` | `nvidia-smi` funziona dentro il container |
| `test_k8s_deploy` | Pod schedulato su nodo con GPU |
| `test_health_check` | Endpoint `/health` risponde 200 |
