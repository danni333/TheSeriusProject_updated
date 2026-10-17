# ===== Stage 1: Build =====
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake g++ git make ninja-build libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF \
    && cmake --build build --config Release -j$(nproc)

# ===== Stage 2: Runtime =====
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/build/serius /app/serius
COPY --from=builder /build/build/src/Release/serius_core.lib /app/lib/ 2>/dev/null || true

EXPOSE 8080
ENV LD_LIBRARY_PATH=/app/lib
ENTRYPOINT ["./serius", "--server", "--port", "8080"]
