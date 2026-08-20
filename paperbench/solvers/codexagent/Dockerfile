FROM pb-env:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y nodejs npm \
    && npm install -g @openai/codex@0.138.0 \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/agent/node-v22.19.0-linux-x64/bin:${PATH}"

RUN curl --fail --location --retry 5 \
        --output /tmp/node-v22.19.0-linux-x64.tar.xz \
        https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.xz \
    && tar -xJf /tmp/node-v22.19.0-linux-x64.tar.xz -C /home/agent \
    && rm /tmp/node-v22.19.0-linux-x64.tar.xz \
    && npm install -g @earendil-works/pi-coding-agent@0.84.0 \
    && pi --version

ENV DEBIAN_FRONTEND=
