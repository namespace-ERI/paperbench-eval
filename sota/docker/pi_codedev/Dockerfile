FROM pb-env-codex:latest

ARG PI_VERSION=0.84.0
ARG RUNTIME_PROXY_URL=http://172.17.0.1:7895

ENV http_proxy=${RUNTIME_PROXY_URL} \
    https_proxy=${RUNTIME_PROXY_URL} \
    HTTP_PROXY=${RUNTIME_PROXY_URL} \
    HTTPS_PROXY=${RUNTIME_PROXY_URL} \
    all_proxy=${RUNTIME_PROXY_URL} \
    ALL_PROXY=${RUNTIME_PROXY_URL} \
    npm_config_proxy=${RUNTIME_PROXY_URL} \
    npm_config_https_proxy=${RUNTIME_PROXY_URL} \
    NO_PROXY=127.0.0.1,localhost,::1,arex.autoresearch.eval.baai.ac.cn,10.1.1.47 \
    no_proxy=127.0.0.1,localhost,::1,arex.autoresearch.eval.baai.ac.cn,10.1.1.47

RUN set -eux; \
    test "$(pi --version)" = "${PI_VERSION}"; \
    ln -sf /home/agent/node-v22.19.0-linux-x64/bin/node /usr/local/bin/node; \
    ln -sf /home/agent/node-v22.19.0-linux-x64/bin/npm /usr/local/bin/npm; \
    ln -sf /home/agent/node-v22.19.0-linux-x64/bin/npx /usr/local/bin/npx; \
    ln -sf /home/agent/node-v22.19.0-linux-x64/bin/pi /usr/local/bin/pi; \
    pi --version

LABEL org.paperbench.purpose="Qwen Code-Dev with preinstalled PI and default proxy" \
      org.paperbench.pi-version="${PI_VERSION}"
