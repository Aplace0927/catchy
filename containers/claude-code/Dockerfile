FROM ghcr.io/betarixm/catchy-ubuntu:26.04

# Claude Code
WORKDIR /tmp
RUN curl -fsSL https://claude.ai/install.sh -o /tmp/install-claude-code.sh \
    && bash /tmp/install-claude-code.sh \
    && rm /tmp/install-claude-code.sh

ENV CLAUDE_CONFIG_DIR=/metadata/.claude
ENV PATH="/root/.local/bin:${PATH}"
COPY settings.json /metadata/.claude/settings.json
WORKDIR /workspace
