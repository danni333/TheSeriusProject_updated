// Serius Chat UI - Client-side logic

class ChatUI {
    constructor() {
        this.messages = [];
        this.systemPrompt = "You are a helpful assistant.";
        this.temperature = 0.7;
        this.topP = 0.9;
        this.topK = 40;
        this.maxTokens = 256;
        this.model = "default";
        this.abortController = null;
        // Max chat messages sent to the server per request (~20 turns).
        // The UI keeps full history; only the prompt is windowed so the
        // server-side prefill doesn't grow unbounded.
        this.maxHistoryMessages = 40;

        this.initElements();
        this.loadState();
        this.initEventListeners();
        this.loadModels();
    }

    static storageKey = 'serius-chat-state';

    initElements() {
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.messagesContainer = document.getElementById('messages');
        this.modelSelect = document.getElementById('model-select');
        this.settingsBtn = document.getElementById('settings-btn');
        this.settingsPanel = document.getElementById('settings-panel');
        this.closeSettingsBtn = document.getElementById('close-settings');
        this.temperatureInput = document.getElementById('temperature');
        this.topPInput = document.getElementById('top_p');
        this.topKInput = document.getElementById('top_k');
        this.maxTokensInput = document.getElementById('max_tokens');
        this.systemPromptInput = document.getElementById('system-prompt');
        this.modelsBtn = document.getElementById('models-btn');
        this.modelsPanel = document.getElementById('models-panel');
        this.closeModelsBtn = document.getElementById('close-models');
        this.modelsList = document.getElementById('models-list');
        this.loadNameInput = document.getElementById('load-name');
        this.loadPathInput = document.getElementById('load-path');
        this.loadTypeSelect = document.getElementById('load-type');
        this.loadQuantInput = document.getElementById('load-quant');
        this.loadStatus = document.getElementById('load-status');
        this.loadModelBtn = document.getElementById('load-model-btn');
    }

    initEventListeners() {
        this.sendBtn.addEventListener('click', () => {
            if (this.abortController) {
                this.abortController.abort();
            } else {
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
        });
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.settingsBtn.addEventListener('click', () => {
            this.settingsPanel.classList.toggle('hidden');
        });

        this.closeSettingsBtn.addEventListener('click', () => {
            this.settingsPanel.classList.add('hidden');
        });

        this.temperatureInput.addEventListener('input', (e) => {
            this.temperature = parseFloat(e.target.value);
            document.getElementById('temp-value').textContent = this.temperature;
            this.saveState();
        });

        this.topPInput.addEventListener('input', (e) => {
            this.topP = parseFloat(e.target.value);
            document.getElementById('topp-value').textContent = this.topP;
            this.saveState();
        });

        this.topKInput.addEventListener('input', (e) => {
            this.topK = parseInt(e.target.value);
            document.getElementById('topk-value').textContent = this.topK;
            this.saveState();
        });

        this.maxTokensInput.addEventListener('input', (e) => {
            this.maxTokens = parseInt(e.target.value);
            document.getElementById('maxtok-value').textContent = this.maxTokens;
            this.saveState();
        });

        this.systemPromptInput.addEventListener('input', (e) => {
            this.systemPrompt = e.target.value;
            this.saveState();
        });

        this.modelSelect.addEventListener('change', (e) => {
            this.model = e.target.value;
            this.saveState();
        });

        this.modelsBtn.addEventListener('click', () => {
            this.modelsPanel.classList.toggle('hidden');
            if (!this.modelsPanel.classList.contains('hidden')) {
                this.refreshModelsPanel();
            }
        });

        this.closeModelsBtn.addEventListener('click', () => {
            this.modelsPanel.classList.add('hidden');
        });

        this.loadModelBtn.addEventListener('click', () => this.loadModel());
    }

    async refreshModelsPanel() {
        await this.renderModelsList();
    }

    async renderModelsList() {
        this.modelsList.innerHTML = '<div class="model-row">Loading…</div>';
        try {
            const response = await fetch('/v1/models');
            const data = await response.json();
            const models = (data.data && Array.isArray(data.data)) ? data.data : [];

            if (models.length === 0) {
                this.modelsList.innerHTML = '<div class="model-row">No models loaded</div>';
                return;
            }

            this.modelsList.innerHTML = '';
            for (const m of models) {
                const row = document.createElement('div');
                row.className = 'model-row';

                const info = document.createElement('span');
                info.className = 'model-info';
                info.textContent = m.id + (m.type ? ` (${m.type})` : '');
                row.appendChild(info);

                const unloadBtn = document.createElement('button');
                unloadBtn.className = 'unload-btn';
                unloadBtn.textContent = 'Unload';
                unloadBtn.addEventListener('click', () => this.unloadModel(m.id));
                row.appendChild(unloadBtn);

                this.modelsList.appendChild(row);
            }
        } catch (error) {
            this.modelsList.innerHTML = '<div class="model-row">Failed to load models</div>';
            console.error('Failed to list models:', error);
        }
    }

    async loadModel() {
        const path = this.loadPathInput.value.trim();
        if (!path) {
            this.loadStatus.textContent = 'Path is required';
            return;
        }

        const payload = {
            path: path,
            type: this.loadTypeSelect.value
        };
        const name = this.loadNameInput.value.trim();
        if (name) payload.name = name;
        const quant = this.loadQuantInput.value.trim();
        if (quant) payload.quant = quant;

        this.loadModelBtn.disabled = true;
        this.loadStatus.textContent = 'Loading… (may take a while for large models)';
        try {
            const response = await fetch('/v1/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                const msg = data.error && (data.error.message || data.error);
                throw new Error(msg || `HTTP ${response.status}`);
            }
            this.loadStatus.textContent = `Loaded: ${data.id || name || path}`;
            this.loadPathInput.value = '';
            this.loadNameInput.value = '';
            this.loadQuantInput.value = '';
            await this.loadModels();
            await this.renderModelsList();
        } catch (error) {
            this.loadStatus.textContent = 'Error: ' + error.message;
            console.error('Model load failed:', error);
        } finally {
            this.loadModelBtn.disabled = false;
        }
    }

    async unloadModel(name) {
        if (!confirm(`Unload model "${name}"?`)) return;
        try {
            const response = await fetch('/v1/models/' + encodeURIComponent(name), {
                method: 'DELETE'
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                const msg = data.error && (data.error.message || data.error);
                throw new Error(msg || `HTTP ${response.status}`);
            }
            if (this.model === name) {
                this.model = 'default';
            }
            await this.loadModels();
            await this.renderModelsList();
        } catch (error) {
            console.error('Model unload failed:', error);
            this.loadStatus.textContent = 'Unload error: ' + error.message;
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/v1/models');
            const data = await response.json();

            if (data.data && Array.isArray(data.data)) {
                this.modelSelect.innerHTML = '';
                data.data.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.id;
                    this.modelSelect.appendChild(option);
                });
                this.model = this.modelSelect.value;
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    escapeHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
                .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Minimal markdown renderer: fenced code blocks, inline code, bold.
    // Input is HTML-escaped first, so output is XSS-safe. Unclosed fences
    // (mid-stream) are tolerated and rendered as a code block.
    renderMarkdown(text) {
        const escaped = this.escapeHtml(text);
        const blocks = [];
        let html = escaped.replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (m, lang, code) => {
            blocks.push(`<pre><code${lang ? ` class="language-${lang}"` : ''}>${code.replace(/\n$/, '')}</code></pre>`);
            return `%%MD_BLOCK_${blocks.length - 1}%%`;
        });
        html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        return html.replace(/%%MD_BLOCK_(\d+)%%/g, (m, i) => blocks[parseInt(i, 10)]);
    }

    saveState() {
        try {
            localStorage.setItem(ChatUI.storageKey, JSON.stringify({
                messages: this.messages,
                settings: {
                    systemPrompt: this.systemPrompt,
                    temperature: this.temperature,
                    topP: this.topP,
                    topK: this.topK,
                    maxTokens: this.maxTokens,
                    model: this.model
                }
            }));
        } catch (e) {
            // storage full or unavailable — non-fatal
        }
    }

    loadState() {
        let state;
        try {
            state = JSON.parse(localStorage.getItem(ChatUI.storageKey) || 'null');
        } catch (e) {
            return;
        }
        if (!state) return;

        if (state.settings) {
            const s = state.settings;
            if (typeof s.systemPrompt === 'string') {
                this.systemPrompt = s.systemPrompt;
                this.systemPromptInput.value = s.systemPrompt;
            }
            if (typeof s.temperature === 'number') {
                this.temperature = s.temperature;
                this.temperatureInput.value = s.temperature;
                document.getElementById('temp-value').textContent = s.temperature;
            }
            if (typeof s.topP === 'number') {
                this.topP = s.topP;
                this.topPInput.value = s.topP;
                document.getElementById('topp-value').textContent = s.topP;
            }
            if (typeof s.topK === 'number') {
                this.topK = s.topK;
                this.topKInput.value = s.topK;
                document.getElementById('topk-value').textContent = s.topK;
            }
            if (typeof s.maxTokens === 'number') {
                this.maxTokens = s.maxTokens;
                this.maxTokensInput.value = s.maxTokens;
                document.getElementById('maxtok-value').textContent = s.maxTokens;
            }
            if (typeof s.model === 'string') this.model = s.model;
        }

        if (Array.isArray(state.messages)) {
            // Restore at most the last 200 messages to bound DOM/localStorage.
            this.messages = state.messages.slice(-200);
            for (const m of this.messages) {
                if (m && typeof m.role === 'string' && typeof m.content === 'string') {
                    this.renderMessage(m.role, m.content);
                }
            }
        }
    }

    // Append a message bubble to the DOM only (no state push).
    renderMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = this.renderMarkdown(content);

        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    addMessage(role, content) {
        this.renderMessage(role, content);
        this.messages.push({ role, content });
        this.saveState();
    }

    appendToLastMessage(content) {
        if (this.messages.length === 0) return;
        const last = this.messages[this.messages.length - 1];
        last.content += content;

        const contentDiv = this.messagesContainer.lastChild &&
            this.messagesContainer.lastChild.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.innerHTML = this.renderMarkdown(last.content);
        }
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    async sendMessage() {
        if (this.abortController) return;  // already streaming
        const content = this.messageInput.value.trim();
        if (!content) return;

        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        this.addMessage('user', content);

        // 'system' entries in this.messages are UI error notices, not real
        // system prompts — keep them out of the transcript sent to the model.
        const history = this.messages
            .filter(m => m.role !== 'system')
            .slice(-this.maxHistoryMessages);
        const messages = [
            { role: 'system', content: this.systemPrompt },
            ...history
        ];

        this.abortController = new AbortController();
        this.sendBtn.textContent = 'Stop';
        this.sendBtn.disabled = false;

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: this.abortController.signal,
                body: JSON.stringify({
                    model: this.model,
                    messages: messages,
                    temperature: this.temperature,
                    top_p: this.topP,
                    top_k: this.topK,
                    max_tokens: this.maxTokens,
                    stream: true
                })
            });

            if (!response.ok) {
                let detail = '';
                try {
                    const errBody = await response.json();
                    if (errBody.error) {
                        detail = typeof errBody.error === 'string'
                            ? errBody.error
                            : (errBody.error.message || JSON.stringify(errBody.error));
                    }
                } catch (_) { /* non-JSON error body */ }
                throw new Error(`HTTP ${response.status}${detail ? ': ' + detail : ''}`);
            }

            this.addMessage('assistant', '');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let sseBuffer = '';

            const handleLine = (rawLine) => {
                const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
                if (!line.startsWith('data: ')) {
                    return false;
                }
                const data = line.slice(6);
                if (data === '[DONE]') {
                    return true;
                }
                try {
                    const json = JSON.parse(data);
                    if (json.error) {
                        const msg = typeof json.error === 'string' ? json.error : (json.error.message || 'Server error');
                        this.addMessage('system', 'Error: ' + msg);
                        return true;
                    }
                    if (json.choices && json.choices[0] && json.choices[0].delta) {
                        const delta = json.choices[0].delta;
                        if (delta.content) {
                            this.appendToLastMessage(delta.content);
                        }
                    }
                } catch (e) {
                    console.error('Failed to parse SSE chunk:', e);
                }
                return false;
            };

            let done = false;
            while (!done) {
                const { done: streamDone, value } = await reader.read();
                if (streamDone) break;

                sseBuffer += decoder.decode(value, { stream: true });
                const lines = sseBuffer.split('\n');
                // Last element may be an incomplete line; keep it buffered.
                sseBuffer = lines.pop();

                for (const line of lines) {
                    if (handleLine(line)) {
                        done = true;
                        break;
                    }
                }
            }

            if (!done && sseBuffer) {
                handleLine(sseBuffer);
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Generation aborted by user');
                // Drop a trailing empty assistant bubble if nothing streamed.
                const last = this.messages[this.messages.length - 1];
                if (last && last.role === 'assistant' && !last.content &&
                    this.messagesContainer.lastChild) {
                    this.messagesContainer.lastChild.remove();
                    this.messages.pop();
                }
            } else {
                console.error('Error sending message:', error);
                this.addMessage('system', 'Error: ' + error.message);
            }
        } finally {
            this.abortController = null;
            this.sendBtn.textContent = 'Send';
            this.saveState();
        }
    }
}

// Initialize the chat UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatUI = new ChatUI();
});
