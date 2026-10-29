// Serius Chat UI - Client-side logic

class ChatUI {
    constructor() {
        this.messages = [];
        this.systemPrompt = "You are a helpful assistant.";
        this.temperature = 0.7;
        this.topP = 0.9;
        this.maxTokens = 256;
        this.model = "default";

        this.initElements();
        this.initEventListeners();
        this.loadModels();
    }

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
        this.maxTokensInput = document.getElementById('max_tokens');
        this.systemPromptInput = document.getElementById('system-prompt');
    }

    initEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
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
        });

        this.topPInput.addEventListener('input', (e) => {
            this.topP = parseFloat(e.target.value);
            document.getElementById('topp-value').textContent = this.topP;
        });

        this.maxTokensInput.addEventListener('input', (e) => {
            this.maxTokens = parseInt(e.target.value);
            document.getElementById('maxtok-value').textContent = this.maxTokens;
        });

        this.systemPromptInput.addEventListener('input', (e) => {
            this.systemPrompt = e.target.value;
        });

        this.modelSelect.addEventListener('change', (e) => {
            this.model = e.target.value;
        });
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

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

        this.messages.push({ role, content });
    }

    appendToLastMessage(content) {
        if (this.messagesContainer.lastChild) {
            const contentDiv = this.messagesContainer.lastChild.querySelector('.message-content');
            contentDiv.textContent += content;
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

            if (this.messages.length > 0) {
                this.messages[this.messages.length - 1].content += content;
            }
        }
    }

    async sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content) return;

        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.sendBtn.disabled = true;

        this.addMessage('user', content);

        const messages = [
            { role: 'system', content: this.systemPrompt },
            ...this.messages
        ];

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.model,
                    messages: messages,
                    temperature: this.temperature,
                    top_p: this.topP,
                    max_tokens: this.maxTokens,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.addMessage('assistant', '');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            return;
                        }

                        try {
                            const json = JSON.parse(data);
                            if (json.choices && json.choices[0] && json.choices[0].delta) {
                                const delta = json.choices[0].delta;
                                if (delta.content) {
                                    this.appendToLastMessage(delta.content);
                                }
                            }
                        } catch (e) {
                            console.error('Failed to parse SSE chunk:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('system', 'Error: Failed to get response from server');
        } finally {
            this.sendBtn.disabled = false;
        }
    }
}

// Initialize the chat UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatUI = new ChatUI();
});
