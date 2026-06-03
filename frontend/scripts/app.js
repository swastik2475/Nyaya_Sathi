const API_URL = 'http://127.0.0.1:8000/query';
let sessionId = crypto.randomUUID();
let isLoading = false;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('userInput').focus();
  renderHistory();
});

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function fillInput(text) {
  const input = document.getElementById('userInput');
  input.value = text;
  autoResize(input);
  input.focus();
}

function toggleTag(el) {
  el.classList.toggle('on');
}

function newChat() {
  sessionId = crypto.randomUUID();
  document.getElementById('messages').innerHTML = '';
  document.getElementById('emptyState').style.display = '';
}

function now() {
  return new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
}

function appendMsg(role, text, sources) {
  document.getElementById('emptyState').style.display = 'none';
  const wrap = document.getElementById('messages');

  const msgEl = document.createElement('div');
  msgEl.className = `msg ${role}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'msg-avatar';
  avatarEl.textContent = role === 'user' ? 'U' : '⚖';

  const bodyEl = document.createElement('div');
  bodyEl.className = 'msg-body';

  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'msg-bubble';
  bubbleEl.textContent = text;
  bodyEl.appendChild(bubbleEl);

  if (role === 'ai' && sources && sources.length > 0) {
    const srcWrap = document.createElement('div');
    srcWrap.className = 'sources';
    sources.forEach((s) => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.textContent = s;
      srcWrap.appendChild(chip);
    });
    bodyEl.appendChild(srcWrap);
  }

  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';
  const timeEl = document.createElement('span');
  timeEl.className = 'msg-time';
  timeEl.textContent = now();
  metaEl.appendChild(timeEl);

  if (role === 'ai') {
    const fb = document.createElement('div');
    fb.className = 'feedback';
    fb.innerHTML = `<button class="fb-btn up" onclick="this.style.color='var(--green)';this.style.background='var(--green-dim)';">👍 Helpful</button><button class="fb-btn down" onclick="this.style.color='var(--red)';this.style.background='var(--red-dim)';">👎 Not helpful</button>`;
    bodyEl.appendChild(metaEl);
    bodyEl.appendChild(fb);
  } else {
    bodyEl.appendChild(metaEl);
  }

  msgEl.appendChild(avatarEl);
  msgEl.appendChild(bodyEl);
  wrap.appendChild(msgEl);

  const area = document.getElementById('chatArea');
  area.scrollTop = area.scrollHeight;
  return msgEl;
}

function showTyping() {
  const wrap = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg ai';
  el.id = 'typing';
  el.innerHTML = `
    <div class="msg-avatar">⚖</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  wrap.appendChild(el);
  const area = document.getElementById('chatArea');
  area.scrollTop = area.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function sendMessage() {
  if (isLoading) return;
  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  isLoading = true;
  document.getElementById('sendBtn').disabled = true;

  appendMsg('user', text, []);
  showTyping();

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question: text }),
    });

    removeTyping();

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    appendMsg('ai', data.answer, data.sources || []);
  } catch (err) {
    removeTyping();
    appendMsg(
      'ai',
      '⚠️ Could not connect to NyayaSathi backend. Make sure it\'s running:\n\nuvicorn main:app --reload --port 8000',
      []
    );
  }

  isLoading = false;
  document.getElementById('sendBtn').disabled = false;
  input.focus();
}

function renderHistory() {
  const historyItems = [
    { text: 'How to file an FIR online in India?', time: 'now', active: true },
    { text: 'Rights under Article 21 of the Constitution', time: '2h' },
    { text: 'Tele-Law scheme eligibility for rural women', time: '1d' },
    { text: 'IPC Section 498A — domestic violence', time: '3d' },
  ];
  const container = document.getElementById('historyList');
  container.innerHTML = '';
  historyItems.forEach((item) => {
    const row = document.createElement('div');
    row.className = `chat-item${item.active ? ' active' : ''}`;
    row.innerHTML = `<div class="chat-item-icon">💬</div><div style="flex:1;min-width:0"><div class="chat-item-text">${item.text}</div></div><div class="chat-item-ts">${item.time}</div>`;
    container.appendChild(row);
  });
}
