import { useMemo, useState } from 'react';
import { chatWithAgent } from '../services/api';

const starterMessages = [
  {
    role: 'assistant',
    text: 'Hi, I am the GaitML Research Assistant. Ask me questions like: who is the best model for flatfoot, who is the second best model for autism, or compare models for young-old.',
  },
];

const quickPrompts = [
  'Who is the best model for flatfoot?',
  'Second best for autism',
  'Compare models for young-old',
];

function localFallback(message) {
  const text = message.toLowerCase();
  if (text.includes('shap') || text.includes('feature')) {
    return 'For SHAP or permutation importance, start with the top-ranked features, then explain their biomechanical meaning. Stable features across several models are stronger evidence.';
  }
  if (text.includes('kaggle') || text.includes('gpu') || text.includes('run')) {
    return 'For Kaggle, copy the project from /kaggle/input to /kaggle/working first, then run the pipeline phases from the project root.';
  }
  return 'I need the backend agent running to answer ranked model questions from real results. Start the FastAPI backend, then ask again.';
}

export default function GaitResearchAssistant() {
  const [isOpen, setIsOpen] = useState(true);
  const [messages, setMessages] = useState(starterMessages);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);

  const statusText = useMemo(() => (isThinking ? 'reading result files' : 'GaitML agent ready'), [isThinking]);

  async function sendMessage(value = input) {
    const trimmed = value.trim();
    if (!trimmed || isThinking) return;

    setMessages((current) => [...current, { role: 'user', text: trimmed }]);
    setInput('');
    setIsThinking(true);

    try {
      const response = await chatWithAgent(trimmed);
      setMessages((current) => [...current, { role: 'assistant', text: response.answer }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', text: localFallback(trimmed) },
      ]);
    } finally {
      setIsThinking(false);
    }
  }

  if (!isOpen) {
    return (
      <button
        className="assistant-launcher"
        type="button"
        onClick={() => setIsOpen(true)}
        aria-label="Open GaitML Research Assistant"
      >
        <span>AI</span>
      </button>
    );
  }

  return (
    <section className="research-assistant" aria-label="GaitML Research Assistant">
      <header className="assistant-header">
        <div className="assistant-brand">
          <div className="assistant-logo">AI</div>
          <div>
            <h2>GaitML Research Assistant</h2>
            <p><span className="assistant-status-dot" />{statusText}</p>
          </div>
        </div>
        <div className="assistant-actions">
          <button type="button" title="Minimize assistant" onClick={() => setIsOpen(false)}>
            -
          </button>
          <button type="button" title="Clear chat" onClick={() => setMessages(starterMessages)}>
            x
          </button>
        </div>
      </header>

      <div className="assistant-toolbar" aria-label="Research shortcuts">
        {quickPrompts.map((prompt) => (
          <button key={prompt} type="button" onClick={() => sendMessage(prompt)} disabled={isThinking}>
            {prompt}
          </button>
        ))}
      </div>

      <div className="assistant-messages">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`assistant-message ${message.role}`}>
            {message.text}
          </div>
        ))}
        {isThinking && (
          <div className="assistant-message assistant">Checking the ranked model results...</div>
        )}
      </div>

      <form
        className="assistant-input-row"
        onSubmit={(event) => {
          event.preventDefault();
          sendMessage();
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask: who is the second best model for autism?"
          aria-label="Message GaitML Research Assistant"
          disabled={isThinking}
        />
        <button type="submit" aria-label="Send message" disabled={isThinking}>Send</button>
      </form>
    </section>
  );
}