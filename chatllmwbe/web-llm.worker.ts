import { SendToWorkerMessageEventData } from '@/types/web-llm';

let initialized = false;

globalThis.addEventListener(
  'message',
  async ({ data }: { data: SendToWorkerMessageEventData }) => {
    if (!initialized) {
      globalThis.postMessage({});
      initialized = true;
    }

    // Set the message and conversation index from the worker data
    const message = data.msg || '';
    const curConversationIndex = data.curConversationIndex || 0;

    if (data.ifNewConverstaion) {
      globalThis.postMessage({ type: 'chatting', msg: "Starting a new conversation...", action: 'append' });
      return;
    }

    // Call the backend API instead of running local model inference
    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: message, model: "gpt-4", stream: true })
      });

      if (!response.ok) {
        throw new Error("Failed to get response from backend");
      }

      // Streaming response handling
      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let result = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          result += decoder.decode(value, { stream: true });
          globalThis.postMessage({ type: 'chatting', msg: result, action: 'append' });
        }
      } else {
        const responseText = await response.text();
        globalThis.postMessage({ type: 'chatting', msg: responseText, action: 'append' });
      }
    } catch (error) {
      console.error("Error calling backend:", error);
      globalThis.postMessage({ type: 'error', msg: "Error: Unable to get response from backend", action: 'append' });
    }
  },
  { passive: true },
);
