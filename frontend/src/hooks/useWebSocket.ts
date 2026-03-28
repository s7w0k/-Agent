import { useEffect, useRef } from 'react';
import { useChatStore } from '../stores/chatStore';
import { MonitorEvent } from '../types';

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const store = useChatStore();

  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const connect = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) return;

      console.log('Connecting to WebSocket:', url);
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('WebSocket connected');
        store.setConnected(true);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        store.setConnected(false);
        wsRef.current = null;
        // 只在非正常关闭时重连
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // 错误时不触发重连，让 onclose 处理
      };

      ws.onmessage = (event) => {
        try {
          const data: MonitorEvent = JSON.parse(event.data);
          console.log('Received event:', data.type);

          store.addEvent(data);
          store.updateTimeline([...store.currentEvents, data]);
          store.calculateTokens([...store.currentEvents, data]);

          switch (data.type) {
            case 'received':
              store.setProcessing(true);
              store.clearEvents();
              break;

            case 'step_start':
            case 'llm_start':
            case 'tool_start':
              store.setCurrentStep(data);
              break;

            case 'step_end':
            case 'llm_end':
            case 'tool_end':
              break;

            case 'final':
              if (data.content) {
                store.addMessage({
                  id: Date.now().toString(),
                  role: 'assistant',
                  content: data.content,
                  timestamp: new Date(),
                  events: [...store.currentEvents],
                });
              }
              store.setProcessing(false);
              store.setCurrentStep(null);
              break;

            case 'error':
              console.error('Error:', data.message);
              store.setProcessing(false);
              store.setCurrentStep(null);
              break;
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      wsRef.current = ws;
    };

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url]);

  const sendMessage = (message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      store.addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: message,
        timestamp: new Date(),
      });
      
      wsRef.current.send(message);
      store.setProcessing(true);
    } else {
      console.error('WebSocket not connected');
    }
  };

  return { sendMessage };
}