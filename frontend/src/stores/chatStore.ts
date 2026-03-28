import { create } from 'zustand';
import { Message, MonitorEvent, TimelineStep } from '../types';

interface ChatState {
  messages: Message[];
  currentEvents: MonitorEvent[];
  timelineSteps: TimelineStep[];
  isConnected: boolean;
  isProcessing: boolean;
  currentStep: MonitorEvent | null;
  totalTokens: {
    input: number;
    output: number;
    total: number;
  };
  
  addMessage: (message: Message) => void;
  addEvent: (event: MonitorEvent) => void;
  clearEvents: () => void;
  setConnected: (connected: boolean) => void;
  setProcessing: (processing: boolean) => void;
  setCurrentStep: (step: MonitorEvent | null) => void;
  updateTimeline: (events: MonitorEvent[]) => void;
  calculateTokens: (events: MonitorEvent[]) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  currentEvents: [],
  timelineSteps: [],
  isConnected: false,
  isProcessing: false,
  currentStep: null,
  totalTokens: {
    input: 0,
    output: 0,
    total: 0,
  },

  addMessage: (message) => {
    set((state) => ({
      messages: [...state.messages, message],
    }));
  },

  addEvent: (event) => {
    set((state) => ({
      currentEvents: [...state.currentEvents, event],
    }));
  },

  clearEvents: () => {
    set({ currentEvents: [], currentStep: null });
  },

  setConnected: (connected) => {
    set({ isConnected: connected });
  },

  setProcessing: (processing) => {
    set({ isProcessing: processing });
  },

  setCurrentStep: (step) => {
    set({ currentStep: step });
  },

  updateTimeline: (events) => {
    const steps: TimelineStep[] = [];
    const startTime = events.find(e => e.type === 'start')?.timestamp;
    const baseTime = startTime ? new Date(startTime).getTime() : Date.now();

    // 前端独立计数
    let llmCallCount = 0;
    let toolCallCount = 0;
    let stepCount = 0;

    events.forEach((event, index) => {
      if (event.type === 'step_start' || event.type === 'llm_start' || event.type === 'tool_start') {
        const eventTime = new Date(event.timestamp).getTime();
        const matchingEnd = events.slice(index + 1).find(e => 
          (event.type === 'step_start' && e.type === 'step_end' && e.step === event.step) ||
          (event.type === 'llm_start' && e.type === 'llm_end') ||
          (event.type === 'tool_start' && e.type === 'tool_end' && event.tool_name === e.tool_name)
        );

        let name = '';
        let type: 'node' | 'llm' | 'tool' = 'node';
        let details = '';

        if (event.type === 'step_start') {
          stepCount++;
          name = `${stepCount}. 节点 [${event.node}]`;
          type = 'node';
        } else if (event.type === 'llm_start') {
          llmCallCount++;
          name = `${stepCount + llmCallCount + toolCallCount}. LLM 调用 #${llmCallCount}`;
          type = 'llm';
        } else if (event.type === 'tool_start') {
          toolCallCount++;
          name = `${stepCount + llmCallCount + toolCallCount}. 工具 #${toolCallCount}: ${event.tool_name}`;
          type = 'tool';
          details = event.input || '';
        }

        steps.push({
          id: `${event.type}-${index}`,
          name,
          type,
          startTime: eventTime - baseTime,
          endTime: matchingEnd ? new Date(matchingEnd.timestamp).getTime() - baseTime : undefined,
          duration: matchingEnd 
            ? new Date(matchingEnd.timestamp).getTime() - eventTime 
            : undefined,
          status: matchingEnd ? 'success' : 'running',
          details,
        });
      }
    });

    set({ timelineSteps: steps });
  },

  calculateTokens: (events) => {
    let inputTokens = 0;
    let outputTokens = 0;

    events.forEach(event => {
      if (event.type === 'llm_end' && event.tokens) {
        inputTokens += event.tokens.input_tokens || 0;
        outputTokens += event.tokens.output_tokens || 0;
      }
    });

    set({
      totalTokens: {
        input: inputTokens,
        output: outputTokens,
        total: inputTokens + outputTokens,
      },
    });
  },
}));