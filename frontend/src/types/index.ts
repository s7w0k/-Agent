export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  events?: MonitorEvent[];
}

export interface MonitorEvent {
  type: 'start' | 'step_start' | 'step_end' | 'llm_start' | 'llm_end' | 'tool_start' | 'tool_end' | 'complete' | 'error' | 'received' | 'final';
  timestamp: string;
  step?: number;
  node?: string;
  call_number?: number;
  tool_name?: string;
  input?: string;
  output?: string;
  has_tool_calls?: boolean;
  tool_names?: string[];
  tokens?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
  message?: string;
  content?: string;
  llm_calls?: number;
  tool_calls?: number;
  total_steps?: number;
  duration?: number;
}

export interface ExecutionSummary {
  start_time?: string;
  duration?: number;
  llm_calls?: number;
  tool_calls?: number;
  total_steps?: number;
}

export interface TimelineStep {
  id: string;
  name: string;
  type: 'node' | 'llm' | 'tool';
  startTime: number;
  endTime?: number;
  duration?: number;
  status: 'running' | 'success' | 'error';
  details?: string;
}