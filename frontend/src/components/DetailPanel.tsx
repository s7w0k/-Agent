import React from 'react';
import { X, Clock, Cpu, Wrench } from 'lucide-react';
import { MonitorEvent, TimelineStep } from '../types';
import { useChatStore } from '../stores/chatStore';

interface DetailPanelProps {
  isOpen: boolean;
  onClose: () => void;
  events?: MonitorEvent[];
}

const WaterfallChart: React.FC<{ steps: TimelineStep[] }> = ({ steps }) => {
  if (steps.length === 0) return null;

  const maxTime = Math.max(...steps.map(s => s.endTime || s.startTime + 100));

  const getStepColor = (type: string) => {
    switch (type) {
      case 'llm':
        return 'bg-indigo-500';
      case 'tool':
        return 'bg-orange-500';
      default:
        return 'bg-purple-500';
    }
  };

  return (
    <div className="space-y-2">
      {/* 标题 */}
      <div className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-4">
        <Clock size={16} />
        执行时间线
      </div>

      {/* 时间轴 */}
      <div className="relative">
        {/* 时间刻度 */}
        <div className="flex justify-between text-xs text-gray-400 mb-2">
          <span>0ms</span>
          <span>{maxTime}ms</span>
        </div>

        {/* 步骤条 */}
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center gap-3 mb-2">
            <div className="w-32 text-xs text-gray-600 truncate" title={step.name}>
              {step.name}
            </div>
            <div className="flex-1 h-6 bg-gray-100 rounded relative">
              <div
                className={`absolute h-full rounded ${getStepColor(step.type)}`}
                style={{
                  left: `${(step.startTime / maxTime) * 100}%`,
                  width: step.duration 
                    ? `${(step.duration / maxTime) * 100}%` 
                    : '2px',
                }}
                title={`${step.name} - ${step.duration ? `${step.duration}ms` : '运行中'}`}
              />
            </div>
            <div className="w-16 text-xs text-gray-500 text-right">
              {step.duration ? `${step.duration}ms` : '...'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const EventTimeline: React.FC<{ events: MonitorEvent[] }> = ({ events }) => {
  const getIcon = (type: string) => {
    switch (type) {
      case 'start':
        return '🚀';
      case 'step_start':
      case 'step_end':
        return '📍';
      case 'llm_start':
      case 'llm_end':
        return '🤖';
      case 'tool_start':
      case 'tool_end':
        return '🔧';
      case 'complete':
        return '✅';
      case 'error':
        return '❌';
      default:
        return '•';
    }
  };

  const getMessage = (event: MonitorEvent, index: number, allEvents: MonitorEvent[]) => {
    // 计算当前事件的独立计数
    const llmCount = allEvents.slice(0, index + 1).filter(e => e.type === 'llm_start').length;
    const toolCount = allEvents.slice(0, index + 1).filter(e => e.type === 'tool_start').length;
    const stepCount = allEvents.slice(0, index + 1).filter(e => e.type === 'step_start').length;

    switch (event.type) {
      case 'start':
        return '🚀 开始执行';
      case 'step_start':
        return `${stepCount + llmCount + toolCount}. 📍 节点 [${event.node}] 开始`;
      case 'step_end':
        return `✅ 节点 [${event.node}] 完成`;
      case 'llm_start':
        return `${stepCount + llmCount + toolCount}. 🤖 LLM 调用 #${llmCount}`;
      case 'llm_end':
        return event.has_tool_calls
          ? `决定调用工具：${event.tool_names?.join(', ')}`
          : '生成回答';
      case 'tool_start':
        return `${stepCount + llmCount + toolCount}. 🔧 工具 #${toolCount}: ${event.tool_name}`;
      case 'tool_end':
        return `✅ 工具完成：${event.tool_name}`;
      case 'complete':
        return '🎉 执行完成';
      case 'error':
        return `❌ 错误：${event.message}`;
      default:
        return event.type;
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return '';
    }
  };

  return (
    <div className="space-y-3">
      <div className="text-sm font-medium text-gray-700 mb-4">事件详情</div>
      {events.map((event, index) => (
        <div key={index} className="flex items-start gap-3">
          <span className="text-lg">{getIcon(event.type)}</span>
          <div className="flex-1 min-w-0">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-800">{getMessage(event, index, events)}</span>
              <span className="text-xs text-gray-400">{formatTime(event.timestamp)}</span>
            </div>
            {event.input && (
              <div className="mt-1 text-xs bg-gray-50 p-2 rounded break-all">
                输入: {event.input}
              </div>
            )}
            {event.tokens && (
              <div className="mt-1 text-xs bg-indigo-50 p-2 rounded text-indigo-700 inline-block">
                <span className="font-medium">Token:</span> {event.tokens.input_tokens} (输入) + {event.tokens.output_tokens} (输出) = <span className="font-semibold">{event.tokens.total_tokens}</span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export const DetailPanel: React.FC<DetailPanelProps> = ({ isOpen, onClose, events }) => {
  const { timelineSteps, currentEvents, totalTokens } = useChatStore();
  const displayEvents = events || currentEvents;

  // 计算本次对话的 token
  const sessionTokens = {
    input: 0,
    output: 0,
    total: 0,
  };
  displayEvents.forEach(e => {
    if (e.type === 'llm_end' && e.tokens) {
      sessionTokens.input += e.tokens.input_tokens || 0;
      sessionTokens.output += e.tokens.output_tokens || 0;
    }
  });
  sessionTokens.total = sessionTokens.input + sessionTokens.output;

  const stats = {
    total: displayEvents.length,
    llmCalls: displayEvents.filter(e => e.type === 'llm_start').length,
    toolCalls: displayEvents.filter(e => e.type === 'tool_start').length,
  };

  return (
    <>
      {/* 遮罩层 */}
      <div
        className={`fixed inset-0 bg-black/30 z-40 transition-opacity ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* 侧边面板 */}
      <div
        className={`fixed left-0 top-0 h-full w-[420px] bg-white shadow-xl z-50 transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex flex-col h-full">
          {/* 头部 */}
          <div className="flex items-center justify-between px-5 py-4 border-b">
            <h2 className="text-lg font-semibold text-gray-800">执行详情</h2>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X size={20} className="text-gray-500" />
            </button>
          </div>

          {/* 内容 */}
          <div className="flex-1 overflow-y-auto p-5 space-y-6">
            {/* 统计 */}
            <div className="grid grid-cols-4 gap-3">
              <div className="bg-gray-50 rounded-lg p-2 text-center">
                <div className="text-xs text-gray-500 mb-1">总事件</div>
                <div className="text-lg font-semibold text-gray-800">{stats.total}</div>
              </div>
              <div className="bg-indigo-50 rounded-lg p-2 text-center">
                <div className="text-xs text-indigo-600 mb-1 flex items-center justify-center gap-1">
                  <Cpu size={12} /> LLM
                </div>
                <div className="text-lg font-semibold text-indigo-600">{stats.llmCalls}</div>
              </div>
              <div className="bg-orange-50 rounded-lg p-2 text-center">
                <div className="text-xs text-orange-600 mb-1 flex items-center justify-center gap-1">
                  <Wrench size={12} /> 工具
                </div>
                <div className="text-lg font-semibold text-orange-600">{stats.toolCalls}</div>
              </div>
              <div className="bg-purple-50 rounded-lg p-2 text-center">
                <div className="text-xs text-purple-600 mb-1">Token</div>
                <div className="text-lg font-semibold text-purple-600">{sessionTokens.total}</div>
              </div>
            </div>

            {/* 瀑布图 */}
            {timelineSteps.length > 0 && <WaterfallChart steps={timelineSteps} />}

            {/* 事件时间线 */}
            {displayEvents.length > 0 && <EventTimeline events={displayEvents} />}
          </div>
        </div>
      </div>
    </>
  );
};