import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { User, Bot, Send, FileText, Sparkles, Loader2 } from 'lucide-react';
import { Message } from '../types';
import { useChatStore } from '../stores/chatStore';
import { useWebSocket } from '../hooks/useWebSocket';

interface ChatMessageProps {
  message: Message;
  onShowDetails: (events: Message['events']) => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onShowDetails }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* 头像 */}
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-600'
        }`}
      >
        {isUser ? <User size={20} /> : <Bot size={20} />}
      </div>

      {/* 消息内容 */}
      <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block max-w-[80%] text-left rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-blue-500 text-white rounded-tr-sm'
              : 'bg-gray-100 text-gray-800 rounded-tl-sm'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const isInline = !match;
                    
                    return !isInline ? (
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* 详情按钮 */}
        {!isUser && message.events && message.events.length > 0 && (
          <button
            onClick={() => onShowDetails(message.events)}
            className="mt-2 inline-flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
          >
            <FileText size={14} />
            查看详情
          </button>
        )}
      </div>
    </div>
  );
};

interface ChatBoxProps {
  onShowDetails: (events: Message['events']) => void;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ onShowDetails }) => {
  const [input, setInput] = useState('');
  const [useMultiAgent, setUseMultiAgent] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages, isProcessing, currentStep, addMessage } = useChatStore();
  const { sendMessage } = useWebSocket(`ws://${window.location.host}/ws`);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStep]);

  // 多 Agent API 调用
  const handleMultiAgentSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    setIsLoading(true);
    const userMessage = input.trim();

    // 添加用户消息
    addMessage({
      id: Date.now().toString(),
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
    });
    setInput('');

    try {
      const response = await fetch('/api/multi-agent/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          enable_search: true,
          enable_visualization: true,
          style: 'friendly',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `请求失败 (${response.status})`);
      }

      const data = await response.json();

      // 添加 AI 回复
      let replyContent = data.guide_content || '';

      // 如果有路线图，附加图片信息
      if (data.route_map?.image_url) {
        replyContent += `\n\n---\n🗺️ **路线图已生成**\n\n![路线图](${data.route_map.image_url})`;
      }

      addMessage({
        id: Date.now().toString(),
        role: 'assistant',
        content: replyContent,
        timestamp: new Date(),
      });
    } catch (error) {
      console.error('多 Agent 请求失败:', error);
      addMessage({
        id: Date.now().toString(),
        role: 'assistant',
        content: '抱歉，请求失败了，请稍后重试。',
        timestamp: new Date(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isProcessing) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  const getStepMessage = () => {
    if (!currentStep) return null;
    
    // 计算当前步骤的独立计数
    const llmCount = useChatStore.getState().currentEvents.filter(e => e.type === 'llm_start').length;
    const toolCount = useChatStore.getState().currentEvents.filter(e => e.type === 'tool_start').length;
    
    switch (currentStep.type) {
      case 'step_start':
        return `📍 执行节点：${currentStep.node}`;
      case 'llm_start':
        return `🤖 LLM 思考中... (第 ${llmCount} 次调用)`;
      case 'llm_end':
        if (currentStep.has_tool_calls && currentStep.tool_names) {
          return `💬 决定调用工具：${currentStep.tool_names.join(', ')}`;
        }
        return `💬 LLM 完成，生成回答`;
      case 'tool_start':
        return `🔧 调用工具：${currentStep.tool_name} (第 ${toolCount} 次)`;
      case 'tool_end':
        return `✅ 工具完成：${currentStep.tool_name}`;
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot size={32} className="text-gray-400" />
              </div>
              <h2 className="text-xl font-semibold text-gray-800 mb-2">
                旅行规划助手
              </h2>
              <p className="text-gray-500">
                输入您的问题，开始智能旅行规划
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} onShowDetails={onShowDetails} />
          ))}

          {/* 当前步骤显示 */}
          {isProcessing && currentStep && (
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl px-4 py-3 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-800 truncate">
                    {getStepMessage()}
                  </div>
                  {currentStep.type === 'tool_start' && currentStep.input && (
                    <div className="text-xs text-gray-500 mt-1 truncate">
                      参数：{currentStep.input}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* 输入框 */}
      <div className="border-t bg-white px-6 py-4">
        {/* 模式切换 */}
        <div className="max-w-4xl mx-auto mb-3 flex items-center gap-2">
          <button
            onClick={() => setUseMultiAgent(false)}
            className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
              !useMultiAgent ? 'bg-blue-100 text-blue-700' : 'text-gray-500 hover:bg-gray-100'
            }`}
            disabled={isProcessing}
          >
            🤖 标准模式
          </button>
          <button
            onClick={() => setUseMultiAgent(true)}
            className={`px-3 py-1.5 text-sm rounded-lg transition-colors flex items-center gap-1.5 ${
              useMultiAgent ? 'bg-purple-100 text-purple-700' : 'text-gray-500 hover:bg-gray-100'
            }`}
            disabled={isProcessing}
          >
            <Sparkles size={14} />
            多 Agent 模式
            {useMultiAgent && <span className="text-xs">(含路线图)</span>}
          </button>
        </div>

        <form onSubmit={useMultiAgent ? handleMultiAgentSubmit : handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={useMultiAgent ? "输入旅行需求，如：帮我规划北京到苏州三日游..." : "输入您的问题..."}
              disabled={isProcessing || isLoading}
              className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!input.trim() || isProcessing || isLoading}
              className={`px-6 py-3 rounded-xl transition-colors flex items-center gap-2 ${
                useMultiAgent
                  ? 'bg-purple-500 text-white hover:bg-purple-600'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              {isLoading ? '生成中...' : '发送'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};