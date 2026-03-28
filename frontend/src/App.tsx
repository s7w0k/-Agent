import React, { useState } from 'react';
import { ChatBox } from './components/ChatBox';
import { DetailPanel } from './components/DetailPanel';
import { useChatStore } from './stores/chatStore';
import { MonitorEvent } from './types';
import { Bot, Wifi, WifiOff } from 'lucide-react';

function App() {
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [selectedEvents, setSelectedEvents] = useState<MonitorEvent[] | undefined>();
  const { isConnected, currentEvents, totalTokens } = useChatStore();

  const handleShowDetails = (events: MonitorEvent[] | undefined) => {
    setSelectedEvents(events);
    setIsPanelOpen(true);
  };

  const handleClosePanel = () => {
    setIsPanelOpen(false);
  };

  return (
    <div className="h-screen flex flex-col bg-white">
      {/* 详情面板 */}
      <DetailPanel
        isOpen={isPanelOpen}
        onClose={handleClosePanel}
        events={selectedEvents}
      />

      {/* 顶部栏 */}
      <header className="flex-shrink-0 border-b bg-white">
        <div className="max-w-5xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white">
              <Bot size={24} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-800">旅行规划助手</h1>
              <p className="text-xs text-gray-500">AI 驱动的智能旅行规划</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Token 统计 */}
            {totalTokens.total > 0 && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-50 rounded-full text-sm text-purple-700">
                <span className="font-medium">Token: {totalTokens.total}</span>
                <span className="text-purple-400">|</span>
                <span className="text-xs">输入：{totalTokens.input}</span>
                <span className="text-purple-400">|</span>
                <span className="text-xs">输出：{totalTokens.output}</span>
              </div>
            )}
            
            <div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm ${
                isConnected
                  ? 'bg-green-50 text-green-700'
                  : 'bg-red-50 text-red-700'
              }`}
            >
              {isConnected ? (
                <>
                  <Wifi size={14} />
                  <span>已连接</span>
                </>
              ) : (
                <>
                  <WifiOff size={14} />
                  <span>未连接</span>
                </>
              )}
            </div>
            
            {currentEvents.length > 0 && (
              <button
                onClick={() => handleShowDetails(currentEvents)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-sm text-gray-700 transition-colors"
              >
                <span className="w-4 h-4 bg-blue-500 rounded-full text-white text-xs flex items-center justify-center">
                  {currentEvents.length}
                </span>
                <span>详情</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* 主内容 */}
      <main className="flex-1 min-h-0">
        <ChatBox onShowDetails={handleShowDetails} />
      </main>
    </div>
  );
}

export default App;