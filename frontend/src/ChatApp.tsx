/* =============================================================================
   RAG Chat Application
============================================================================= */

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import toast, { Toaster } from 'react-hot-toast';
import { 
  Send, StopCircle, Loader2, BotMessageSquare, User, MessageCircle, 
  MessageSquareQuote, Library, PanelLeftClose, PanelLeftOpen,
  PanelRightClose, PanelRightOpen, Trash2, SlidersHorizontal,
  GripVertical, FileText
} from 'lucide-react';

// ============================================ Data Model ============================================
interface Reference { 
  ref_id: string; 
  title: string; 
  author: string; 
  page: number; 
  line: number; 
  content: string; 
  score: number;
}
interface Message { 
  role: 'user' | 'assistant'; 
  content: string; 
  references: Reference[]; 
}
interface ChatSettings { 
  llmProvider: 'ollama' | 'gemini'; 
  ragMode: boolean; 
  topK: number; 
  threshold: number; 
}

// ============================================ Global ============================================
const API_BASE = '/api';
const MIN_SIDEBAR_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 600;
const DEFAULT_SIDEBAR_WIDTH = 320;

// ============================================ Left Sidebar ============================================

/**
 * Left Sidebar. Mobile overlay component for mobile devices.
 * @param isOpen - Whether the overlay is open.
 * @param onToggle - Function to toggle the overlay.
 * @returns 
 */
function SettingsMobileOverlay({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  if (!isOpen) return null;
  return (
    <div 
      className="fixed inset-0 bg-black/60 z-40 lg:hidden"
      onClick={onToggle}
    />
  );
}

/**
 * Left Sidebar. Settings panel header.
 */
function SettingsHeader({ onToggle }: { onToggle: () => void }) {
  return (
    <div className="h-14 border-b border-neutral-800 flex items-center justify-between px-4">
      <div className="flex items-center gap-2 text-neutral-300">
        <SlidersHorizontal size={16} />
        <span className="font-medium text-sm">Settings</span>
      </div>
      <button
        onClick={onToggle}
        className="p-1.5 hover:bg-neutral-800 rounded-lg transition-colors text-neutral-400 hover:text-white"
      >
        <PanelLeftClose size={18} />
      </button>
    </div>
  );
}

/**
 * Left Sidebar. RAG toggle button.
 */
function SettingsRagToggleButton({ settings, onSettingsChange }: { settings: ChatSettings; onSettingsChange: (settings: ChatSettings) => void }) {
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">Mode</h4>
      <div className="flex items-center justify-between p-3 bg-neutral-900 rounded-xl border border-neutral-800">
        <div className="flex items-center gap-2">
          <span className="text-sm text-neutral-300">RAG Mode</span>
        </div>
        <button
          onClick={() => onSettingsChange({ ...settings, ragMode: !settings.ragMode })}
          className={`w-11 h-6 rounded-full relative transition-colors ${
            settings.ragMode ? "bg-blue-600" : "bg-neutral-700"
          }`}
        >
          <div className={`w-4 h-4 bg-white rounded-full absolute top-1 transition-all ${
            settings.ragMode ? "left-6" : "left-1"
          }`} />
        </button>
      </div>
    </div>
  );
}

/**
 * Left Sidebar. RAG settings.
 */
function SettingsRagSettings({ settings, onSettingsChange }: { settings: ChatSettings; onSettingsChange: (settings: ChatSettings) => void }) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {/* RAG Mode Toggle */}
      <SettingsRagToggleButton settings={settings} onSettingsChange={onSettingsChange} />
      {/* RAG Settings */}
      <div className={`space-y-4 transition-opacity duration-200 ${
        !settings.ragMode ? "opacity-40 pointer-events-none" : ""
      }`}>
        <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">RAG Parameters</h4>
        
        {/* Top K Slider */}
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800 space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-neutral-300">Top K Results</span>
            <span className="text-sm font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
              {settings.topK}
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            value={settings.topK}
            onChange={(e) => onSettingsChange({ ...settings, topK: Number(e.target.value) })}
            className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <p className="text-[11px] text-neutral-500">Number of documents to retrieve</p>
        </div>

        {/* Threshold Slider */}
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800 space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-neutral-300">Similarity Threshold</span>
            <span className="text-sm font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
              {settings.threshold.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={settings.threshold}
            onChange={(e) => onSettingsChange({ ...settings, threshold: Number(e.target.value) })}
            className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <p className="text-[11px] text-neutral-500">Minimum relevance score for results</p>
        </div>
      </div>

      {/* LLM Provider */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">LLM Provider</h4>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <select
            value={settings.llmProvider}
            onChange={(e) => onSettingsChange({ ...settings, llmProvider: e.target.value as 'ollama' | 'gemini' })}
            className="w-full bg-transparent text-sm text-neutral-300 focus:outline-none cursor-pointer"
          >
            <option value="gemini" className="bg-neutral-900">Gemini</option>
            <option value="ollama" className="bg-neutral-900">Ollama</option>
          </select>
        </div>
      </div>

      {/* Source document links */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">RAG source documents</h4>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <a
            href="https://www.cs.utexas.edu/~rossbach/cs380p/papers/ulk3.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-between text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Understanding the LINUX KERNEL</span>
          </a>
        </div>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <a
            href="https://ftp.utcluj.ro/pub/users/civan/CPD/3.RESURSE/6.Book_2012_Distributed%20systems%20_Couloris.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-between text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Distributed Systems: Concepts and Design</span>
          </a>
        </div>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <a
            href="https://eclass.uoa.gr/modules/document/file.php/D245/2015/DistrComp.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-between text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Distributed Computing Principles, Algorithms, and Systems</span>
          </a>
        </div>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <a
            href="https://people.uncw.edu/huberr/constructivism.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-between text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Constructivism in Practice and Theory: Toward a Better Understanding</span>
          </a>
        </div>
        <div className="p-3 bg-neutral-900 rounded-xl border border-neutral-800">
          <a
            href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://webfiles.ucpress.edu/oa/9780520395770_WEB.pdf&ved=2ahUKEwiMsZDgx6yRAxVQDjQIHbfDL3sQFnoECB0QAQ&usg=AOvVaw0XkmPFJe82OvSPHN2bU07E"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-between text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Globalization: Past, Present, Future</span>
          </a>
        </div>
      </div>
    </div>
  );
}

/**
 * Left Sidebar. Settings footer.
 */
function ClearChatButton({ onClear }: { onClear: () => void }) {
  return (
    <div className="p-4 border-t border-neutral-800">
      <button
        onClick={onClear}
        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-xl text-red-400 text-sm font-medium transition-colors"
      >
        <Trash2 size={16} />
        Clear Chat
      </button>
    </div>
  );
}

/**
 * Left Sidebar - Settings Panel
 */
function SettingsSidebar({ 
  isOpen, 
  onToggle, 
  settings, 
  onSettingsChange, 
  onClear 
}: { 
  isOpen: boolean;
  onToggle: () => void;
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
  onClear: () => void;
}) {
  return (
    <>
      {/* Mobile Overlay */}
      <SettingsMobileOverlay isOpen={isOpen} onToggle={onToggle} />
      {/* Sidebar */}
      <aside className={`
        fixed lg:relative inset-y-0 left-0 z-50
        w-72 bg-neutral-950 border-r border-neutral-800
        transform transition-all duration-300 ease-out
        ${isOpen ? 'translate-x-0 lg:w-72' : '-translate-x-full lg:translate-x-0 lg:w-0 lg:border-r-0'}
        flex flex-col shrink-0 overflow-hidden
      `}>
        {/* Settings Header */}
        <SettingsHeader onToggle={onToggle} />
        {/* Settings Content */}
        <SettingsRagSettings settings={settings} onSettingsChange={onSettingsChange} />
        {/* Clear Chat */}
        <ClearChatButton onClear={onClear} />
      </aside>
    </>
  );
}

// ============================================ Right Sidebar ============================================

/**
 * Right Sidebar. Mobile overlay component for mobile devices.
 */
function ReferencesMobileOverlay({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  if (!isOpen) return null;
  return (
    <div 
      className="fixed inset-0 bg-black/60 z-40 lg:hidden"
      onClick={onToggle}
    />
  );
}

/**
 * Right Sidebar. Resize handle for resizing the sidebar.
 */
function ReferencesResizeHandle({ 
  isOpen, 
  onMouseDown 
}: { 
  isOpen: boolean; 
  onMouseDown: (e: React.MouseEvent) => void;
}) {
  if (!isOpen) return null;
  return (
    <div
      onMouseDown={onMouseDown}
      className="absolute left-0 top-0 bottom-0 w-2 cursor-col-resize hover:bg-blue-500/50 transition-colors group z-10 hidden lg:flex items-center"
    >
      <div className="absolute left-0 w-6 h-20 flex items-center justify-center -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
        <GripVertical size={14} className="text-neutral-500" />
      </div>
    </div>
  );
}

/**
 * Right Sidebar. Header component.
 */
function ReferencesHeader({ 
  onToggle,
  referenceCount 
}: { 
  onToggle: () => void; 
  referenceCount: number;
}) {
  return (
    <div className="h-14 border-b border-neutral-800 flex items-center justify-between px-4 shrink-0">
      <button
        onClick={onToggle}
        className="p-1.5 hover:bg-neutral-800 rounded-lg transition-colors text-neutral-400 hover:text-white"
      >
        <PanelRightClose size={18} />
      </button> 
      <div className="flex items-center gap-2 text-neutral-300">
        <MessageSquareQuote size={16} />
        <span className="font-medium text-sm">References</span>
        {referenceCount > 0 && (
          <span className="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 text-[10px] font-semibold rounded-full">
            {referenceCount}
          </span>
        )}
      </div> 
    </div>
  );
}

/**
 * Right Sidebar. Empty state when no references.
 */
function ReferencesEmptyState({ hasSelection }: { hasSelection: boolean }) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-neutral-500">
      <MessageSquareQuote size={32} className="mb-3 opacity-50" />
      <p className="text-sm">{hasSelection ? "No references for this response" : "No response selected"}</p>
      <p className="text-xs text-neutral-600 mt-1">
        {hasSelection ? "This response has no RAG references" : "Click on a bot response to view its references"}
      </p>
    </div>
  );
}

/**
 * Right Sidebar. Reference card.
 */
function ReferencesCard({ reference }: { reference: Reference }) {
  return (
    <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-xl space-y-3 hover:border-neutral-700 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 text-xs font-bold rounded">
            [{reference.ref_id}]
          </span>
          <span className="text-sm font-medium text-neutral-200 line-clamp-1">
            {reference.title}
          </span>
        </div>
        <span className="shrink-0 px-2 py-0.5 bg-emerald-500/10 text-emerald-400 text-[10px] font-mono rounded">
          Similarity: {Math.round(reference.score * 100)}%
        </span>
      </div>

      {/* Metadata */}
      <div className="text-xs text-neutral-500">
        by <span className="text-neutral-400">{reference.author}</span>
        <span className="mx-1.5">•</span>
        Page {reference.page}, Line {reference.line}
      </div>

      {/* Full Content */}
      <div className="p-3 bg-neutral-950 border-l-2 border-blue-500/30 rounded-r-lg">
        <p className="text-xs text-neutral-400 leading-relaxed">
          "{reference.content}"
        </p>
      </div>
    </div>
  );
}

/**
 * Right Sidebar. References list content.
 */
function ReferencesContent({ references, hasSelection }: { references: Reference[]; hasSelection: boolean }) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-3">
      {references.length === 0 ? (
        <ReferencesEmptyState hasSelection={hasSelection} />
      ) : (
        references.map((ref, idx) => (
          <ReferencesCard key={idx} reference={ref} />
        ))
      )}
    </div>
  );
}

/**
 * Right Sidebar. Footer with width indicator.
 */
function ReferencesFooter({ width }: { width: number }) {
  return (
    <div className="px-4 py-2 border-t border-neutral-800 text-[10px] text-neutral-600 text-center shrink-0 hidden lg:block">
      Drag left edge to resize • {width}px
    </div>
  );
}

/**
 * Right Sidebar. RAG References Panel
 */
function ReferencesSidebar({ 
  isOpen, 
  onToggle, 
  references,
  width,
  onWidthChange,
  hasSelection
}: { 
  isOpen: boolean;
  onToggle: () => void;
  references: Reference[];
  width: number;
  onWidthChange: (width: number) => void;
  hasSelection: boolean;
}) {
  // References
  const sidebarRef = useRef<HTMLElement>(null);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(width);
  // Resize Handle
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [width]);
  // Resize Handle Mouse Move
  useEffect(() => {
    // Resize Handle Mouse Move
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = startX.current - e.clientX;
      const newWidth = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, startWidth.current + delta));
      onWidthChange(newWidth);
    };
    // Resize Handle Mouse Up
    const handleMouseUp = () => {
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    // Add listener for resize handle mouse move and up
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    // Remove listener for resize handle mouse move and up on unmount
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [onWidthChange]);

  return (
    <>
      {/* Mobile Overlay */}
      <ReferencesMobileOverlay isOpen={isOpen} onToggle={onToggle} />
      
      {/* Sidebar */}
      <aside 
        ref={sidebarRef}
        style={{ width: isOpen ? `${width}px` : '0px' }}
        className={`
          fixed lg:relative inset-y-0 right-0 z-50
          bg-neutral-950 border-l border-neutral-800
          transform transition-all duration-300 ease-out
          ${isOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0 lg:border-l-0'}
          flex flex-col shrink-0 overflow-hidden
        `}
      >
        {/* Resize Handle */}
        <ReferencesResizeHandle isOpen={isOpen} onMouseDown={handleMouseDown} />
        {/* Header */}
        <ReferencesHeader onToggle={onToggle} referenceCount={references.length} />
        {/* References Content */}
        <ReferencesContent references={references} hasSelection={hasSelection} />
        {/* Footer */}
        <ReferencesFooter width={width} />
      </aside>
    </>
  );
}

// ============================================ Main Chat ============================================

/**
 * Main chat. LLM bot profile icon component.
 */
function MainChatBotProfileIcon() {
  return (
    <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center mr-3 mt-1 shrink-0">
      <BotMessageSquare size={18} />
    </div>
  );
}

/**
 * Main chat. User profile icon component.
 */
function MainChatUserProfileIcon() {
  return (
    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white ml-3 mt-1 shrink-0">
      <User size={16} />
    </div>
  );
}

/**
 * Main chat. Message bubble component for chat interface.
 */
function MainChatMessageBubble({ 
  isUser, 
  message, 
  isSelected,
  onClick 
}: { 
  isUser: boolean; 
  message: Message; 
  isSelected?: boolean;
  onClick?: (e: React.MouseEvent) => void;
}) {
  return (
    <div 
      onClick={onClick}
      className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
        isUser
          ? "bg-blue-600 text-white rounded-br-sm"
          : `bg-neutral-900 border text-neutral-200 rounded-bl-sm transition-all cursor-pointer ${
              isSelected 
                ? "border-blue-500 ring-2 ring-blue-500/30" 
                : "border-neutral-800 hover:border-neutral-700"
            }`
      }`}
    >
      {isUser ? (
        <div className="whitespace-pre-wrap">{message.content}</div>
      ) : (
        <div className="prose prose-invert prose-sm max-w-none prose-p:my-1.5 prose-pre:bg-neutral-950 prose-pre:border prose-pre:border-neutral-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {/* Remove leading/trailing whitespace and highlight references in brackets */}
            {message.content.trim().replace(/\[([\d,\s]+)\]/g, '<span class="text-blue-400 font-bold mx-0.5">[$1]</span>')}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}

/**
 * Main chat. Message history item component.
 */
function MainChatMessageHistory({ 
  message, 
  isSelected, 
  onSelect 
}: { 
  message: Message; 
  isSelected: boolean;
  onSelect: (message: Message | null) => void;
}) {
  const isUser = message.role === 'user';
  
  const handleBubbleClick = (e: React.MouseEvent) => {
    if (!isUser) {
      e.stopPropagation();
      onSelect(isSelected ? null : message);
    }
  };
  
  return (
    <div className={`flex w-full mb-5 animate-in slide-in-from-bottom-2 duration-200 ${
      isUser ? "justify-end" : "justify-start"
    }`}>
      {!isUser && <MainChatBotProfileIcon />} 
      <div className="max-w-[85%] lg:max-w-[75%]">
        <MainChatMessageBubble 
          isUser={isUser} 
          message={message} 
          isSelected={isSelected}
          onClick={!isUser ? handleBubbleClick : undefined}
        />
      </div>
      {isUser && <MainChatUserProfileIcon />}
    </div>
  );
}

/**
 * Main chat. Chat application.
 */
export default function ChatApp() {
  // State
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatSettings, setChatSettings] = useState<ChatSettings>({
    llmProvider: 'ollama',
    ragMode: false,
    topK: 5,
    threshold: 0.6,
  });
  const [userInput, setUserInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const [rightSidebarWidth, setRightSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [allReferences, setAllReferences] = useState<Reference[]>([]);
  const [documentCount, setDocumentCount] = useState<number | null>(null);
  const [selectedResponse, setSelectedResponse] = useState<Message | null>(null);
  // Fetch total document count on component mount
  useEffect(() => {
    async function fetchDocCount() {
      try {
        const res = await fetch(`${API_BASE}/document/count`);
        const data = await res.json();
        setDocumentCount(data.count);
      } catch (err) {
        console.error("Failed to fetch document count:", err);
      }
    }
    fetchDocCount();
  }, []);
  const abortController = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isStreaming]);

  // Collect all references from messages (for badge count in header)
  useEffect(() => {
    const refs = chatMessages
      .filter(m => m.role === 'assistant')
      .flatMap(m => m.references);
    setAllReferences(refs);
  }, [chatMessages]);

  // Get references to display in sidebar (only from selected response)
  const displayedReferences = selectedResponse?.references ?? [];

  // Handle click outside to deselect
  const handleMainAreaClick = useCallback(() => {
    setSelectedResponse(null);
  }, []);

  // Send handler
  const handleSend = async () => {
    if (!userInput.trim() || isStreaming) return;
    const prompt = userInput.trim();
    setUserInput('');
    const userMessage: Message = { role: 'user', content: prompt, references: [] };
    const accumulatedMessages = [...chatMessages, userMessage];
    setChatMessages(accumulatedMessages);
    setIsStreaming(true);
    abortController.current = new AbortController();
    
    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: accumulatedMessages,
          provider: chatSettings.llmProvider,
          rag_mode: chatSettings.ragMode,
          topk: chatSettings.topK,
          threshold: chatSettings.threshold,
        }),
        signal: abortController.current.signal,
      });
      
      if (!res.ok) throw new Error(res.statusText);
      
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let assistantMessage: Message = { role: 'assistant', content: '', references: [] };
      let pendingReferences: Reference[] = [];
      let buffer = '';
      
      /**
       * Parse buffer and process references.
       * @param buf - Buffer to parse
       * @returns Remaining buffer after processing
       */
      const parseBuffer = (buf: string): string => {
        const lines = buf.split('\n\n');
        buf = lines.pop() || '';  // Keep last partial line for next iteration
        // Process lines
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const [event, data] = line.split('\n');
            const eventType = event.slice(7);
            const payload = JSON.parse(data.slice(6));
            // Process references
            if (eventType === 'reference') {
              pendingReferences.push(...payload.content);
            } else if (eventType === 'text') {
              assistantMessage.content += payload.content;
              setChatMessages([...accumulatedMessages, { ...assistantMessage }]);
            } else if (eventType === 'error') {
              toast.error(payload.content);
            }
          } catch (err) {
            // Skip malformed events
          }
        }
        return buf;
      };
      // Parse buffer and process references
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = parseBuffer(buffer);
      }
      if (buffer) parseBuffer(buffer);
      const matches = [...assistantMessage.content.matchAll(/\[([\d,\s]+)\]/g)];
      if (matches.length > 0) {
        assistantMessage.references = pendingReferences;
      }
      setChatMessages([...accumulatedMessages, assistantMessage]);
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        toast.error('Stream failed');
      }
    } finally {
      setIsStreaming(false);
      abortController.current = null;
    }
  };
  // Clear chat history and references
  const handleClear = () => {
    setChatMessages([]);
    setAllReferences([]);
    setSelectedResponse(null);
  };

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-200 font-sans selection:bg-blue-500/30 overflow-hidden">
      <Toaster 
        position="top-center" 
        toastOptions={{ 
          style: { background: '#171717', color: '#fff', border: '1px solid #262626' } 
        }} 
      />

      {/* Left Sidebar - Settings */}
      <SettingsSidebar
        isOpen={leftSidebarOpen}
        onToggle={() => setLeftSidebarOpen(!leftSidebarOpen)}
        settings={chatSettings}
        onSettingsChange={setChatSettings}
        onClear={handleClear}
      />

      {/* Main Chat Area - Flexible Width */}
      <main className="flex-1 flex flex-col min-w-0 relative" onClick={handleMainAreaClick}>
        {/* Glowing border effect for RAG mode */}
        {chatSettings.ragMode && (
          <>
            <div className="absolute inset-0 pointer-events-none z-20">
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent animate-pulse-slow" />
              <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-blue-500 to-transparent animate-pulse-slow" />
              <div className="absolute inset-y-0 left-0 w-px bg-gradient-to-b from-transparent via-blue-500 to-transparent animate-pulse-slow" />
              <div className="absolute inset-y-0 right-0 w-px bg-gradient-to-b from-transparent via-blue-500 to-transparent animate-pulse-slow" />
            </div>
            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_80px_rgba(59,130,246,0.08)] z-10" />
          </>
        )}

        {/* Header */}
        <header className="h-14 border-b border-neutral-800 flex items-center justify-between px-4 bg-neutral-950/80 backdrop-blur z-10 relative">
          {/* Left - Toggle Button (hidden when sidebar is open) */}
          <div className="w-10">
            {!leftSidebarOpen && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setLeftSidebarOpen(true);
                }}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors text-neutral-400 hover:text-white"
                title="Open Settings"
              >
                <PanelLeftOpen size={18} />
              </button>
            )}
          </div>

          {/* Center - Mode Indicator */}
          <div className="flex items-center gap-2">
            {chatSettings.ragMode ? (
              <div className="flex items-center gap-2 px-4 py-1.5 bg-blue-500/10 border border-blue-500/20 rounded-full">
                <Library size={14} className="text-blue-400" />
                <span className="text-sm font-medium text-blue-400">RAG Mode</span>
                <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                {/* Total document Count */}
                <div className="flex items-center gap-1">
                  <FileText size={16} />
                  <span className="font-medium text-sm">Documents count: {documentCount !== null ? documentCount : 'Loading...'}</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-4 py-1.5 bg-neutral-800/50 border border-neutral-700 rounded-full">
                <MessageCircle size={14} className="text-neutral-400" />
                <span className="text-sm font-medium text-neutral-400">Standard Mode</span>
              </div>
            )}
          </div> 

          {/* Right - Toggle Button (hidden when sidebar is open) */}
          <div className="w-10 flex justify-end">
            {!rightSidebarOpen && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setRightSidebarOpen(true);
                }}
                className="p-2 hover:bg-neutral-800 rounded-lg transition-colors text-neutral-400 hover:text-white relative"
                title="Open References"
              >
                <PanelRightOpen size={18} />
                {allReferences.length > 0 && (
                  <span className="absolute -top-1 -right-1 w-4 h-4 bg-blue-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                    {allReferences.length}
                  </span>
                )}
              </button>
            )}
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth">
          {chatMessages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-8 max-w-2xl mx-auto animate-in fade-in zoom-in-95 duration-500">
              <div className={`
                w-20 h-20 rounded-2xl flex items-center justify-center shadow-lg mb-6
                ${chatSettings.ragMode 
                  ? 'bg-blue-500/10 border border-blue-500/20' 
                  : 'bg-neutral-900 border border-neutral-800'
                }
              `}>
                <BotMessageSquare 
                  size={36} 
                  className={chatSettings.ragMode ? 'text-blue-400' : 'text-neutral-500'} 
                />
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">Smart Study Assistant</h2>
              <p className="text-neutral-400 text-center mb-6 max-w-md">
                Your AI-powered research companion with citation-backed answers.
              </p>
              {chatSettings.ragMode && (
                <div className="flex items-center gap-2 text-xs text-blue-400/70">
                  <Library size={12} />
                  <span>RAG-enhanced responses with document references</span>
                </div>
              )}
            </div>
          ) : (
            <div className="max-w-3xl mx-auto">
              {chatMessages.map((m, i) => (
                <MainChatMessageHistory 
                  key={i} 
                  message={m} 
                  isSelected={selectedResponse === m}
                  onSelect={setSelectedResponse}
                />
              ))}
              {isStreaming && (
                <div className="flex items-center gap-2 text-neutral-500 text-xs ml-11 animate-pulse">
                  <Loader2 size={12} className="animate-spin" />
                  <span>Thinking...</span>
                </div>
              )}
              <div ref={scrollRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-neutral-800 bg-neutral-950/80 backdrop-blur relative z-10" onClick={(e) => e.stopPropagation()}>
          <div className="max-w-3xl mx-auto">
            <div className={`
              flex items-end gap-2 rounded-xl shadow-lg transition-all p-2
              ${chatSettings.ragMode 
                ? 'bg-neutral-900 border border-blue-500/30 focus-within:border-blue-500/50 focus-within:shadow-[0_0_20px_rgba(59,130,246,0.15)]' 
                : 'bg-neutral-900 border border-neutral-700 focus-within:border-neutral-600'
              }
            `}>
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask a question..."
                className="flex-1 bg-transparent text-white px-2 py-1 text-sm max-h-40 min-h-[24px] resize-none focus:outline-none overflow-y-auto"
                rows={1}
                disabled={isStreaming}
              />
              <button
                onClick={isStreaming ? () => abortController.current?.abort() : handleSend}
                disabled={!userInput.trim() && !isStreaming}
                className={`
                  shrink-0 p-2 rounded-lg transition-all
                  ${userInput.trim() || isStreaming 
                    ? 'bg-blue-600 hover:bg-blue-500 text-white' 
                    : 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
                  }
                `}
              >
                {isStreaming ? <StopCircle size={16} /> : <Send size={16} />}
              </button>
            </div>
            <div className="text-[10px] text-neutral-600 text-center mt-2">
              AI can make mistakes. Please verify important information.
            </div>
          </div>
        </div>
      </main>

      {/* Right Sidebar - References (Resizable) */}
      <ReferencesSidebar
        isOpen={rightSidebarOpen}
        onToggle={() => setRightSidebarOpen(!rightSidebarOpen)}
        references={displayedReferences}
        width={rightSidebarWidth}
        onWidthChange={setRightSidebarWidth}
        hasSelection={selectedResponse !== null}
      />

      {/* Global Styles */}
      <style>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.8; }
        }
        .animate-pulse-slow {
          animation: pulse-slow 3s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}