import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Plus,
  Trash2,
  Paperclip,
  Bot,
  User,
  Menu,
  X,
  Sparkles,
  AlertCircle,
  Cpu,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function App() {
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
    loadSidebar();
    showWelcomeMessage();
  }, []);

  const showWelcomeMessage = () => {
    // Only show if no messages exist
    setMessages([
      {
        role: "AI",
        content:
          "Welcome to Veris Engine. I am your specialized CRAG assistant. How can I help you ground your data today?",
      },
    ]);
  };

  const checkBackendHealth = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/sessions");
      if (!res.ok) throw new Error();
    } catch {
      setIsDemoMode(true);
      setStatus("Demo Mode");
    }
  };

  const loadSidebar = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/sessions");
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error("Failed to load sessions", err);
    }
  };

  const loadChat = async (id) => {
    setCurrentSessionId(id);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/session/${id}`);
      const data = await res.json();
      setMessages(data.history || []);
      // Add a slight delay for smoother UI transition before scroll
      setTimeout(
        () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
    } catch {}
    if (window.innerWidth < 768) setSidebarOpen(false);
  };

  const startNewChat = () => {
    setCurrentSessionId("sess_" + Math.random().toString(36).substr(2, 9));
    setMessages([]);
    if (window.innerWidth < 768) setSidebarOpen(false);
  };

  const deleteSession = async (e, id) => {
    e.stopPropagation();
    await fetch(`http://127.0.0.1:8000/api/v1/session/${id}`, {
      method: "DELETE",
    });
    if (currentSessionId === id) startNewChat();
    loadSidebar();
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (isDemoMode) {
      setLoading(true);
      const demoResponses = [
        "Demo Mode: This is a simulated response. You can view the full architecture in the README.md file provided.",
        "I'm currently running in Preview Mode. To enable full RAG, please connect to the local server.",
      ];
      const response =
        demoResponses[Math.floor(Math.random() * demoResponses.length)];

      setMessages((prev) => [...prev, { role: "AI", content: demoResponses }]);
      setLoading(false);
      setStatus("Demo Mode");
      return;
    }

    const sid =
      currentSessionId || "sess_" + Math.random().toString(36).substr(2, 9);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sid);
    setStatus("Uploading...");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/upload", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "AI", content: `📎 ${file.name} processed.` },
        ]);
        setStatus("Idle");
      }
    } catch {
      alert("Upload failed.");
      setStatus("Idle");
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: "User", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setStatus("Thinking...");

    if (isDemoMode) {
      const demoAnswers = {
        "How does the Grader work?":
          "The Grader acts as an 'LLM-as-a-judge'. After the Hybrid Retriever fetches documents, the Grader evaluates them at zero-temperature. It checks if the context contains information relevant to your question. If it's even slightly related, it returns 'yes' and allows the Synthesizer to generate an answer. If it's off-topic, it rejects the context to prevent hallucinations.",
        "Explain the Hybrid Retriever":
          "Our Hybrid Retriever combines two powerful search methods: 1) Dense Semantic Search (Vector) using Nomic embeddings to understand 'intent', and 2) Sparse Lexical Search (BM25) to ensure exact keyword matches. We merge these to provide the most precise context possible.",
        "View project architecture":
          "The Nexus CRAG system is a multi-agent pipeline built on LangGraph. It processes queries through a deterministic workflow: Query Rewriter -> Hybrid Retriever -> Document Grader -> Synthesizer.",
      };

      const response =
        demoAnswers[input] ||
        "This is a demo response. Agent is offline, connect to local server to see the full system.";

      // Add the full response immediately in one go
      setMessages((prev) => [...prev, { role: "AI", content: response }]);

      setLoading(false);
      setStatus("Demo Mode");
      return;
    }

    // Backend API logic
    const sid =
      currentSessionId || "sess_" + Math.random().toString(36).substr(2, 9);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input, session_id: sid }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { role: "AI", content: data.answer }]);
      setLoading(false); // <--- Added this
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "AI", content: "Error: Could not reach agent." },
      ]);
      setLoading(false); // <--- Added this
    } finally {
      setStatus("Idle");
    }
  };

  useEffect(
    () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }),
    [messages, loading],
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0B0F19] text-slate-200 font-sans relative">
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
        <div className="absolute -top-[20%] -left-[10%] w-[500px] h-[500px] bg-indigo-900/20 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[10%] right-[10%] w-[400px] h-[400px] bg-blue-900/10 rounded-full blur-[100px]"></div>
      </div>

      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 288, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="z-30 h-full bg-[#0F1420]/80 backdrop-blur-xl border-r border-slate-800 flex flex-col shadow-2xl"
          >
            <div className="p-4 border-b border-slate-800/50">
              <div className="flex items-center justify-between mb-6">
                <h1 className="text-xl font-bold flex items-center gap-2">
                  <Sparkles className="text-indigo-500" /> Veris
                </h1>
                <button
                  className="md:hidden"
                  onClick={() => setSidebarOpen(false)}
                >
                  <X size={20} />
                </button>
              </div>
              <motion.button
                whileTap={{ scale: 0.98 }}
                onClick={startNewChat}
                className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white p-3 rounded-xl flex items-center justify-center gap-2 font-medium transition shadow-lg shadow-indigo-500/20"
              >
                <Plus size={20} /> New Conversation
              </motion.button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {sessions.map((s) => (
                <motion.div
                  key={s.id}
                  whileHover={{ x: 5 }}
                  onClick={() => loadChat(s.id)}
                  className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition ${currentSessionId === s.id ? "bg-indigo-900/30 border border-indigo-500/30" : "hover:bg-slate-800/50"}`}
                >
                  <span className="truncate text-sm text-slate-300">
                    {s.title}
                  </span>
                  <Trash2
                    size={14}
                    className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400"
                    onClick={(e) => deleteSession(e, s.id)}
                  />
                </motion.div>
              ))}
            </div>
            {/* Sidebar Footer */}
            <div className="p-4 border-t border-slate-800/50 bg-[#0A0D15] flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                <User size={16} className="text-indigo-400" />
              </div>
              <div className="text-xs">
                <div className="font-semibold text-slate-200">
                  Enterprise User
                </div>
                <div className="text-slate-500 text-[10px]">System Online</div>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <main className="flex-1 flex flex-col h-full bg-transparent relative z-10">
        <header className="h-16 flex items-center px-6 border-b border-slate-800/50 bg-[#0F1420]/30 backdrop-blur-md justify-between">
          <div className="flex items-center">
            <button
              className="mr-4 text-slate-400 hover:text-white transition"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu size={20} />
            </button>
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-full border ${isDemoMode ? "bg-amber-900/20 border-amber-500/30 text-amber-500" : "bg-slate-800/50 border-slate-700"}`}
            >
              <span
                className={`w-2 h-2 rounded-full animate-pulse ${isDemoMode ? "bg-amber-500" : "bg-emerald-500"}`}
              ></span>
              <span className="text-[11px] font-bold tracking-wider">
                {status}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-widest border border-slate-800 px-3 py-1 rounded-lg">
            <Cpu size={12} className="text-indigo-400" /> OLLAMA_LLAMA3
          </div>
          {isDemoMode && (
            <div className="flex items-center gap-2 text-amber-500/70 text-xs font-bold">
              <AlertCircle size={14} /> PREVIEW MODE
            </div>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-8 flex flex-col">
          {messages.map((m, i) => (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              key={i}
              className={`flex gap-4 ${m.role === "User" ? "justify-end" : ""}`}
            >
              {m.role === "AI" && (
                <div className="w-9 h-9 rounded-full bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20 shrink-0">
                  <Bot size={18} className="text-indigo-400" />
                </div>
              )}
              <motion.div
                layout
                className={`max-w-[80%] md:max-w-[60%] p-4 rounded-2xl shadow-sm ${m.role === "User" ? "bg-indigo-600 text-white" : "bg-[#161B28]/80 border border-slate-800"}`}
              >
                <p className="text-[15px] leading-relaxed">{m.content}</p>
              </motion.div>
            </motion.div>
          ))}
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-4"
            >
              <div className="w-9 h-9 rounded-full bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20 shrink-0">
                <Bot size={18} className="text-indigo-400" />
              </div>
              <div className="bg-[#161B28]/80 border border-slate-800 p-4 rounded-2xl flex gap-1 items-center">
                {[0, 1, 2].map((n) => (
                  <div
                    key={n}
                    className="w-2 h-2 bg-indigo-500/50 rounded-full animate-bounce"
                    style={{ animationDelay: `${n * 0.15}s` }}
                  ></div>
                ))}
              </div>
            </motion.div>
          )}
          {isDemoMode && messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-4 justify-center mt-10"
            >
              {[
                "How does the Grader work?",
                "Explain the Hybrid Retriever",
                "View project architecture",
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setInput(suggestion);
                    sendMessage();
                  }}
                  className="bg-[#161B28] border border-slate-700 hover:border-indigo-500/50 p-4 rounded-xl text-sm transition-all hover:shadow-lg hover:shadow-indigo-500/10"
                >
                  {suggestion}
                </button>
              ))}
            </motion.div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="p-6 shrink-0">
          <motion.div
            whileHover={{ scale: 1.01 }}
            className="max-w-4xl mx-auto flex items-center gap-3 bg-[#161B28] p-3 rounded-2xl border border-slate-700 focus-within:border-indigo-500 transition-all shadow-2xl"
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleUpload}
            />
            <button
              onClick={() => fileInputRef.current.click()}
              className="p-3 text-slate-400 hover:text-indigo-400 transition"
            >
              <Paperclip size={20} />
            </button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Type your question..."
              className="flex-1 bg-transparent p-2 focus:outline-none"
            />
            <motion.button
              whileTap={{ scale: 0.9 }}
              onClick={sendMessage}
              className="bg-indigo-600 text-white p-3 rounded-xl shadow-lg shadow-indigo-600/20"
            >
              <Send size={20} />
            </motion.button>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

export default App;
