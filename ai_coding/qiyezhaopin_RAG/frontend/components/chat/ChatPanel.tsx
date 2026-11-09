"use client";

import { useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronDown,
  Database,
  FileText,
  GitCompare,
  Network,
  Sparkles,
  User,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChatInput from "./ChatInput";
import { streamChat } from "@/lib/api";
import { useWorkbench } from "@/lib/store/workbench";
import type { ChatEvent, ChatMessage, Citation } from "@/lib/types";

let msgSeq = 0;

const samples = [
  "RAG算法工程师 需要掌握哪些技能？",
  "推荐几个 大模型 相关的岗位",
  "字节跳动 有哪些岗位在招？",
];

const capabilities = [
  { icon: Network, title: "图检索", desc: "Neo4j 多跳遍历" },
  { icon: Database, title: "向量检索", desc: "Milvus 语义匹配" },
  { icon: GitCompare, title: "人岗匹配", desc: "技能 Gap 分析" },
];

function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-8 px-6 py-10 text-center">
      <div className="flex flex-col items-center gap-4">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-400 via-accent-500 to-accent-2 shadow-xl shadow-accent-500/30">
          <Sparkles className="h-7 w-7 text-white" />
        </div>
        <h2 className="text-gradient text-2xl font-semibold">招聘知识图谱 · 智能问答</h2>
        <p className="max-w-md text-sm leading-relaxed text-muted">
          基于 Neo4j 图检索 + Milvus 向量检索的增强问答系统，支持人岗匹配与岗位推荐，回答附带引用溯源。
        </p>
      </div>
      <div className="grid w-full max-w-lg grid-cols-1 gap-2.5 sm:grid-cols-3">
        {capabilities.map((c) => (
          <div
            key={c.title}
            className="card card-hover flex flex-col items-start gap-1.5 p-3.5 text-left"
          >
            <c.icon className="h-4 w-4 text-accent-400" />
            <span className="text-xs font-medium text-ink">{c.title}</span>
            <span className="text-[11px] text-faint">{c.desc}</span>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap justify-center gap-2.5">
        {samples.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="card card-hover px-4 py-2 text-xs text-muted hover:text-ink"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (!citations || citations.length === 0) return null;
  return (
    <div className="mt-2 overflow-hidden rounded-xl border border-line bg-panel/40 backdrop-blur-sm">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-muted transition-colors hover:text-ink"
      >
        <FileText className="h-3.5 w-3.5 text-accent-400" />
        检索依据 · {citations.length} 条引用
        <span className="ml-auto flex items-center gap-2 text-[10px] text-faint">
          图检索 + 向量
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          />
        </span>
      </button>
      {open && (
        <div className="space-y-1 border-t border-line p-2">
          {citations.map((c) => (
            <div
              key={c.index}
              className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-xs transition-colors hover:bg-white/5"
            >
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-accent-500/15 font-mono text-[10px] text-accent-300">
                {c.index}
              </span>
              <span className="truncate font-medium text-ink">
                {c.job_name || c.node_id || "—"}
              </span>
              <span className="ml-auto flex shrink-0 items-center gap-3">
                <span className="text-[10px] text-muted">{c.source}</span>
                <span className="font-mono text-[10px] text-accent-300">
                  {Math.round((c.score ?? 0) * 100)}%
                </span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Bubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          isUser ? "bg-amber-500/20 text-amber-300" : "bg-accent-500/20 text-accent-300"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className={`max-w-[82%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block rounded-2xl px-4 py-2.5 text-left text-sm shadow-md ${
            isUser
              ? "bg-gradient-to-br from-accent-500 to-accent-600 text-white"
              : "glass"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{msg.content}</p>
          ) : (
            <div className="md">
              {msg.content ? (
                <>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  {msg.streaming && <span className="streaming-caret" />}
                </>
              ) : msg.streaming ? (
                <div className="flex items-center gap-2 py-1 text-muted">
                  <Sparkles className="h-3.5 w-3.5 animate-pulse text-accent-400" />
                  <span className="skeleton h-3.5 w-40 rounded-full" />
                </div>
              ) : (
                <span className="text-muted">…</span>
              )}
            </div>
          )}
        </div>
        {!isUser && msg.citations && msg.citations.length > 0 && (
          <Citations citations={msg.citations} />
        )}
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const pushTrace = useWorkbench((s) => s.pushTrace);
  const resetObs = useWorkbench((s) => s.reset);
  const setStatus = useWorkbench((s) => s.setStatus);
  const setRoute = useWorkbench((s) => s.setRoute);
  const setCitations = useWorkbench((s) => s.setCitations);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const send = async (query: string) => {
    if (!query.trim() || streaming) return;
    const userMsg: ChatMessage = { id: `m${++msgSeq}`, role: "user", content: query };
    const assistantId = `m${++msgSeq}`;
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    setStreaming(true);
    resetObs(query);

    const append = (delta: string) =>
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + delta } : m,
        ),
      );
    const patch = (p: Partial<ChatMessage>) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...p } : m)),
      );

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        query,
        (evt: ChatEvent) => {
          switch (evt.type) {
            case "status":
              setStatus(evt.content);
              pushTrace(evt.content);
              break;
            case "route_decision":
              setRoute(evt.content);
              pushTrace(
                `路由决策：${evt.content.strategy}（置信度 ${evt.content.confidence.toFixed(2)}）`,
              );
              break;
            case "citations":
              setCitations(evt.content);
              patch({ citations: evt.content });
              pushTrace(`检索到 ${evt.content.length} 条引用证据`);
              break;
            case "token":
              append(evt.content);
              break;
            case "done":
              patch({ streaming: false });
              pushTrace("生成完成");
              break;
            case "error":
              patch({ streaming: false, error: true });
              append(`\n\n> ⚠️ ${evt.content}`);
              break;
          }
        },
        controller.signal,
      );
    } catch {
      patch({ streaming: false, error: true });
      append("\n\n> ⚠️ 请求失败或已中断");
    } finally {
      setStreaming(false);
    }
  };

  const stop = () => abortRef.current?.abort();

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <Welcome onPick={send} />
        ) : (
          messages.map((m) => <Bubble key={m.id} msg={m} />)
        )}
      </div>
      <ChatInput onSend={send} onStop={stop} streaming={streaming} />
    </div>
  );
}