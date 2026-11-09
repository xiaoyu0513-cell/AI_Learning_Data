"use client";

import { useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  Code2,
  FileText,
  GitCompare,
  Globe,
  ImagePlus,
  Mic,
  Paperclip,
  Plus,
  Send,
  Square,
} from "lucide-react";

const menuItems = [
  { icon: ImagePlus, label: "上传图片或文件" },
  { icon: FileText, label: "上传文档" },
  { icon: Mic, label: "最近音频" },
  { icon: Globe, label: "Web 搜索" },
  { icon: Code2, label: "代码" },
  { icon: BrainCircuit, label: "RAG 检索" },
  { icon: GitCompare, label: "对比会话" },
];

export default function ChatInput({
  onSend,
  onStop,
  streaming,
}: {
  onSend: (q: string) => void;
  onStop: () => void;
  streaming: boolean;
}) {
  const [value, setValue] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const submit = () => {
    const q = value.trim();
    if (!q || streaming) return;
    onSend(q);
    setValue("");
  };

  return (
    <div className="shrink-0 border-t border-line bg-gradient-to-t from-canvas to-transparent px-4 pb-4 pt-2">
      <div className="mx-auto max-w-3xl">
        <div className="glass relative flex items-end gap-1 rounded-3xl p-2 shadow-2xl shadow-black/30 transition-all focus-within:border-accent-400/40 focus-within:ring-2 focus-within:ring-accent-500/20">
          <div ref={menuRef} className="relative shrink-0">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              className="flex h-9 w-9 items-center justify-center rounded-2xl text-muted transition-all hover:bg-white/5 hover:text-ink active:scale-95"
              title="附加"
            >
              <Plus className={`h-4 w-4 transition-transform ${menuOpen ? "rotate-45" : ""}`} />
            </button>
            {menuOpen && (
              <div className="absolute bottom-11 left-0 z-20 w-56 animate-fade-in overflow-hidden rounded-xl border border-line bg-panel/95 shadow-2xl backdrop-blur-xl">
                {menuItems.map((m) => (
                  <button
                    key={m.label}
                    onClick={() => setMenuOpen(false)}
                    className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left text-xs text-muted transition-colors hover:bg-white/5 hover:text-ink"
                  >
                    <m.icon className="h-4 w-4 text-faint" />
                    {m.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <textarea
            ref={ref}
            value={value}
            rows={1}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="输入你的问题，例如：适合 3 年经验、会 Go 和 K8s 的 AI 基础设施岗位"
            className="max-h-40 flex-1 resize-none bg-transparent px-1 py-2 text-sm outline-none placeholder:text-faint"
          />

          <button
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl text-muted transition-colors hover:bg-white/5 hover:text-ink active:scale-95"
            title="附件"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <button
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl text-muted transition-colors hover:bg-white/5 hover:text-ink active:scale-95"
            title="语音"
          >
            <Mic className="h-4 w-4" />
          </button>

          <button
            onClick={streaming ? onStop : submit}
            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl transition-all active:scale-95 ${
              streaming
                ? "bg-rose-500/20 text-rose-300 hover:bg-rose-500/30"
                : "bg-gradient-to-br from-accent-500 to-accent-600 text-white shadow-lg shadow-accent-500/30 hover:shadow-accent-500/50"
            }`}
          >
            {streaming ? <Square className="h-4 w-4" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
        <p className="mt-2 px-3 text-center text-[11px] text-faint">
          Enter 发送 · Shift+Enter 换行 · 支持岗位推荐与人岗匹配查询
        </p>
      </div>
    </div>
  );
}