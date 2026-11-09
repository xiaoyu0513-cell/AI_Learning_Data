"use client";

import { create } from "zustand";
import type { Citation, RouteDecision } from "@/lib/types";

export type TraceStep = {
  id: number;
  text: string;
  time: string;
};

export type Observability = {
  status: string;
  query: string;
  route: RouteDecision | null;
  citations: Citation[];
  trace: TraceStep[];
};

type WorkbenchState = {
  observability: Observability;
  rightOpen: boolean;
  setStatus: (s: string) => void;
  setRoute: (r: RouteDecision) => void;
  setCitations: (c: Citation[]) => void;
  pushTrace: (text: string) => void;
  reset: (query: string) => void;
  toggleRight: () => void;
};

let seq = 0;
const now = () => new Date().toLocaleTimeString("zh-CN");

export const useWorkbench = create<WorkbenchState>((set) => ({
  observability: {
    status: "空闲",
    query: "",
    route: null,
    citations: [],
    trace: [],
  },
  rightOpen: false,
  toggleRight: () => set((st) => ({ rightOpen: !st.rightOpen })),
  setStatus: (s) =>
    set((st) => ({ observability: { ...st.observability, status: s } })),
  setRoute: (r) =>
    set((st) => ({ observability: { ...st.observability, route: r } })),
  setCitations: (c) =>
    set((st) => ({ observability: { ...st.observability, citations: c } })),
  pushTrace: (text) =>
    set((st) => ({
      observability: {
        ...st.observability,
        trace: [
          ...st.observability.trace,
          { id: ++seq, text, time: now() },
        ],
      },
    })),
  reset: (query) =>
    set(() => ({
      observability: {
        status: "检索中...",
        query,
        route: null,
        citations: [],
        trace: [{ id: ++seq, text: `已接收查询：${query}`, time: now() }],
      },
    })),
}));