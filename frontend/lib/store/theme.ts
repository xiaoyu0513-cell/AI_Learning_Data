"use client";

import { create } from "zustand";

type ThemeState = {
  photo: string | null; // 背景照片 dataURL
  opacity: number; // 背景透明度 0-1
  setWallpaper: (photo: string | null, opacity: number) => void;
};

const KEY = "recruitment-wallpaper";

function load(): { photo: string | null; opacity: number } {
  if (typeof window === "undefined") return { photo: null, opacity: 0.25 };
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const p = JSON.parse(raw) as { photo: string | null; opacity?: number };
      return { photo: p.photo ?? null, opacity: p.opacity ?? 0.25 };
    }
  } catch {
    // 忽略损坏缓存
  }
  return { photo: null, opacity: 0.25 };
}

const init = load();

export const useTheme = create<ThemeState>((set) => ({
  photo: init.photo,
  opacity: init.opacity,
  setWallpaper: (photo, opacity) => {
    set({ photo, opacity });
    try {
      if (photo) {
        localStorage.setItem(KEY, JSON.stringify({ photo, opacity }));
      } else {
        localStorage.removeItem(KEY);
      }
    } catch {
      // 图片过大导致存储失败时，仅本次会话生效
    }
  },
}));