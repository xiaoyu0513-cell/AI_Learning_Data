"use client";

import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Check,
  ImagePlus,
  Link2,
  Loader2,
  Palette,
  RotateCcw,
  X,
} from "lucide-react";
import {
  applyPalette,
  extractPalette,
  resetTheme,
  type Palette as ThemePalette,
} from "@/lib/theme";
import { useTheme } from "@/lib/store/theme";

export default function ThemeSwitcher() {
  const [open, setOpen] = useState(false);
  const [palette, setPalette] = useState<ThemePalette | null>(null);
  const [preview, setPreview] = useState<string | null>(null); // 原图 dataURL
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const [asWallpaper, setAsWallpaper] = useState(false);
  const [bgOpacity, setBgOpacity] = useState(0.25);

  const setWallpaper = useTheme((s) => s.setWallpaper);

  const fileRef = useRef<HTMLInputElement>(null);

  const loadFromSrc = (src: string) => {
    setBusy(true);
    setErr("");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        setPalette(extractPalette(img));
        setPreview(src);
      } catch {
        setErr("无法解析这张图片的像素，请换一张（外链图片需支持跨域，建议直接上传本地照片）");
      } finally {
        setBusy(false);
      }
    };
    img.onerror = () => {
      setErr("图片加载失败，请检查地址或改用上传本地照片");
      setBusy(false);
    };
    img.src = src;
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadFromSrc(reader.result as string);
    reader.readAsDataURL(file);
  };

  const apply = () => {
    if (!palette) return;
    applyPalette(palette);
    if (preview && asWallpaper) {
      setWallpaper(preview, bgOpacity);
    } else if (!asWallpaper) {
      setWallpaper(null, bgOpacity);
    }
    setOpen(false);
  };

  const reset = () => {
    resetTheme();
    setWallpaper(null, 0.25);
    setPalette(null);
    setPreview(null);
    setUrl("");
    setAsWallpaper(false);
    setBgOpacity(0.25);
    setErr("");
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title="照片主题"
        className="flex h-8 w-8 items-center justify-center rounded-lg border border-line text-muted transition-colors hover:bg-white/5 hover:text-ink"
      >
        <Palette className="h-4 w-4" />
      </button>

      {open &&
        createPortal(
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <div
            className="card max-h-[88vh] w-full max-w-md overflow-y-auto p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between">
              <div>
                <h2 className="text-base font-semibold">照片主题</h2>
                <p className="mt-0.5 text-xs text-faint">
                  提取照片主色作强调色，也可用照片作半透明背景
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="rounded-lg p-1.5 text-muted transition-colors hover:bg-white/5 hover:text-ink"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* 上传 */}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={busy}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-line-strong px-4 py-6 text-sm text-muted transition-colors hover:border-accent-400/50 hover:text-ink"
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ImagePlus className="h-4 w-4" />
              )}
              上传本地照片
            </button>

            {/* URL */}
            <div className="mt-3 flex items-center gap-2">
              <div className="flex flex-1 items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-muted focus-within:border-accent-400/50">
                <Link2 className="h-4 w-4 shrink-0" />
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="或粘贴图片 URL…"
                  className="w-full bg-transparent text-sm text-ink outline-none placeholder:text-faint"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && url.trim()) loadFromSrc(url.trim());
                  }}
                />
              </div>
              <button
                onClick={() => url.trim() && loadFromSrc(url.trim())}
                disabled={busy || !url.trim()}
                className="rounded-lg bg-accent-600 px-3 py-2 text-sm text-white disabled:opacity-50"
              >
                提取
              </button>
            </div>

            {err && (
              <div className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
                {err}
              </div>
            )}

            {/* 预览与设置 */}
            {palette && (
              <div className="mt-4 space-y-3">
                <div className="flex items-center gap-4">
                  {preview && (
                    <img
                      src={preview}
                      alt="预览"
                      className="h-16 w-16 rounded-lg object-cover"
                    />
                  )}
                  <div className="flex-1">
                    <div className="mb-2 flex gap-2">
                      <div className="flex items-center gap-1.5 text-xs text-muted">
                        <span
                          className="h-4 w-4 rounded-full border border-white/20"
                          style={{ background: palette.primary }}
                        />
                        主色
                      </div>
                      <div className="flex items-center gap-1.5 text-xs text-muted">
                        <span
                          className="h-4 w-4 rounded-full border border-white/20"
                          style={{ background: palette.secondary }}
                        />
                        辅助色
                      </div>
                    </div>
                    <div
                      className="h-2 rounded-full"
                      style={{
                        background: `linear-gradient(90deg, ${palette.primary}, ${palette.secondary})`,
                      }}
                    />
                  </div>
                </div>

                {/* 背景开关 + 透明度 */}
                <div className="rounded-xl border border-line bg-panel/60 p-3">
                  <label className="flex cursor-pointer items-center justify-between gap-3">
                    <span className="text-sm text-ink">用照片作半透明背景</span>
                    <input
                      type="checkbox"
                      checked={asWallpaper}
                      onChange={(e) => setAsWallpaper(e.target.checked)}
                      className="accent-accent-500 h-4 w-4"
                    />
                  </label>

                  <div className="mt-3 flex items-center gap-3">
                    <span className="shrink-0 text-xs text-faint">背景透明度</span>
                    <input
                      type="range"
                      min={0.05}
                      max={1}
                      step={0.05}
                      value={bgOpacity}
                      onChange={(e) => setBgOpacity(Number(e.target.value))}
                      className="accent-accent-500 w-full"
                    />
                    <span className="tnum w-10 shrink-0 text-right text-xs text-muted">
                      {Math.round(bgOpacity * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* 操作 */}
            <div className="mt-5 flex items-center justify-end gap-2">
              <button
                onClick={reset}
                className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-2 text-xs text-muted transition-colors hover:text-ink"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                恢复默认
              </button>
              <button
                onClick={apply}
                disabled={!palette}
                className="flex items-center gap-1.5 rounded-lg bg-accent-600 px-4 py-2 text-sm text-white transition-colors hover:bg-accent-500 disabled:opacity-50"
              >
                <Check className="h-4 w-4" />
                应用主题
              </button>
            </div>
          </div>
        </div>
        , document.body)}
    </>
  );
}