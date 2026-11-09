"use client";

import { useEffect } from "react";
import { loadStoredTheme } from "@/lib/theme";
import { useTheme } from "@/lib/store/theme";

export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const photo = useTheme((s) => s.photo);
  const opacity = useTheme((s) => s.opacity);

  useEffect(() => {
    loadStoredTheme();
  }, []);

  return (
    <>
      {photo && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-0"
          style={{
            backgroundImage: `url(${photo})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            opacity,
            filter: "blur(3px) saturate(1.1)",
          }}
        />
      )}
      <div className="relative z-10 h-full">{children}</div>
    </>
  );
}