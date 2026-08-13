import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host");
  const protocol = requestHeaders.get("x-forwarded-proto")
    ?? (host?.startsWith("localhost") ? "http" : "https");
  const origin = host ? `${protocol}://${host}` : "http://localhost:3000";
  const socialImage = new URL("/og.png", origin).toString();
  return {
    title: "Tank Trouble AI Arena",
    description: "浏览器原生 Tank Trouble AI 对战：Tactical 挑战 Laika、人机对战与自博弈。",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "Tank Trouble AI Arena",
      description: "Tank Trouble Tactical vs Laika — 浏览器原生实时搜索对战。",
      images: [{ url: socialImage, width: 1536, height: 864 }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Tank Trouble AI Arena",
      description: "Tank Trouble Tactical vs Laika — 浏览器原生实时搜索对战。",
      images: [socialImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
