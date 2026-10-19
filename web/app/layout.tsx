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
    description: "浏览器里的 Tank Trouble AI 对战：强化学习 AI 对战原版 Laika、人机对战与 AI 对打。",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "Tank Trouble AI Arena",
      description: "强化学习 AI 对战原版 Laika，95.5% 真胜率，全部在浏览器内运行。",
      images: [{ url: socialImage, width: 1536, height: 864 }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Tank Trouble AI Arena",
      description: "强化学习 AI 对战原版 Laika，95.5% 真胜率，全部在浏览器内运行。",
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
