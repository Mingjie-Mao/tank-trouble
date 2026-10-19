import type { Metadata } from "next";
import { Arena } from "./Arena";

export const metadata: Metadata = {
  title: "Tank Trouble AI Arena",
  description: "强化学习 AI 对战 Tank Trouble 原版 AI Laika。",
};

export default function Home() {
  return <Arena />;
}
