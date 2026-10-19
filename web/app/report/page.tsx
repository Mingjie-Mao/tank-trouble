import type { Metadata } from "next";
import { Report } from "./Report";

export const metadata: Metadata = {
  title: "Tank Trouble AI — Report",
  description: "League PPO 与 Tactical 的方法与结果：联赛自博弈、安全护盾、MPC 老师与蒸馏。",
};

export default function ReportPage() {
  return <Report />;
}
