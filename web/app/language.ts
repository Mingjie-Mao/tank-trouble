"use client";

import { useSyncExternalStore } from "react";

export type Language = "zh" | "en";

const STORAGE_KEY = "tank-trouble-language";
const CHANGE_EVENT = "tank-trouble-language-change";

function snapshot(): Language {
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === "zh" || saved === "en") return saved;
  return window.navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

function subscribe(callback: () => void) {
  window.addEventListener(CHANGE_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(CHANGE_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

/** The page language, shared by the arena and the report and kept in localStorage. */
export function useLanguage(): Language {
  return useSyncExternalStore(subscribe, snapshot, (): Language => "zh");
}

export function setLanguage(next: Language) {
  window.localStorage.setItem(STORAGE_KEY, next);
  window.dispatchEvent(new Event(CHANGE_EVENT));
}
