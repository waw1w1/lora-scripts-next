import { apiData } from "./client"
import type { DownloadSourcesPayload } from "../engines/downloadSources"

export interface SourceCheckResult {
  ok: boolean
  status: number | null
  latency_ms: number | null
  url: string
  error: string | null
}

export type SourceCheckChannel = "pip" | "pytorch" | "huggingface" | "github"

export type SourceCheckMap = Record<SourceCheckChannel, SourceCheckResult>

export const downloadSourcesApi = {
  /** Probe one channel (pass `channel`) or all four; per-channel calls keep a hanging mirror from delaying the rest. */
  check: (payload: DownloadSourcesPayload = {}, channel?: SourceCheckChannel) =>
    apiData<Partial<SourceCheckMap>>("/api/download-sources/check", {
      method: "POST",
      body: JSON.stringify(channel ? { ...payload, channel } : payload),
    }),
}
