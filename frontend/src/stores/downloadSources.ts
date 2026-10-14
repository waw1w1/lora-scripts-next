import { defineStore } from "pinia"
import { computed, ref } from "vue"
import { downloadSourcesApi, type SourceCheckChannel, type SourceCheckMap, type SourceCheckResult } from "../api/downloadSources"
import {
  readDownloadSources,
  resolvedDownloadSourcesPayload,
  type DownloadSourcesPayload,
  type DownloadSourcesPrefs,
} from "../engines/downloadSources"

export type SourceStatus = "unknown" | "checking" | "ok" | "slow" | "fail"

export const SLOW_THRESHOLD_MS = 2000

/** No response within this window → presumptively red; a late success flips it back. */
export const PRESUMED_FAIL_AFTER_MS = 1000

export const SOURCE_CHANNELS: SourceCheckChannel[] = ["pip", "pytorch", "huggingface", "github"]

export function statusOf(result: SourceCheckResult | null | undefined): SourceStatus {
  if (!result) return "unknown"
  if (!result.ok) return "fail"
  return (result.latency_ms ?? 0) > SLOW_THRESHOLD_MS ? "slow" : "ok"
}

export const useDownloadSourcesStore = defineStore("downloadSources", () => {
  const prefs = ref<DownloadSourcesPrefs>(readDownloadSources())
  const results = ref<Partial<SourceCheckMap>>({})
  const pending = ref<Set<SourceCheckChannel>>(new Set())
  const timedOut = ref<Set<SourceCheckChannel>>(new Set())
  const didAutoCheck = ref(false)

  const checking = computed(() => pending.value.size > 0)

  function reloadPrefs() {
    prefs.value = readDownloadSources()
  }

  function statusFor(channel: SourceCheckChannel): SourceStatus {
    const result = results.value[channel]
    if (result) return statusOf(result)
    if (timedOut.value.has(channel)) return "fail"
    if (pending.value.has(channel)) return "checking"
    return "unknown"
  }

  function resultFor(channel: SourceCheckChannel): SourceCheckResult | null {
    return results.value[channel] ?? null
  }

  /** True while the channel probe is still running past the presumed-fail window. */
  function presumedFail(channel: SourceCheckChannel): boolean {
    return !results.value[channel] && pending.value.has(channel) && timedOut.value.has(channel)
  }

  /** Per-channel request generation; only the latest check may mutate state. */
  const generations = new Map<SourceCheckChannel, number>()

  async function checkOne(channel: SourceCheckChannel, payload: DownloadSourcesPayload) {
    const generation = (generations.get(channel) ?? 0) + 1
    generations.set(channel, generation)
    const isLatest = () => generations.get(channel) === generation
    pending.value.add(channel)
    timedOut.value.delete(channel)
    delete results.value[channel]
    const timer = window.setTimeout(() => {
      if (isLatest() && pending.value.has(channel)) timedOut.value.add(channel)
    }, PRESUMED_FAIL_AFTER_MS)
    try {
      const data = await downloadSourcesApi.check(payload, channel)
      if (!isLatest()) return
      results.value[channel] = data[channel] ?? { ok: false, status: null, latency_ms: null, url: "", error: "empty probe result" }
    } catch (error) {
      if (!isLatest()) return
      results.value[channel] = {
        ok: false,
        status: null,
        latency_ms: null,
        url: "",
        error: error instanceof Error ? error.message : String(error),
      }
    } finally {
      window.clearTimeout(timer)
      if (isLatest()) {
        pending.value.delete(channel)
        timedOut.value.delete(channel)
      }
    }
  }

  /** Probe every channel with an independent request so one hanging mirror never delays the rest. */
  async function checkAll(payload?: DownloadSourcesPayload) {
    const body = payload ?? resolvedDownloadSourcesPayload(prefs.value)
    await Promise.allSettled(SOURCE_CHANNELS.map((channel) => checkOne(channel, body)))
  }

  /** Startup auto-check: at most once per session. */
  async function checkOnce() {
    if (didAutoCheck.value) return
    didAutoCheck.value = true
    reloadPrefs()
    await checkAll()
  }

  return { prefs, results, checking, statusFor, resultFor, presumedFail, reloadPrefs, checkAll, checkOnce }
})
