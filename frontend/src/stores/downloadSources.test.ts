// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createPinia, setActivePinia } from "pinia"
import { DOWNLOAD_SOURCES_KEY, chinaDownloadSources, writeDownloadSources } from "../engines/downloadSources"
import { SOURCE_CHANNELS, statusOf, useDownloadSourcesStore } from "./downloadSources"

const checkMock = vi.fn()
vi.mock("../api/downloadSources", () => ({
  downloadSourcesApi: { check: (...args: unknown[]) => checkMock(...args) },
}))

const OK_RESULT = { ok: true, status: 200, latency_ms: 120, url: "https://example.com", error: null }

function channelResult(channel: string, latency = 120) {
  return { [channel]: { ...OK_RESULT, latency_ms: latency } }
}

beforeEach(() => {
  setActivePinia(createPinia())
  localStorage.removeItem(DOWNLOAD_SOURCES_KEY)
  checkMock.mockReset()
  checkMock.mockImplementation((_payload: unknown, channel: string) => Promise.resolve(channelResult(channel)))
})

afterEach(() => {
  vi.useRealTimers()
})

describe("statusOf", () => {
  it("maps missing / failed / slow / ok results", () => {
    expect(statusOf(null)).toBe("unknown")
    expect(statusOf(undefined)).toBe("unknown")
    expect(statusOf({ ...OK_RESULT, ok: false, error: "boom" })).toBe("fail")
    expect(statusOf({ ...OK_RESULT, latency_ms: 5000 })).toBe("slow")
    expect(statusOf(OK_RESULT)).toBe("ok")
  })
})

describe("downloadSources store", () => {
  it("checks every channel with its own request", async () => {
    const store = useDownloadSourcesStore()
    await store.checkAll()
    expect(checkMock).toHaveBeenCalledTimes(4)
    for (const channel of SOURCE_CHANNELS) {
      expect(checkMock).toHaveBeenCalledWith({}, channel)
      expect(store.resultFor(channel)?.ok).toBe(true)
      expect(store.statusFor(channel)).toBe("ok")
    }
  })

  it("a hanging channel does not delay the others", async () => {
    let resolveHang: ((value: unknown) => void) | undefined
    checkMock.mockImplementation((_payload: unknown, channel: string) =>
      channel === "github" ? new Promise((resolve) => { resolveHang = resolve }) : Promise.resolve(channelResult(channel)),
    )
    const store = useDownloadSourcesStore()
    const all = store.checkAll()
    await vi.waitFor(() => {
      expect(store.statusFor("pip")).toBe("ok")
      expect(store.statusFor("huggingface")).toBe("ok")
    })
    expect(store.statusFor("github")).toBe("checking")
    resolveHang?.(channelResult("github"))
    await all
    expect(store.statusFor("github")).toBe("ok")
  })

  it("presumes fail after 1s without response, then flips back on a late success", async () => {
    vi.useFakeTimers()
    const resolvers: Array<() => void> = []
    checkMock.mockImplementation((_payload: unknown, channel: string) =>
      new Promise((resolve) => { resolvers.push(() => resolve(channelResult(channel))) }),
    )
    const store = useDownloadSourcesStore()
    const all = store.checkAll()

    expect(store.statusFor("pip")).toBe("checking")
    await vi.advanceTimersByTimeAsync(1100)
    expect(store.statusFor("pip")).toBe("fail")
    expect(store.presumedFail("pip")).toBe(true)

    resolvers.forEach((resolve) => resolve())
    await all
    expect(store.statusFor("pip")).toBe("ok")
  })

  it("checkOnce auto-checks at most once per session", async () => {
    const store = useDownloadSourcesStore()
    await store.checkOnce()
    await store.checkOnce()
    expect(checkMock).toHaveBeenCalledTimes(4)
  })

  it("sends explicit prefs with each per-channel request", async () => {
    const store = useDownloadSourcesStore()
    writeDownloadSources(chinaDownloadSources())
    store.reloadPrefs()
    await store.checkAll()
    expect(checkMock).toHaveBeenCalledWith(
      {
        pip_index_url: "https://pypi.tuna.tsinghua.edu.cn/simple",
        pytorch_index_url: "https://mirrors.aliyun.com/pytorch-wheels",
        hf_endpoint: "https://hf-mirror.com",
        github_url_prefix: "https://ghfast.top/",
      },
      "pip",
    )
  })

  it("marks a channel fail when its probe request fails", async () => {
    checkMock.mockRejectedValue(new Error("network down"))
    const store = useDownloadSourcesStore()
    await store.checkAll()
    expect(store.statusFor("pip")).toBe("fail")
    expect(store.resultFor("pip")?.error).toContain("network down")
    expect(store.checking).toBe(false)
  })

  it("a stale overlapping check cannot overwrite the latest result", async () => {
    const staleResolvers = new Map<string, (value: unknown) => void>()
    let call = 0
    checkMock.mockImplementation((_payload: unknown, channel: string) => {
      call += 1
      return call <= 4
        ? new Promise((resolve) => { staleResolvers.set(channel, resolve) })
        : Promise.resolve(channelResult(channel, 55))
    })
    const store = useDownloadSourcesStore()
    const stale = store.checkAll()
    const fresh = store.checkAll()
    await fresh
    for (const channel of SOURCE_CHANNELS) {
      staleResolvers.get(channel)?.({ [channel]: { ok: false, status: null, latency_ms: null, url: "", error: "stale" } })
    }
    await stale
    expect(store.resultFor("pip")?.latency_ms).toBe(55)
    expect(store.statusFor("pip")).toBe("ok")
    expect(store.checking).toBe(false)
  })
})
