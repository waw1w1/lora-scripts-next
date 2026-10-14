/** Download-source prefs for engine installs (Fast / Musubi). UI-first; backend wires later. */

export const DOWNLOAD_SOURCES_KEY = "nt.training.downloadSources"

/** Window event fired after prefs are saved so live UI (sidebar chips) can refresh. */
export const DOWNLOAD_SOURCES_CHANGED_EVENT = "nt:download-sources-changed"

export type SourcePreset = "auto" | "official" | "china" | "custom"

/** "auto" defers to the engine installer defaults (e.g. HF → hf-mirror.com). */
export type PipChoice = "auto" | "official" | "tsinghua" | "aliyun" | "douban" | "custom"
export type PytorchChoice = "auto" | "official" | "aliyun" | "tsinghua" | "bfsu" | "custom"
export type HfChoice = "auto" | "official" | "hf-mirror" | "modelscope" | "custom"
/** GitHub clone/download URL prefixes (same pattern as portable updater). */
export type GithubChoice = "auto" | "official" | "ghfast" | "ghproxy" | "custom"

export interface ChannelConfig<T extends string> {
  choice: T
  customUrl: string
}

export interface DownloadSourcesPrefs {
  preset: SourcePreset
  pip: ChannelConfig<PipChoice>
  pytorch: ChannelConfig<PytorchChoice>
  huggingface: ChannelConfig<HfChoice>
  github: ChannelConfig<GithubChoice>
}

export const PIP_URLS: Record<Exclude<PipChoice, "custom" | "auto">, string> = {
  official: "https://pypi.org/simple",
  tsinghua: "https://pypi.tuna.tsinghua.edu.cn/simple",
  aliyun: "https://mirrors.aliyun.com/pypi/simple",
  douban: "https://pypi.douban.com/simple",
}

/**
 * PyTorch wheel index bases. Installer should append `/cuXXX` (e.g. cu128)
 * to match the CUDA tag used by Fast / Musubi.
 */
export const PYTORCH_URLS: Record<Exclude<PytorchChoice, "custom" | "auto">, string> = {
  official: "https://download.pytorch.org/whl",
  aliyun: "https://mirrors.aliyun.com/pytorch-wheels",
  tsinghua: "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels",
  bfsu: "https://mirrors.bfsu.edu.cn/pytorch-wheels",
}

export const HF_URLS: Record<Exclude<HfChoice, "custom" | "auto">, string> = {
  official: "https://huggingface.co",
  "hf-mirror": "https://hf-mirror.com",
  /** ModelScope's Hugging Face compatible endpoint (subset of models). */
  modelscope: "https://huggingface.modelscope.cn",
}

/** Prefix applied before https://github.com/... when cloning / fetching releases. */
export const GITHUB_PREFIXES: Record<Exclude<GithubChoice, "custom" | "auto">, string> = {
  official: "",
  ghfast: "https://ghfast.top/",
  ghproxy: "https://mirror.ghproxy.com/",
}

export function defaultDownloadSources(): DownloadSourcesPrefs {
  return {
    preset: "auto",
    pip: { choice: "auto", customUrl: "" },
    pytorch: { choice: "auto", customUrl: "" },
    huggingface: { choice: "auto", customUrl: "" },
    github: { choice: "auto", customUrl: "" },
  }
}

/** China-friendly preset: mirror pip / PyTorch / HF + GitHub proxy. */
export function chinaDownloadSources(): DownloadSourcesPrefs {
  return {
    preset: "china",
    pip: { choice: "tsinghua", customUrl: "" },
    pytorch: { choice: "aliyun", customUrl: "" },
    huggingface: { choice: "hf-mirror", customUrl: "" },
    github: { choice: "ghfast", customUrl: "" },
  }
}

function asChoice<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback
}

export function readDownloadSources(): DownloadSourcesPrefs {
  const fallback = defaultDownloadSources()
  try {
    const parsed = JSON.parse(localStorage.getItem(DOWNLOAD_SOURCES_KEY) || "null")
    if (!parsed || typeof parsed !== "object") return fallback
    const preset = asChoice(parsed.preset, ["auto", "official", "china", "custom"] as const, "auto")
    return {
      preset,
      pip: {
        choice: asChoice(parsed.pip?.choice, ["auto", "official", "tsinghua", "aliyun", "douban", "custom"] as const, "auto"),
        customUrl: typeof parsed.pip?.customUrl === "string" ? parsed.pip.customUrl : "",
      },
      pytorch: {
        choice: asChoice(
          parsed.pytorch?.choice,
          ["auto", "official", "aliyun", "tsinghua", "bfsu", "custom"] as const,
          "auto",
        ),
        customUrl: typeof parsed.pytorch?.customUrl === "string" ? parsed.pytorch.customUrl : "",
      },
      huggingface: {
        choice: asChoice(
          parsed.huggingface?.choice,
          ["auto", "official", "hf-mirror", "modelscope", "custom"] as const,
          "auto",
        ),
        customUrl: typeof parsed.huggingface?.customUrl === "string" ? parsed.huggingface.customUrl : "",
      },
      github: {
        choice: asChoice(parsed.github?.choice, ["auto", "official", "ghfast", "ghproxy", "custom"] as const, "auto"),
        customUrl: typeof parsed.github?.customUrl === "string" ? parsed.github.customUrl : "",
      },
    }
  } catch {
    return fallback
  }
}

export function writeDownloadSources(prefs: DownloadSourcesPrefs) {
  localStorage.setItem(DOWNLOAD_SOURCES_KEY, JSON.stringify(prefs))
  if (typeof window !== "undefined") window.dispatchEvent(new Event(DOWNLOAD_SOURCES_CHANGED_EVENT))
}

export function applyPreset(preset: SourcePreset, current: DownloadSourcesPrefs): DownloadSourcesPrefs {
  if (preset === "auto") return defaultDownloadSources()
  if (preset === "official") {
    return {
      preset: "official",
      pip: { choice: "official", customUrl: "" },
      pytorch: { choice: "official", customUrl: "" },
      huggingface: { choice: "official", customUrl: "" },
      github: { choice: "official", customUrl: "" },
    }
  }
  if (preset === "china") return chinaDownloadSources()
  return { ...current, preset: "custom" }
}

/** Explicit URL for a channel; "" when the channel defers to engine defaults ("auto"). */
export function resolvePipIndexUrl(prefs: DownloadSourcesPrefs): string {
  if (prefs.pip.choice === "auto") return ""
  if (prefs.pip.choice === "custom") return prefs.pip.customUrl.trim() || PIP_URLS.official
  return PIP_URLS[prefs.pip.choice]
}

export function resolvePytorchIndexUrl(prefs: DownloadSourcesPrefs): string {
  if (prefs.pytorch.choice === "auto") return ""
  if (prefs.pytorch.choice === "custom") return prefs.pytorch.customUrl.trim() || PYTORCH_URLS.official
  return PYTORCH_URLS[prefs.pytorch.choice]
}

export function resolveHfEndpoint(prefs: DownloadSourcesPrefs): string {
  if (prefs.huggingface.choice === "auto") return ""
  if (prefs.huggingface.choice === "custom") return prefs.huggingface.customUrl.trim() || HF_URLS.official
  return HF_URLS[prefs.huggingface.choice]
}

export function resolveGithubPrefix(prefs: DownloadSourcesPrefs): string {
  if (prefs.github.choice === "auto") return ""
  if (prefs.github.choice === "custom") return prefs.github.customUrl.trim()
  return GITHUB_PREFIXES[prefs.github.choice]
}

/** Payload fields sent with Fast / Musubi install & repair requests. */
export interface DownloadSourcesPayload {
  pip_index_url?: string
  pytorch_index_url?: string
  hf_endpoint?: string
  github_url_prefix?: string
}

/**
 * Channels left on "auto" are omitted so the backend keeps the engine
 * installer defaults (see mikazuki/download_sources.py: empty → None).
 */
export function resolvedDownloadSourcesPayload(prefs: DownloadSourcesPrefs = readDownloadSources()): DownloadSourcesPayload {
  const payload: DownloadSourcesPayload = {}
  const pip = resolvePipIndexUrl(prefs)
  const pytorch = resolvePytorchIndexUrl(prefs)
  const hf = resolveHfEndpoint(prefs)
  const github = resolveGithubPrefix(prefs)
  if (pip) payload.pip_index_url = pip
  if (pytorch) payload.pytorch_index_url = pytorch
  if (hf) payload.hf_endpoint = hf
  if (github) payload.github_url_prefix = github
  return payload
}

/** Summary chips for the collapsed header (resolved choice labels, not URLs). */
export function activeSourceSummary(prefs: DownloadSourcesPrefs): {
  pip: string
  pytorch: string
  huggingface: string
  github: string
} {
  return {
    pip: prefs.pip.choice,
    pytorch: prefs.pytorch.choice,
    huggingface: prefs.huggingface.choice,
    github: prefs.github.choice,
  }
}
