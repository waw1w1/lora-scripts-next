import { stringify } from "smol-toml"

// Display only: submission, copying and export keep the original string array.
export function formatConfigPreview(config: Record<string, unknown>): string {
  if (!Array.isArray(config.preview_samples) || !config.preview_samples.length) return stringify(config)
  try {
    const samples = config.preview_samples.map((value: unknown) => {
      if (typeof value !== "string") throw new Error("Invalid sample")
      const parsed: unknown = JSON.parse(value)
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Invalid sample")
      return parsed as Record<string, unknown>
    })
    const ordinary = { ...config }
    delete ordinary.preview_samples
    return stringify(ordinary).trimEnd() + "\n\n" + samples.map((sample, index) =>
      `# Sample ${index + 1}\n[[preview_samples]]\n${stringify(sample).trimEnd()}`
    ).join("\n\n") + "\n"
  } catch {
    return stringify(config)
  }
}
