import { describe, expect, it } from "vitest"
import { stringify } from "smol-toml"
import { formatConfigPreview } from "./configPreview"

describe("config preview", () => {
  it("keeps ordinary configurations unchanged", () => {
    const config = { learning_rate: "1e-4", path: "D:/models/test" }
    expect(formatConfigPreview(config)).toBe(stringify(config))
  })
  it("groups samples and puts each field on a separate line without mutating inputs", () => {
    const config = { steps: 100, preview_samples: [JSON.stringify({ prompt: "girl, smile", seed: 42 }), JSON.stringify({ prompt: "second", seed: 9 })] }
    const original = stringify(config)
    const text = formatConfigPreview(config)
    expect(text).toContain('steps = 100')
    expect(text).toContain('# Sample 1\n[[preview_samples]]\nprompt = "girl, smile"\nseed = 42')
    expect(text).toContain('# Sample 2')
    expect(stringify(config)).toBe(original)
  })
  it("keeps malformed sample values visible in the original format", () => {
    const config = { preview_samples: ["broken"] }
    expect(formatConfigPreview(config)).toBe(stringify(config))
  })
})
