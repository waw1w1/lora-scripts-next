// @vitest-environment jsdom
import { flushPromises, mount } from "@vue/test-utils"
import { afterEach, expect, it, vi } from "vitest"
import { i18n } from "../i18n"
import { enginesApi } from "../api/engines"
import TrainingSelector from "./TrainingSelector.vue"

afterEach(() => { vi.restoreAllMocks() })

function mountSelector() {
  return mount(TrainingSelector, {
    props: { model: "anima", engine: "kohya", target: "lora" },
    global: { plugins: [i18n] },
  })
}

it("labels every engine with its install state (#443)", async () => {
  vi.spyOn(enginesApi, "list").mockResolvedValue([
    { id: "kohya", state: "ready", featureEnabled: true },
    { id: "anima-fast", state: "not_installed", featureEnabled: true },
    { id: "musubi", state: "broken", featureEnabled: true },
    { id: "ai-toolkit", state: "installing", featureEnabled: true },
  ] as never)
  const wrapper = mountSelector()
  await flushPromises()

  const options = wrapper.findAll("select")[1].findAll("option").map((option) => option.text())
  expect(options.find((text) => text.startsWith("Kohya-ss"))).toContain("已就绪")
  expect(options.find((text) => text.startsWith("Anima Fast"))).toContain("未安装")
  expect(options.find((text) => text.startsWith("Musubi-Tuner"))).toContain("需修复")
  expect(options.find((text) => text.startsWith("AI Toolkit"))).toContain("安装中")
  // No status reported: the plain name stays, no misleading label.
  expect(options.find((text) => text.startsWith("DiffSynth-Studio"))).toBe("DiffSynth-Studio")
  wrapper.unmount()
})

it("keeps unsupported engines disabled and explains the two reasons apart (#443)", async () => {
  vi.spyOn(enginesApi, "list").mockResolvedValue([
    { id: "kohya", state: "ready", featureEnabled: true },
    // Supported for Anima but missing: must show up in the actionable hint.
    { id: "anima-fast", state: "not_installed", featureEnabled: true },
    // DiffSynth is listed but unsupported for the Anima model.
    { id: "diffsynth", state: "ready", featureEnabled: true },
  ] as never)
  const wrapper = mountSelector()
  await flushPromises()

  const engineSelect = wrapper.findAll("select")[1]
  const unsupported = engineSelect.findAll("option").filter((option) => option.attributes("disabled") !== undefined)
  expect(unsupported.map((option) => option.text().split(" · ")[0])).toContain("DiffSynth-Studio")

  const hints = wrapper.findAll(".selector-hint").map((hint) => hint.text())
  expect(hints.some((text) => text.includes("该模型暂不支持"))).toBe(true)
  expect(hints.some((text) => text.includes("尚未安装或需要修复"))).toBe(true)
  wrapper.unmount()
})

it("ignores engine status failures instead of breaking the selector (#443)", async () => {
  vi.spyOn(enginesApi, "list").mockRejectedValue(new Error("offline"))
  const wrapper = mountSelector()
  await flushPromises()

  const options = wrapper.findAll("select")[1].findAll("option").map((option) => option.text())
  expect(options).toContain("Kohya-ss")
  wrapper.unmount()
})
