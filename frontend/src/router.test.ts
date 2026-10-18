// @vitest-environment jsdom
import { describe, expect, it } from "vitest"
import router from "./router"

describe("generic plugin routes", () => {
  it("resolves plugin settings and logical artifact detail routes before the catch-all", () => {
    expect(router.getRoutes().find((route) => route.path === "/settings/plugins")?.redirect).toBe("/settings/advanced")
    expect(router.resolve("/settings/plugins/sample-plugin").name).toBe("plugin-settings")
    expect(router.resolve("/plugins/sample-plugin/artifacts/artifact-1").name).toBe("plugin-artifact-detail")
  })

  it("does not contain duplicate settings engine routes", () => {
    expect(router.getRoutes().filter((route) => route.path === "/settings/engines")).toHaveLength(1)
  })

  it("redirects legacy Gradio tag editor URLs to the Vue dataset editor", () => {
    for (const path of ["/tageditor.html", "/native-tageditor.html", "/dataset-editor.html"]) {
      const record = router.getRoutes().find((route) => route.path === path)
      expect(record?.redirect, path).toBe("/dataset/editor")
    }
  })
})
