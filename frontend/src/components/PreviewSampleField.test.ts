// @vitest-environment jsdom
import { mount } from "@vue/test-utils"
import { defineComponent } from "vue"
import { describe, expect, it } from "vitest"
import PreviewSampleField from "./PreviewSampleField.vue"

const Input = defineComponent({
  props: { modelValue: { type: [String, Number], default: "" } }, emits: ["update:modelValue"],
  template: '<input :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />',
})

describe("shared PreviewSampleField t2i controls", () => {
  it("starts with one sample and keeps independent values on add/edit/delete", async () => {
    const wrapper = mount(PreviewSampleField, { global: { components: { ElInput: Input, ElInputNumber: Input } } })
    expect(wrapper.findAll(".preview-sample-item")).toHaveLength(1)
    await wrapper.get(".preview-sample-add").trigger("click")
    let samples = wrapper.emitted("update:samples")!.at(-1)![0] as string[]
    await wrapper.setProps({ samples })
    expect(wrapper.findAll(".preview-sample-item")).toHaveLength(2)
    await wrapper.findAll(".preview-sample-prompt input")[1]!.setValue("第二组提示词")
    samples = wrapper.emitted("update:samples")!.at(-1)![0] as string[]
    expect(JSON.parse(samples[0]!)).toMatchObject({ prompt: "", seed: 42 })
    expect(JSON.parse(samples[1]!)).toMatchObject({ prompt: "第二组提示词", width: 1024 })
    expect(JSON.parse(samples[1]!)).not.toHaveProperty("sampler")
    expect(JSON.parse(samples[1]!)).not.toHaveProperty("controlImages")
    await wrapper.setProps({ samples })
    await wrapper.findAll(".preview-sample-remove")[0]!.trigger("click")
    samples = wrapper.emitted("update:samples")!.at(-1)![0] as string[]
    expect(samples).toHaveLength(1)
    expect(JSON.parse(samples[0]!)).toMatchObject({ prompt: "第二组提示词" })
  })
})
