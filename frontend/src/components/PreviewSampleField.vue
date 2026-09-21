<!-- Adapted from wochenlong/feat/ai-toolkit-klein PreviewSampleField.vue: t2i fields only. -->
<script setup lang="ts">
import { computed } from "vue"

interface PreviewSample {
  prompt: string
  width: number
  height: number
  seed: number
  guidance_scale: number
  sample_steps: number
}

const DEFAULT_SAMPLE_SETTINGS = {
  width: 1024,
  height: 1024,
  seed: 42,
  guidance_scale: 4,
  sample_steps: 20,
}

const props = defineProps<{
  samples?: string[]
  disabled?: boolean
}>()

const emit = defineEmits<{
  "update:samples": [value: string[]]
}>()

function parseSample(value: string): PreviewSample | undefined {
  try {
    const parsed = JSON.parse(value) as Partial<PreviewSample>
    if (typeof parsed.prompt !== "string") return undefined
    return {
      prompt: parsed.prompt,
      width: typeof parsed.width === "number" ? parsed.width : DEFAULT_SAMPLE_SETTINGS.width,
      height: typeof parsed.height === "number" ? parsed.height : DEFAULT_SAMPLE_SETTINGS.height,
      seed: typeof parsed.seed === "number" ? parsed.seed : DEFAULT_SAMPLE_SETTINGS.seed,
      guidance_scale: typeof parsed.guidance_scale === "number" ? parsed.guidance_scale : DEFAULT_SAMPLE_SETTINGS.guidance_scale,
      sample_steps: typeof parsed.sample_steps === "number" ? parsed.sample_steps : DEFAULT_SAMPLE_SETTINGS.sample_steps,
    }
  } catch {
    return undefined
  }
}

const sampleValues = computed(() => {
  const parsed = (props.samples || []).map(parseSample).filter((item): item is PreviewSample => Boolean(item))
  if (parsed.length) return parsed
  return [{
    prompt: "",
    ...DEFAULT_SAMPLE_SETTINGS,
  }]
})

function emitSamples(samples: PreviewSample[]) {
  emit("update:samples", samples.map((sample) => JSON.stringify(sample)))
}

function cloneSamples() {
  return sampleValues.value.map((sample) => ({ ...sample }))
}

function updatePrompt(sampleIndex: number, prompt: string) {
  const samples = cloneSamples()
  samples[sampleIndex].prompt = prompt
  emitSamples(samples)
}

function updateSetting(sampleIndex: number, key: keyof typeof DEFAULT_SAMPLE_SETTINGS, value: string | number | undefined) {
  if (value === undefined) return
  const samples = cloneSamples()
  samples[sampleIndex][key] = value as never
  emitSamples(samples)
}

function addSample() {
  emitSamples([...sampleValues.value, { prompt: "", ...DEFAULT_SAMPLE_SETTINGS }])
}

function removeSample(index: number) {
  if (sampleValues.value.length <= 1) return
  emitSamples(sampleValues.value.filter((_, sampleIndex) => sampleIndex !== index))
}

</script>

<template>
  <div class="preview-sample-control">
    <div class="preview-sample-heading">
      <strong>预览样例</strong>
      <span>每组使用独立的提示词、尺寸、种子和推理参数。</span>
    </div>
    <div v-for="(sample, sampleIndex) in sampleValues" :key="sampleIndex" class="preview-sample-item">
      <div class="preview-sample-item-heading">
        <span><code>samples[{{ sampleIndex }}]</code></span>
        <button v-if="sampleValues.length > 1" type="button" class="preview-sample-remove" :disabled="disabled" @click="removeSample(sampleIndex)">
          删除样例
        </button>
      </div>
      <div class="preview-sample-prompt">
        <span class="preview-sample-label"><code>prompt</code></span>
        <el-input
          :model-value="sample.prompt"
          type="textarea"
          :rows="2"
          :disabled="disabled"
          placeholder="提示词"
          @update:model-value="updatePrompt(sampleIndex, $event)"
        />
      </div>
      <div class="preview-sample-settings">
        <label class="preview-sample-setting">
          <span class="preview-sample-label"><code>width</code></span>
          <el-input-number :model-value="sample.width" :min="64" :step="64" :disabled="disabled" @update:model-value="updateSetting(sampleIndex, 'width', $event)" />
        </label>
        <label class="preview-sample-setting">
          <span class="preview-sample-label"><code>height</code></span>
          <el-input-number :model-value="sample.height" :min="64" :step="64" :disabled="disabled" @update:model-value="updateSetting(sampleIndex, 'height', $event)" />
        </label>
        <label class="preview-sample-setting">
          <span class="preview-sample-label"><code>seed</code></span>
          <el-input-number :model-value="sample.seed" :min="0" :step="1" :disabled="disabled" @update:model-value="updateSetting(sampleIndex, 'seed', $event)" />
        </label>
        <label class="preview-sample-setting">
          <span class="preview-sample-label"><code>guidance_scale</code></span>
          <el-input-number :model-value="sample.guidance_scale" :min="1" :max="30" :step="0.1" :precision="1" :disabled="disabled" @update:model-value="updateSetting(sampleIndex, 'guidance_scale', $event)" />
        </label>
        <label class="preview-sample-setting">
          <span class="preview-sample-label"><code>sample_steps</code></span>
          <el-input-number :model-value="sample.sample_steps" :min="1" :max="300" :step="1" :disabled="disabled" @update:model-value="updateSetting(sampleIndex, 'sample_steps', $event)" />
        </label>
      </div>
    </div>
    <button type="button" class="preview-sample-add" :disabled="disabled" @click="addSample">
      + 添加样例
    </button>
  </div>

</template>
