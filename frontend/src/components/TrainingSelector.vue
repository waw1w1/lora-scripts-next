<script setup lang="ts">
import { computed, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { enginesApi } from "../api/engines"
import type { EngineRuntimeState } from "../engines/catalog"
import {
  TRAINING_ENGINES,
  TRAINING_MODELS,
  TRAINING_TARGETS,
  isEngineSupported,
  isTargetSupported,
  type TrainingEngine,
  type TrainingModel,
  type TrainingTarget,
} from "../training/modules"

const model = defineModel<TrainingModel>("model", { required: true })
const engine = defineModel<TrainingEngine>("engine", { required: true })
const target = defineModel<TrainingTarget>("target", { required: true })
const { t } = useI18n()

const unsupportedEngines = computed(() => TRAINING_ENGINES.filter((item) => !isEngineSupported(model.value, item)))
const unsupportedTargets = computed(() => TRAINING_TARGETS.filter((item) => !isTargetSupported(model.value, engine.value, item)))

// Install state per engine, so the dropdown answers "is this engine installed?"
// separately from "does this model support it?" (the latter greys an option out).
const engineStates = ref<Partial<Record<TrainingEngine, EngineRuntimeState>>>({})
const STATE_KEYS: Partial<Record<EngineRuntimeState, string>> = {
  ready: "ready",
  installed_unverified: "installedUnverified",
  not_installed: "notInstalled",
  unknown: "unknown",
  installing: "installing",
  auditing: "installing",
  broken: "broken",
  disabled: "disabled",
  coming_soon: "comingSoon",
}

onMounted(async () => {
  try {
    const statuses = await enginesApi.list()
    const next: Partial<Record<TrainingEngine, EngineRuntimeState>> = {}
    for (const status of statuses) next[status.id] = status.state
    engineStates.value = next
  } catch {
    // Install state is advisory here: fall back to plain engine names.
  }
})

function engineOptionLabel(item: TrainingEngine): string {
  const name = t(`training.selector.engines.${item}`)
  const state = engineStates.value[item]
  const key = state ? STATE_KEYS[state] : undefined
  return key ? `${name} · ${t(`training.selector.engineStates.${key}`)}` : name
}

const missingEngines = computed(() => TRAINING_ENGINES.filter((item) => {
  if (!isEngineSupported(model.value, item)) return false
  const state = engineStates.value[item]
  return state === "not_installed" || state === "broken"
}))
</script>

<template>
  <div class="workbench-selector">
    <div class="selector-card">
      <h2>{{ t("training.selector.group") }}</h2>
      <div class="selector-grid">
        <label>{{ t("training.selector.model") }}
          <select v-model="model">
            <option v-for="item in TRAINING_MODELS" :key="item" :value="item">{{ t(`training.selector.models.${item}`) }}</option>
          </select>
        </label>
        <label>{{ t("training.selector.engine") }}
          <select v-model="engine">
            <option v-for="item in TRAINING_ENGINES" :key="item" :value="item" :disabled="!isEngineSupported(model, item)">{{ engineOptionLabel(item) }}</option>
          </select>
        </label>
        <p v-if="unsupportedEngines.length" class="selector-hint">{{ t("training.selector.unsupportedForModel", { list: unsupportedEngines.map((item) => t(`training.selector.engines.${item}`)).join(" / ") }) }}</p>
        <p v-if="missingEngines.length" class="selector-hint">{{ t("training.selector.engineMissing", { list: missingEngines.map((item) => t(`training.selector.engines.${item}`)).join(" / ") }) }}</p>
      </div>
    </div>
    <div class="selector-card">
      <h2>{{ t("training.selector.target") }}</h2>
      <div class="segmented" role="group" :aria-label="t('training.selector.targetType')">
        <button v-for="item in TRAINING_TARGETS" :key="item" :class="{ active: target === item }" :disabled="!isTargetSupported(model, engine, item)" @click="target = item">{{ t(`training.selector.targets.${item}`) }}</button>
      </div>
      <p v-if="unsupportedTargets.length" class="selector-hint">{{ t("training.selector.unsupportedForModel", { list: unsupportedTargets.map((item) => t(`training.selector.targets.${item}`)).join(" / ") }) }}</p>
    </div>
  </div>
</template>
