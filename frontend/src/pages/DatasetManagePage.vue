<script setup lang="ts">
import { onActivated, onBeforeUnmount, onDeactivated, ref } from "vue"
import { ElMessage } from "element-plus"
import { useI18n } from "vue-i18n"
import { useRouter } from "vue-router"
import { datasetsApi, type DatasetEntry, type DatasetOverview } from "../api/datasets"
import DatasetUploadDialog from "../components/dataset/DatasetUploadDialog.vue"

const POLL_INTERVAL_MS = 1500
const READY_REFRESH_MS = 15000

const { t } = useI18n()
const router = useRouter()

const rootPath = ref("")
const rootExists = ref(true)
const datasets = ref<DatasetEntry[]>([])
const loading = ref(false)
const refreshing = ref(false)
const rootDialogOpen = ref(false)
const rootInput = ref("")
const rootSaving = ref(false)
const createDialogOpen = ref(false)
const createName = ref("")
const creating = ref(false)
const uploadTarget = ref("")
let timer: number | undefined

function formatBytes(bytes: number | null | undefined) {
  if (bytes == null) return "-"
  if (bytes < 1024) return `${bytes} B`
  const units = ["KB", "MB", "GB", "TB"]
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`
}

function formatTime(iso: string | null | undefined) {
  if (!iso) return "-"
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

function overviewOf(entry: DatasetEntry): DatasetOverview | null {
  return entry.overview
}

function needsPoll(entry: DatasetEntry) {
  const overview = entry.overview
  if (!overview || overview.state !== "ready") return true
  const computedAt = overview.computed_at ? Date.parse(overview.computed_at) : 0
  return Date.now() - computedAt > READY_REFRESH_MS
}

async function pollOverviews() {
  const pending = datasets.value.filter(needsPoll)
  if (!pending.length) return
  await Promise.all(
    pending.map(async (entry) => {
      try {
        const data = await datasetsApi.overview(entry.name)
        entry.overview = data.overview
      } catch {
        entry.overview = { state: "error", file_count: null, captioned_count: null, total_bytes: null, updated_at: null, error: "request failed" }
      }
    }),
  )
}

function stopPolling() {
  window.clearInterval(timer)
  timer = undefined
}

async function load(silent = false) {
  if (silent) refreshing.value = true
  else loading.value = true
  try {
    const data = await datasetsApi.list()
    rootPath.value = data.root
    rootExists.value = data.exists
    datasets.value = data.datasets
    void pollOverviews()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetManage.msg.loadFail"))
  } finally {
    loading.value = false
    refreshing.value = false
  }
}

function openRootDialog() {
  rootInput.value = rootPath.value
  rootDialogOpen.value = true
}

async function saveRoot() {
  if (!rootInput.value.trim() || rootSaving.value) return
  rootSaving.value = true
  try {
    const data = await datasetsApi.updateRoot(rootInput.value.trim())
    rootPath.value = data.root
    rootExists.value = data.exists
    rootDialogOpen.value = false
    ElMessage.success(t("datasetManage.msg.rootSaved"))
    await load(true)
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetManage.msg.rootSaveFail"))
  } finally {
    rootSaving.value = false
  }
}

async function createDataset() {
  const name = createName.value.trim()
  if (!name || creating.value) return
  creating.value = true
  try {
    await datasetsApi.create(name)
    createDialogOpen.value = false
    createName.value = ""
    ElMessage.success(t("datasetManage.msg.created"))
    await load(true)
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetManage.msg.createFail"))
  } finally {
    creating.value = false
  }
}

function openTool(tool: "tagger" | "editor", entry: DatasetEntry) {
  void router.push({ path: `/dataset/${tool}`, query: { path: entry.path } })
}

function openUpload(entry: DatasetEntry) {
  uploadTarget.value = entry.name
}

function onUploaded() {
  void load(true)
}

onActivated(() => {
  void load()
  stopPolling()
  timer = window.setInterval(() => void pollOverviews(), POLL_INTERVAL_MS)
})
onDeactivated(stopPolling)
onBeforeUnmount(stopPolling)
</script>

<template>
  <div class="dataset-manage">
    <section class="dataset-manage-toolbar">
      <div class="dataset-manage-root">
        <span class="eyebrow">{{ t("datasetManage.rootLabel") }}</span>
        <code>{{ rootPath || "-" }}</code>
        <span v-if="!rootExists" class="dataset-manage-root-missing">{{ t("datasetManage.rootMissing") }}</span>
      </div>
      <div class="dataset-manage-actions">
        <button class="secondary-action" :disabled="loading || refreshing" @click="load(true)">{{ t("datasetManage.refresh") }}</button>
        <button class="secondary-action" @click="openRootDialog">{{ t("datasetManage.rootSettings") }}</button>
        <button class="primary-action" @click="createDialogOpen = true">{{ t("datasetManage.create") }}</button>
      </div>
    </section>

    <p v-if="!loading && !datasets.length" class="dataset-manage-empty">{{ t("datasetManage.empty") }}</p>

    <section v-else class="dataset-manage-grid">
      <article v-for="entry in datasets" :key="entry.name" class="dataset-card">
        <header class="dataset-card-header">
          <h2>{{ entry.name }}</h2>
          <span class="dataset-card-path" :title="entry.path">{{ entry.path }}</span>
        </header>
        <dl class="dataset-card-stats">
          <template v-if="overviewOf(entry)?.state === 'ready'">
            <div><dt>{{ t("datasetManage.files") }}</dt><dd>{{ overviewOf(entry)?.file_count }}</dd></div>
            <div><dt>{{ t("datasetManage.captioned") }}</dt><dd>{{ overviewOf(entry)?.captioned_count }}</dd></div>
            <div><dt>{{ t("datasetManage.size") }}</dt><dd>{{ formatBytes(overviewOf(entry)?.total_bytes) }}</dd></div>
            <div><dt>{{ t("datasetManage.updatedAt") }}</dt><dd>{{ formatTime(overviewOf(entry)?.updated_at) }}</dd></div>
          </template>
          <span v-else-if="overviewOf(entry)?.state === 'error'" class="dataset-card-pending">{{ t("datasetManage.statsError") }}</span>
          <span v-else class="dataset-card-pending">{{ t("datasetManage.computing") }}</span>
        </dl>
        <footer class="dataset-card-actions">
          <button class="primary-action" @click="openUpload(entry)">{{ t("datasetManage.upload") }}</button>
          <button class="secondary-action" @click="openTool('tagger', entry)">{{ t("datasetManage.openTagger") }}</button>
          <button class="secondary-action" @click="openTool('editor', entry)">{{ t("datasetManage.openEditor") }}</button>
        </footer>
      </article>
    </section>

    <ElDialog v-model="rootDialogOpen" :title="t('datasetManage.rootDialogTitle')" width="480px">
      <ElInput v-model="rootInput" :placeholder="t('datasetManage.rootDialogPlaceholder')" @keyup.enter="saveRoot" />
      <p class="dataset-manage-dialog-hint">{{ t("datasetManage.rootDialogHint") }}</p>
      <template #footer>
        <button class="secondary-action" @click="rootDialogOpen = false">{{ t("datasetManage.cancel") }}</button>
        <button class="primary-action" :disabled="rootSaving || !rootInput.trim()" @click="saveRoot">{{ t("datasetManage.save") }}</button>
      </template>
    </ElDialog>

    <DatasetUploadDialog
      :model-value="!!uploadTarget"
      :dataset-name="uploadTarget"
      @update:model-value="uploadTarget = ''"
      @uploaded="onUploaded"
    />

    <ElDialog v-model="createDialogOpen" :title="t('datasetManage.createDialogTitle')" width="480px">
      <ElInput v-model="createName" :placeholder="t('datasetManage.createPlaceholder')" @keyup.enter="createDataset" />
      <template #footer>
        <button class="secondary-action" @click="createDialogOpen = false">{{ t("datasetManage.cancel") }}</button>
        <button class="primary-action" :disabled="creating || !createName.trim()" @click="createDataset">{{ t("datasetManage.createConfirm") }}</button>
      </template>
    </ElDialog>
  </div>
</template>
