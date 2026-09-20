<script setup lang="ts">
import { ref, watch } from "vue"
import { ElMessage, ElMessageBox } from "element-plus"
import { useI18n } from "vue-i18n"
import { datasetsApi, type TrashBatch } from "../../api/datasets"

const props = defineProps<{ modelValue: boolean; datasetName: string }>()
const emit = defineEmits<{ "update:modelValue": [boolean]; changed: [] }>()
const { t } = useI18n()

const batches = ref<TrashBatch[]>([])
const loading = ref(false)
const busyId = ref("")

watch(
  () => props.modelValue,
  (open) => {
    if (open) void load()
  },
)

async function load() {
  loading.value = true
  try {
    const data = await datasetsApi.trash(props.datasetName)
    batches.value = data.batches
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetTrash.msg.loadFail"))
  } finally {
    loading.value = false
  }
}

function formatTime(iso: string | null) {
  if (!iso) return "-"
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

async function restore(batch: TrashBatch) {
  if (busyId.value) return
  busyId.value = batch.id
  try {
    const data = await datasetsApi.restoreTrash(props.datasetName, batch.id)
    if (data.restored.length) {
      ElMessage.success(t("datasetTrash.msg.restored", { n: data.restored.length }))
      emit("changed")
    }
    if (data.conflicts.length) ElMessage.warning(t("datasetTrash.msg.conflicts", { n: data.conflicts.length }))
    await load()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetTrash.msg.fail"))
  } finally {
    busyId.value = ""
  }
}

async function emptyBatch(batch: TrashBatch) {
  if (busyId.value) return
  try {
    await ElMessageBox.confirm(t("datasetTrash.confirmDeleteBatch", { count: batch.count }), { type: "warning" })
  } catch {
    return
  }
  busyId.value = batch.id
  try {
    await datasetsApi.emptyTrash(props.datasetName, batch.id)
    ElMessage.success(t("datasetTrash.msg.emptied"))
    await load()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetTrash.msg.fail"))
  } finally {
    busyId.value = ""
  }
}

async function emptyAll() {
  if (busyId.value || !batches.value.length) return
  try {
    await ElMessageBox.confirm(t("datasetTrash.confirmEmptyAll"), { type: "warning" })
  } catch {
    return
  }
  busyId.value = "*"
  try {
    await datasetsApi.emptyTrash(props.datasetName)
    ElMessage.success(t("datasetTrash.msg.emptyDone"))
    await load()
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : t("datasetTrash.msg.fail"))
  } finally {
    busyId.value = ""
  }
}
</script>

<template>
  <ElDialog
    :model-value="modelValue"
    :title="t('datasetTrash.title', { name: datasetName })"
    width="560px"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div v-loading="loading" class="trash-dialog">
      <p v-if="!batches.length && !loading" class="trash-empty">{{ t("datasetTrash.empty") }}</p>
      <article v-for="batch in batches" :key="batch.id" class="trash-batch">
        <header class="trash-batch-header">
          <span>{{ t("datasetTrash.batchInfo", { count: batch.count, time: formatTime(batch.deleted_at) }) }}</span>
          <span class="trash-batch-actions">
            <button class="secondary-action" :disabled="!!busyId" @click="restore(batch)">{{ t("datasetTrash.restore") }}</button>
            <button class="danger-action" :disabled="!!busyId" @click="emptyBatch(batch)">{{ t("datasetTrash.deleteBatch") }}</button>
          </span>
        </header>
        <ul class="trash-batch-paths">
          <li v-for="path in batch.paths" :key="path">{{ path }}</li>
        </ul>
      </article>
    </div>
    <template #footer>
      <button class="secondary-action" :disabled="!!busyId" @click="emit('update:modelValue', false)">{{ t("datasetTrash.close") }}</button>
      <button class="danger-action" :disabled="!!busyId || !batches.length" @click="emptyAll">{{ t("datasetTrash.emptyAll") }}</button>
    </template>
  </ElDialog>
</template>
