<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { startExtraction, getExtractionStatus, listDatasets } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  Scissors,
  Play,
  CheckCircle,
  Settings,
  Wand2,
  Copy,
  Box,
} from 'lucide-vue-next'
import type { DatasetInfo, ExtractionJobStatus } from '@/types/api'

const uiStore = useUiStore()
const { t } = useI18n()

const loading = ref(false)
const extracting = ref(false)
const datasets = ref<DatasetInfo[]>([])
const cocoJsonPath = ref('')
const imagesDir = ref('')
const outputDir = ref('/app/output/extracted')
const minSize = ref(32)
const includeMasks = ref(true)
const useSam3 = ref(false)
const padding = ref(10)
const deduplicate = ref(false)
const showAdvanced = ref(false)
const currentJob = ref<ExtractionJobStatus | null>(null)
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

async function loadDatasets() {
  loading.value = true
  try {
    datasets.value = await listDatasets()
  } catch (e) {
    // Ignore
  } finally {
    loading.value = false
  }
}

async function startExtractionJob() {
  if (!cocoJsonPath.value || !imagesDir.value) {
    uiStore.showError(t('tools.objectExtraction.missingInput'), t('tools.objectExtraction.missingInputMsg'))
    return
  }

  extracting.value = true
  error.value = null
  currentJob.value = null

  try {
    const response = await startExtraction({
      coco_json_path: cocoJsonPath.value,
      images_dir: imagesDir.value,
      output_dir: outputDir.value,
      min_size: minSize.value,
      include_masks: includeMasks.value,
      use_sam3: useSam3.value,
      padding: padding.value,
      deduplicate: deduplicate.value,
    })

    uiStore.showSuccess(t('tools.objectExtraction.notifications.started'), t('tools.objectExtraction.notifications.startedMsg', { id: response.job_id.slice(0, 8) }))
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || t('tools.objectExtraction.extractionFailedMsg')
    uiStore.showError(t('tools.objectExtraction.notifications.failed'), error.value ?? undefined)
    extracting.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getExtractionStatus(jobId)
    currentJob.value = job

    if (job.status === 'completed') {
      stopPolling()
      extracting.value = false
      uiStore.showSuccess(t('tools.objectExtraction.notifications.completed'), t('tools.objectExtraction.notifications.completedMsg'))
    } else if (job.status === 'failed') {
      stopPolling()
      extracting.value = false
      error.value = job.errors[0] || t('tools.objectExtraction.notifications.failed')
      uiStore.showError(t('tools.objectExtraction.notifications.failed'), error.value ?? undefined)
    }
  } catch (e) {
    // Ignore polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
}

onUnmounted(() => {
  stopPolling()
})

// Load datasets on mount
loadDatasets()
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('tools.objectExtraction.title') }}</h2>
      <p class="mt-1 text-gray-400">
        {{ t('tools.objectExtraction.subtitle') }}
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Input -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.objectExtraction.sourceDatasetTitle') }}</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="cocoJsonPath"
            :label="t('tools.objectExtraction.cocoJsonPath')"
            placeholder="/app/datasets/annotations.json"
            :show-files="true"
            file-pattern="*.json"
            path-mode="input"
          />
          <DirectoryBrowser
            v-model="imagesDir"
            :label="t('tools.objectExtraction.imagesDirectory')"
            placeholder="/app/datasets/images"
            path-mode="input"
          />
        </div>
      </div>

      <!-- Output -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.objectExtraction.outputSettingsTitle') }}</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="outputDir"
            :label="t('tools.objectExtraction.outputDirectory')"
            placeholder="/app/output/extracted"
            path-mode="output"
          />

          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>{{ t('tools.objectExtraction.minObjectSize') }}</span>
              <span class="text-white">{{ minSize }}px</span>
            </label>
            <input
              type="range"
              v-model.number="minSize"
              min="8"
              max="256"
              step="8"
              class="w-full accent-primary"
            />
          </div>

          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              v-model="includeMasks"
              class="h-5 w-5 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
            />
            <div>
              <span class="text-gray-300">{{ t('tools.objectExtraction.includeMasks') }}</span>
              <p class="text-xs text-gray-500">{{ t('tools.objectExtraction.includeMasksDesc') }}</p>
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- Advanced Options -->
    <div class="card p-6">
      <button
        class="flex items-center justify-between w-full"
        @click="showAdvanced = !showAdvanced"
      >
        <h3 class="text-lg font-semibold text-white flex items-center gap-2">
          <Settings class="h-5 w-5" />
          {{ t('tools.objectExtraction.advancedOptionsTitle') }}
        </h3>
        <span class="text-gray-400">{{ showAdvanced ? '▲' : '▼' }}</span>
      </button>

      <div v-if="showAdvanced" class="mt-6 space-y-6">
        <div class="grid gap-6 md:grid-cols-2">
          <!-- SAM3 Option -->
          <div class="p-4 bg-background-tertiary rounded-lg">
            <label class="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                v-model="useSam3"
                class="h-5 w-5 rounded border-gray-600 bg-background-primary text-primary focus:ring-primary"
              />
              <div class="flex-1">
                <div class="flex items-center gap-2">
                  <Wand2 class="h-4 w-4 text-yellow-400" />
                  <span class="text-gray-300 font-medium">{{ t('tools.objectExtraction.useSam3') }}</span>
                </div>
                <p class="text-xs text-gray-500 mt-1">
                  {{ t('tools.objectExtraction.useSam3Desc') }}
                </p>
              </div>
            </label>
          </div>

          <!-- Deduplicate Option -->
          <div class="p-4 bg-background-tertiary rounded-lg">
            <label class="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                v-model="deduplicate"
                class="h-5 w-5 rounded border-gray-600 bg-background-primary text-primary focus:ring-primary"
              />
              <div class="flex-1">
                <div class="flex items-center gap-2">
                  <Copy class="h-4 w-4 text-blue-400" />
                  <span class="text-gray-300 font-medium">{{ t('tools.objectExtraction.deduplicate') }}</span>
                </div>
                <p class="text-xs text-gray-500 mt-1">
                  {{ t('tools.objectExtraction.deduplicateDesc') }}
                </p>
              </div>
            </label>
          </div>
        </div>

        <!-- Padding Slider -->
        <div class="p-4 bg-background-tertiary rounded-lg">
          <div class="flex items-center gap-2 mb-3">
            <Box class="h-4 w-4 text-green-400" />
            <label class="text-sm text-gray-300 font-medium">{{ t('tools.objectExtraction.paddingLabel') }}</label>
          </div>
          <div class="flex items-center gap-4">
            <input
              type="range"
              v-model.number="padding"
              min="0"
              max="50"
              step="5"
              class="flex-1 accent-primary"
            />
            <span class="text-white font-mono w-16 text-right">{{ padding }}px</span>
          </div>
          <p class="text-xs text-gray-500 mt-2">
            {{ t('tools.objectExtraction.paddingDesc') }}
          </p>
        </div>
      </div>
    </div>

    <!-- Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">{{ t('tools.objectExtraction.progressTitle') }}</h3>

      <div class="flex items-center gap-4 mb-4">
        <component
          :is="currentJob.status === 'completed' ? CheckCircle : Scissors"
          :class="[
            'h-8 w-8',
            currentJob.status === 'completed' ? 'text-green-400' : 'text-primary animate-pulse'
          ]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">
            {{ currentJob.status === 'running' ? t('tools.objectExtraction.extracting') : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">{{ t('tools.objectExtraction.jobIdLabel', { id: currentJob.job_id.slice(0, 8) }) }}...</p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-primary transition-all"
          :style="{ width: `${currentJob.progress}%` }"
        />
      </div>
    </div>

    <!-- Start Button -->
    <div class="flex justify-end">
      <BaseButton
        :loading="extracting"
        :disabled="extracting || !cocoJsonPath || !imagesDir"
        @click="startExtractionJob"
        size="lg"
      >
        <Scissors class="h-5 w-5" />
        {{ t('tools.objectExtraction.actions.startExtraction') }}
      </BaseButton>
    </div>
  </div>
</template>
