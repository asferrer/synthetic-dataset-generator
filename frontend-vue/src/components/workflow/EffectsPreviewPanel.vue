<script setup lang="ts">
/**
 * EffectsPreviewPanel - single-image preview of the current effects configuration.
 *
 * Picks one sample background and one sample object from the directories already
 * selected in the Prepare step, composes them via POST /augment/compose (single,
 * not batch), and shows the resulting image next to a refresh button.
 */
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  listFiles,
  listDirectories,
  composeSingle,
  getImageUrl,
  type ComposeSingleResult,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import { Sparkles, RefreshCw } from 'lucide-vue-next'
import type { EffectsConfig } from '@/types/api'

const props = defineProps<{
  effectsConfig: EffectsConfig
  backgroundsDir: string | null
  objectsDir: string | null
  outputDir: string | null
}>()

const { t } = useI18n()

const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<ComposeSingleResult | null>(null)

const sampleBackgroundPath = ref<string | null>(null)
const sampleObjectPath = ref<string | null>(null)
const sampleObjectClass = ref<string | null>(null)

const IMAGE_PATTERNS = ['*.jpg', '*.jpeg', '*.png']

const canPreview = computed(() => !!props.backgroundsDir && !!props.objectsDir)

const previewImageUrl = computed(() => {
  // Each run writes to a fresh timestamped output_path already, but add a cache-busting
  // param too in case the browser aggressively caches by URL alone.
  const cacheBust = result.value ? `&t=${Date.now()}` : ''
  return result.value?.success ? `${getImageUrl(result.value.output_path)}${cacheBust}` : null
})

/** Flatten the nested EffectsConfig into the flat shape /augment/compose expects. */
function flattenForSingleCompose(cfg: EffectsConfig): Record<string, unknown> {
  return {
    color_intensity: cfg.color_correction.color_intensity,
    blur_strength: cfg.blur_matching.blur_strength,
    underwater_intensity: cfg.underwater.underwater_intensity,
    caustics_intensity: cfg.caustics.caustics_intensity,
    shadow_opacity: cfg.shadows.shadow_opacity,
    lighting_type: cfg.lighting.lighting_type,
    lighting_intensity: cfg.lighting.lighting_intensity,
    motion_blur_probability: cfg.motion_blur.motion_blur_probability,
    water_clarity: cfg.underwater.water_clarity,
    water_color: cfg.underwater.water_color,
  }
}

function enabledEffectsList(cfg: EffectsConfig): string[] {
  const enabled: Array<[boolean, string]> = [
    [cfg.color_correction.enabled, 'color_correction'],
    [cfg.blur_matching.enabled, 'blur_matching'],
    [cfg.lighting.enabled, 'lighting'],
    [cfg.underwater.enabled, 'underwater'],
    [cfg.motion_blur.enabled, 'motion_blur'],
    [cfg.shadows.enabled, 'shadows'],
    [cfg.caustics.enabled, 'caustics'],
    [cfg.edge_smoothing.enabled, 'edge_smoothing'],
  ]
  return enabled.filter(([on]) => on).map(([, name]) => name)
}

async function resolveSampleBackground(): Promise<string | null> {
  if (sampleBackgroundPath.value) return sampleBackgroundPath.value
  if (!props.backgroundsDir) return null
  const files = await listFiles(props.backgroundsDir, IMAGE_PATTERNS)
  sampleBackgroundPath.value = files[0] || null
  return sampleBackgroundPath.value
}

async function resolveSampleObject(): Promise<{ path: string; className: string } | null> {
  if (sampleObjectPath.value && sampleObjectClass.value) {
    return { path: sampleObjectPath.value, className: sampleObjectClass.value }
  }
  if (!props.objectsDir) return null
  const classDirs = await listDirectories(props.objectsDir)
  if (classDirs.length === 0) return null
  const files = await listFiles(classDirs[0], IMAGE_PATTERNS)
  if (files.length === 0) return null
  sampleObjectPath.value = files[0]
  sampleObjectClass.value = classDirs[0].split(/[\\/]/).pop() || 'object'
  return { path: sampleObjectPath.value, className: sampleObjectClass.value }
}

async function runPreview() {
  if (!canPreview.value) return
  loading.value = true
  error.value = null

  try {
    const [bgPath, obj] = await Promise.all([resolveSampleBackground(), resolveSampleObject()])
    if (!bgPath || !obj) {
      error.value = t('workflow.configure.effectsPanel.preview.noSamplesFound')
      return
    }

    const outDir = (props.outputDir || '/app/output').replace(/\/+$/, '')
    const outputPath = `${outDir}/_preview/preview_${Date.now()}.jpg`

    const response = await composeSingle({
      background_path: bgPath,
      objects: [{ image_path: obj.path, class_name: obj.className }],
      effects: enabledEffectsList(props.effectsConfig),
      effects_config: flattenForSingleCompose(props.effectsConfig),
      output_path: outputPath,
      save_annotations: false,
    })

    result.value = response
    if (!response.success) {
      error.value = response.error || t('workflow.configure.effectsPanel.preview.failed')
    }
  } catch (e: any) {
    error.value = e.message || t('workflow.configure.effectsPanel.preview.failed')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="card p-6 space-y-4">
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-lg font-semibold text-white flex items-center gap-2">
          <Sparkles class="h-5 w-5 text-primary" />
          {{ t('workflow.configure.effectsPanel.preview.title') }}
        </h3>
        <p class="text-sm text-gray-400">{{ t('workflow.configure.effectsPanel.preview.description') }}</p>
      </div>
      <BaseButton
        variant="outline"
        :disabled="!canPreview"
        :title="!canPreview ? t('workflow.configure.effectsPanel.preview.noSources') : undefined"
        :loading="loading"
        @click="runPreview"
      >
        <RefreshCw v-if="result" class="h-4 w-4" />
        <Sparkles v-else class="h-4 w-4" />
        {{ result ? t('workflow.configure.effectsPanel.preview.refresh') : t('workflow.configure.effectsPanel.preview.button') }}
      </BaseButton>
    </div>

    <AlertBox v-if="!canPreview" type="info" :title="t('workflow.configure.effectsPanel.preview.noSources')" />
    <AlertBox v-else-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <div v-if="loading" class="flex justify-center py-8">
      <LoadingSpinner :message="t('workflow.configure.effectsPanel.preview.loading')" />
    </div>

    <div v-else-if="previewImageUrl" class="space-y-2">
      <img :src="previewImageUrl" alt="Effects preview" class="w-full max-w-lg mx-auto rounded-lg" />
      <p class="text-xs text-gray-500 text-center">
        {{ t('workflow.configure.effectsPanel.preview.objectsPlaced') }}: {{ result?.objects_placed }} ·
        {{ t('workflow.configure.effectsPanel.preview.processingTime') }}: {{ result?.processing_time_ms.toFixed(0) }}ms
      </p>
      <p v-if="result && !result.is_valid" class="text-xs text-yellow-400 text-center">
        {{ t('workflow.configure.effectsPanel.preview.rejected') }}
      </p>
    </div>
  </div>
</template>
