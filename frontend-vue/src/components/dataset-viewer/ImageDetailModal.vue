<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { X, CheckCircle, XCircle, Layers } from 'lucide-vue-next'
import type { COCOImage, COCOAnnotation, COCOCategory } from '@/types/coco'
import AnnotationOverlay from './AnnotationOverlay.vue'

const props = defineProps<{
  image: COCOImage
  annotations: COCOAnnotation[]
  categories: COCOCategory[]
  imageUrl: string
}>()

const emit = defineEmits<{ close: [] }>()

const { t } = useI18n()

const quality = computed(() => props.image.quality)

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      @click.self="emit('close')"
    >
      <div class="card w-full max-w-5xl max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between mb-4">
          <div>
            <h3 class="text-lg font-semibold text-white">{{ t('workflow.datasetViewer.modal.title') }}</h3>
            <p class="text-sm text-gray-400 truncate max-w-md" :title="image.file_name">{{ image.file_name }}</p>
          </div>
          <button class="text-gray-400 hover:text-white" @click="emit('close')">
            <X class="h-5 w-5" />
          </button>
        </div>

        <div class="grid gap-6 lg:grid-cols-3">
          <div class="lg:col-span-2 relative bg-black/30 rounded-lg overflow-hidden">
            <img :src="imageUrl" :alt="image.file_name" class="w-full h-auto" />
            <AnnotationOverlay
              :annotations="annotations"
              :categories="categories"
              :width="image.width"
              :height="image.height"
            />
          </div>

          <div class="space-y-4">
            <div class="p-4 bg-background-tertiary rounded-lg space-y-2">
              <p class="text-sm text-gray-400">
                {{ t('workflow.datasetViewer.modal.dimensions') }}:
                <span class="text-white font-mono">{{ image.width }}×{{ image.height }}</span>
              </p>
              <p class="text-sm text-gray-400 flex items-center gap-1">
                <Layers class="h-3.5 w-3.5" />
                {{ t('workflow.datasetViewer.modal.annotationsCount', { count: annotations.length }) }}
              </p>
              <p v-if="image.depth_source" class="text-sm text-gray-400">
                {{ t('workflow.datasetViewer.badges.depthSource', { source: image.depth_source }) }}
              </p>
            </div>

            <div v-if="quality" class="p-4 bg-background-tertiary rounded-lg space-y-2">
              <h4 class="font-medium text-white flex items-center gap-2">
                <component
                  :is="quality.overall_pass ? CheckCircle : XCircle"
                  :class="['h-4 w-4', quality.overall_pass ? 'text-green-400' : 'text-red-400']"
                />
                {{ t('workflow.datasetViewer.modal.quality.title') }}
              </h4>
              <p v-if="quality.perceptual_quality !== undefined" class="text-sm text-gray-400 flex justify-between">
                <span>{{ t('workflow.datasetViewer.modal.quality.perceptual') }}</span>
                <span class="text-white font-mono">{{ quality.perceptual_quality.toFixed(2) }}</span>
              </p>
              <p v-if="quality.anomaly_score !== undefined" class="text-sm text-gray-400 flex justify-between">
                <span>{{ t('workflow.datasetViewer.modal.quality.anomaly') }}</span>
                <span class="text-white font-mono">{{ quality.anomaly_score.toFixed(2) }}</span>
              </p>
              <p v-if="quality.overall_score !== undefined" class="text-sm text-gray-400 flex justify-between">
                <span>{{ t('workflow.datasetViewer.modal.quality.overall') }}</span>
                <span class="text-white font-mono">{{ quality.overall_score.toFixed(2) }}</span>
              </p>
              <p class="text-sm">
                <span :class="quality.overall_pass ? 'text-green-400' : 'text-red-400'">
                  {{ quality.overall_pass ? t('workflow.datasetViewer.modal.quality.pass') : t('workflow.datasetViewer.modal.quality.fail') }}
                </span>
              </p>
              <div v-if="quality.physics_violations?.length" class="pt-2 border-t border-gray-700">
                <p class="text-xs text-gray-500 mb-1">{{ t('workflow.datasetViewer.modal.quality.violations') }}</p>
                <ul class="text-xs text-yellow-400 space-y-1">
                  <li v-for="(v, idx) in quality.physics_violations" :key="idx">{{ v.description }}</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
