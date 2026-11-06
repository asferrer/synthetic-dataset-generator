<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { getGeneratedDatasetMetadata, getGeneratedDatasetCoco, getImageUrl, type GeneratedDatasetMetadata } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import ImageDetailModal from '@/components/dataset-viewer/ImageDetailModal.vue'
import { ArrowLeft, Image, Layers, Tag, HardDrive, ChevronLeft, ChevronRight, CheckCircle, XCircle } from 'lucide-vue-next'
import type { COCODataset, COCOImage as COCOImageType } from '@/types/coco'

const route = useRoute()
const router = useRouter()
const { t } = useI18n()

const jobId = computed(() => route.params.jobId as string)

const loading = ref(true)
const error = ref<string | null>(null)
const metadata = ref<GeneratedDatasetMetadata | null>(null)
const coco = ref<COCODataset | null>(null)
const selectedImage = ref<COCOImageType | null>(null)

const page = ref(1)
const pageSize = 24

const totalPages = computed(() => {
  if (!coco.value) return 1
  return Math.max(1, Math.ceil(coco.value.images.length / pageSize))
})

const pagedImages = computed(() => {
  if (!coco.value) return []
  const start = (page.value - 1) * pageSize
  return coco.value.images.slice(start, start + pageSize)
})

function annotationsFor(imageId: number) {
  return coco.value?.annotations.filter((a) => a.image_id === imageId) || []
}

function imageUrl(image: COCOImageType): string {
  if (!metadata.value) return ''
  const dir = metadata.value.images_dir.replace(/\/+$/, '')
  return getImageUrl(`${dir}/${image.file_name}`)
}

async function load() {
  loading.value = true
  error.value = null
  try {
    const [metaResult, cocoResult] = await Promise.all([
      getGeneratedDatasetMetadata(jobId.value),
      getGeneratedDatasetCoco(jobId.value),
    ])
    metadata.value = metaResult
    coco.value = cocoResult
  } catch (e: any) {
    error.value = e.message || t('workflow.datasetViewer.loadFailed')
  } finally {
    loading.value = false
  }
}

function goBack() {
  router.push('/generate')
}

onMounted(load)
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center gap-4">
      <BaseButton variant="outline" size="sm" @click="goBack">
        <ArrowLeft class="h-4 w-4" />
        {{ t('workflow.datasetViewer.back') }}
      </BaseButton>
      <div>
        <h2 class="text-2xl font-bold text-white">{{ t('workflow.datasetViewer.title') }}</h2>
        <p class="text-sm text-gray-400">{{ t('workflow.datasetViewer.subtitle') }}</p>
      </div>
    </div>

    <div v-if="loading" class="flex justify-center py-16">
      <LoadingSpinner size="lg" :message="t('workflow.datasetViewer.loading')" />
    </div>

    <AlertBox v-else-if="error" type="error" :title="error" />

    <template v-else-if="metadata && coco">
      <div class="grid gap-4 sm:grid-cols-4">
        <div class="card p-4 flex items-center gap-3">
          <Image class="h-6 w-6 text-primary" />
          <div>
            <p class="text-xl font-bold text-white">{{ metadata.num_images }}</p>
            <p class="text-xs text-gray-400">{{ t('workflow.datasetViewer.stats.images') }}</p>
          </div>
        </div>
        <div class="card p-4 flex items-center gap-3">
          <Layers class="h-6 w-6 text-primary" />
          <div>
            <p class="text-xl font-bold text-white">{{ metadata.num_annotations }}</p>
            <p class="text-xs text-gray-400">{{ t('workflow.datasetViewer.stats.annotations') }}</p>
          </div>
        </div>
        <div class="card p-4 flex items-center gap-3">
          <Tag class="h-6 w-6 text-primary" />
          <div>
            <p class="text-xl font-bold text-white">{{ metadata.num_categories }}</p>
            <p class="text-xs text-gray-400">{{ t('workflow.datasetViewer.stats.categories') }}</p>
          </div>
        </div>
        <div class="card p-4 flex items-center gap-3">
          <HardDrive class="h-6 w-6 text-primary" />
          <div>
            <p class="text-xl font-bold text-white">{{ metadata.file_size_mb.toFixed(1) }} MB</p>
            <p class="text-xs text-gray-400">{{ t('workflow.datasetViewer.stats.size') }}</p>
          </div>
        </div>
      </div>

      <EmptyState
        v-if="coco.images.length === 0"
        :icon="Image"
        :title="t('workflow.datasetViewer.empty')"
      />

      <template v-else>
        <div class="grid gap-3 grid-cols-2 sm:grid-cols-4 lg:grid-cols-6">
          <button
            v-for="img in pagedImages"
            :key="img.id"
            class="relative aspect-square rounded-lg overflow-hidden bg-background-tertiary ring-1 ring-gray-700 hover:ring-primary transition-all"
            @click="selectedImage = img"
          >
            <img :src="imageUrl(img)" :alt="img.file_name" loading="lazy" class="w-full h-full object-cover" />
            <span
              v-if="img.quality"
              class="absolute top-1 right-1 rounded-full p-0.5 bg-black/60"
              :title="img.quality.overall_pass ? t('workflow.datasetViewer.badges.qualityPass') : t('workflow.datasetViewer.badges.qualityFail')"
            >
              <CheckCircle v-if="img.quality.overall_pass" class="h-3.5 w-3.5 text-green-400" />
              <XCircle v-else class="h-3.5 w-3.5 text-red-400" />
            </span>
            <span class="absolute bottom-0 inset-x-0 bg-black/60 text-[10px] text-gray-300 px-1 py-0.5 truncate">
              {{ annotationsFor(img.id).length }} {{ t('common.units.objects') }}
            </span>
          </button>
        </div>

        <div class="flex items-center justify-center gap-4">
          <BaseButton variant="outline" size="sm" :disabled="page <= 1" @click="page--">
            <ChevronLeft class="h-4 w-4" />
            {{ t('workflow.datasetViewer.pagination.prev') }}
          </BaseButton>
          <span class="text-sm text-gray-400">
            {{ t('workflow.datasetViewer.pagination.page', { current: page, total: totalPages }) }}
          </span>
          <BaseButton variant="outline" size="sm" :disabled="page >= totalPages" @click="page++">
            {{ t('workflow.datasetViewer.pagination.next') }}
            <ChevronRight class="h-4 w-4" />
          </BaseButton>
        </div>
      </template>

      <ImageDetailModal
        v-if="selectedImage"
        :image="selectedImage"
        :annotations="annotationsFor(selectedImage.id)"
        :categories="coco.categories"
        :image-url="imageUrl(selectedImage)"
        @close="selectedImage = null"
      />
    </template>
  </div>
</template>
