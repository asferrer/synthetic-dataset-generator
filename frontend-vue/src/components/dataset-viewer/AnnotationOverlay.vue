<script setup lang="ts">
import { computed } from 'vue'
import type { COCOAnnotation, COCOCategory } from '@/types/coco'
import { getCategoryColor } from '@/lib/categoryColors'

const props = defineProps<{
  annotations: COCOAnnotation[]
  categories: COCOCategory[]
  width: number
  height: number
}>()

function categoryName(categoryId: number): string {
  return props.categories.find((c) => c.id === categoryId)?.name || `#${categoryId}`
}

function color(categoryId: number): string {
  return getCategoryColor(categoryId, props.categories)
}

// Font size scales with image resolution so labels stay legible at any zoom level
const labelFontSize = computed(() => Math.max(12, Math.round(props.width / 60)))
const strokeWidth = computed(() => Math.max(2, Math.round(props.width / 400)))
</script>

<template>
  <svg
    :viewBox="`0 0 ${width} ${height}`"
    preserveAspectRatio="none"
    class="absolute inset-0 h-full w-full pointer-events-none"
  >
    <g v-for="ann in annotations" :key="ann.id">
      <rect
        :x="ann.bbox[0]"
        :y="ann.bbox[1]"
        :width="ann.bbox[2]"
        :height="ann.bbox[3]"
        fill="none"
        :stroke="color(ann.category_id)"
        :stroke-width="strokeWidth"
      />
      <rect
        :x="ann.bbox[0]"
        :y="Math.max(0, ann.bbox[1] - labelFontSize * 1.4)"
        :width="categoryName(ann.category_id).length * labelFontSize * 0.62 + 8"
        :height="labelFontSize * 1.4"
        :fill="color(ann.category_id)"
      />
      <text
        :x="ann.bbox[0] + 4"
        :y="Math.max(labelFontSize, ann.bbox[1] - labelFontSize * 0.3)"
        :font-size="labelFontSize"
        fill="#ffffff"
        font-weight="600"
      >
        {{ categoryName(ann.category_id) }}
      </text>
    </g>
  </svg>
</template>
