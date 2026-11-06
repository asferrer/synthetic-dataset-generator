<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import TabContainer from '@/components/common/TabContainer.vue'
import EffectsTab from './tabs/EffectsTab.vue'
import ObjectSizesTab from './tabs/ObjectSizesTab.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import { Settings, Ruler, ArrowLeft, ArrowRight } from 'lucide-vue-next'

const router = useRouter()
const { t } = useI18n()

const activeTab = ref('effects')
const effectsTabRef = ref<InstanceType<typeof EffectsTab> | null>(null)

const tabs = [
  { id: 'effects', label: t('workflow.configure.tabs.effects'), icon: Settings },
  { id: 'sizes', label: t('workflow.configure.tabs.objectSizes'), icon: Ruler },
]

function goBack() {
  router.push('/prepare')
}

// Effects/placement config lives as a local copy inside EffectsTab until its
// save() persists it to the store, so the Continue button must trigger it
// before navigating (the Object Sizes tab auto-saves each field, no save needed there).
function goNext() {
  if (activeTab.value === 'effects') {
    if (effectsTabRef.value?.hasValidationErrors) return
    effectsTabRef.value?.save()
    return
  }
  router.push('/generate')
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('workflow.configure.title') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('workflow.configure.subtitle') }}</p>
    </div>

    <TabContainer v-model="activeTab" :tabs="tabs">
      <template #effects>
        <EffectsTab ref="effectsTabRef" @navigate-next="() => router.push('/generate')" @navigate-back="goBack" />
      </template>
      <template #sizes>
        <ObjectSizesTab />
      </template>
    </TabContainer>

    <!-- Navigation Buttons -->
    <div class="flex justify-between pt-4">
      <BaseButton variant="outline" @click="goBack">
        <ArrowLeft class="h-5 w-5" />
        {{ t('common.actions.back') }}
      </BaseButton>
      <BaseButton @click="goNext">
        {{ t('common.actions.continue') }}
        <ArrowRight class="h-5 w-5" />
      </BaseButton>
    </div>
  </div>
</template>
