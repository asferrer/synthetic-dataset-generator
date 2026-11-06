/**
 * Deterministic color palette for COCO category visualization.
 * Same category id always maps to the same color across the app.
 */
const CATEGORY_PALETTE = [
  '#ef4444', '#3b82f6', '#22c55e', '#eab308', '#a855f7',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
  '#06b6d4', '#f43f5e', '#8b5cf6', '#10b981', '#facc15',
]

/** Returns a stable hex color for a category, based on its position in the categories list. */
export function getCategoryColor(categoryId: number, categories: Array<{ id: number }>): string {
  const idx = categories.findIndex((c) => c.id === categoryId)
  const key = idx >= 0 ? idx : categoryId
  return CATEGORY_PALETTE[key % CATEGORY_PALETTE.length]
}
