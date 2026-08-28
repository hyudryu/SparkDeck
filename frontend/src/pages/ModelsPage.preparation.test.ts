import { describe, expect, it } from 'vitest'
import type { StorageTransferPreflightTarget } from '../api/types'
import { recipePreparationRequiredBytes } from './ModelsPage'

function option(overrides: Partial<StorageTransferPreflightTarget>) {
  return {
    node_id: 'node',
    node_name: 'Node',
    eligible: true,
    has_required_weights: false,
    has_model_cache: false,
    free_bytes: 100,
    required_free_bytes: 40,
    download_required_free_bytes: 80,
    transfer_after_download_required_free_bytes: 60,
    ...overrides,
  } satisfies StorageTransferPreflightTarget
}

describe('recipe preparation capacity labels', () => {
  it('shows Hub download capacity for cached missing revisions', () => {
    expect(recipePreparationRequiredBytes(option({
      has_model_cache: true,
      has_required_weights: false,
    }), true)).toBe(80)
  })

  it('shows NAS transfer capacity for cache-empty targets', () => {
    expect(recipePreparationRequiredBytes(option({
      has_model_cache: false,
    }), true)).toBe(40)
  })

  it('uses the conservative maximum before a no-source seed is chosen', () => {
    expect(recipePreparationRequiredBytes(option({}), false)).toBe(80)
  })
})
