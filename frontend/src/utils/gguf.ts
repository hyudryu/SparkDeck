import type { CatalogModel } from '../api/types'

export type GgufQuantization = NonNullable<CatalogModel['quantizations']>[number]

export type GgufArtifactOption = {
  key: string
  filename: string
  quantization: string
  weightSize?: number | null
  files: Array<{ filename: string }>
}

// One artifact option per downloadable GGUF file group (a quantization, or
// one shard set inside it). Shard groups surface under their first shard's
// filename, matching what the Hub publishes in the repo file listing.
export function ggufArtifactOptions(quantizations: NonNullable<CatalogModel['quantizations']>): GgufArtifactOption[] {
  return quantizations.flatMap((variant) => {
    const artifacts = variant.artifacts?.length
      ? variant.artifacts
      : variant.files.some((file) => file.filename.toLocaleLowerCase().endsWith('.gguf'))
        ? [{
          filename: variant.files.find((file) => file.filename.toLocaleLowerCase().endsWith('.gguf'))!.filename,
          files: variant.files,
          weight_size_bytes: variant.weight_size_bytes,
        }]
        : []
    return artifacts.map((artifact) => ({
      key: `${variant.name}\u0000${artifact.filename}`,
      filename: artifact.filename,
      quantization: variant.name,
      weightSize: artifact.weight_size_bytes,
      files: artifact.files,
    }))
  })
}

// A quantization counts as downloaded only when at least one node holds
// every file of its artifact group — compare repo-relative names exactly.
// Sets are per node: shards split across caches do not add up to one
// usable copy.
export function artifactFilesDownloaded(
  files: Array<{ filename: string }>,
  cachedFileSets: ReadonlyArray<ReadonlySet<string>>,
) {
  return files.length > 0
    && cachedFileSets.some((set) => files.every((file) => set.has(file.filename)))
}
