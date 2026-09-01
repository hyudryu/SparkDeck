import type { RuntimeKind } from '../api/types'

const VLLM_KV_CACHE_DTYPES = [
  'auto',
  'bfloat16',
  'float16',
  'fp8',
  'fp8_ds_mla',
  'fp8_e4m3',
  'fp8_e5m2',
  'fp8_inc',
  'fp8_per_token_head',
  'int4_per_token_head',
  'int8_per_token_head',
  'nvfp4',
  'nvfp4_4over6',
  'nvfp4_ds_mla',
  'turboquant_3bit_nc',
  'turboquant_4bit_nc',
  'turboquant_k3v4_nc',
  'turboquant_k8v4',
] as const

const SGLANG_KV_CACHE_DTYPES = [
  'auto',
  'bf16',
  'bfloat16',
  'fp8_e4m3',
  'fp8_e5m2',
  'mxfp8',
  'nvfp4',
  'fp4_mx_block16',
] as const

const optionsFor = (runtime: RuntimeKind): readonly string[] => (
  runtime === 'sglang' ? SGLANG_KV_CACHE_DTYPES
    : runtime === 'vllm' ? VLLM_KV_CACHE_DTYPES
      : []
)

interface KvCacheDtypeSelectProps {
  runtime: RuntimeKind
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}

export function KvCacheDtypeSelect({ runtime, value, disabled, onChange }: KvCacheDtypeSelectProps) {
  const options = optionsFor(runtime)
  const customValue = value && !options.includes(value) ? value : undefined

  return <select disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)}>
    <option value="">Auto / unset</option>
    {customValue && <option value={customValue}>{customValue} (current)</option>}
    {options.map((option) => <option key={option} value={option}>{option === 'auto' ? 'auto (explicit)' : option}</option>)}
  </select>
}
