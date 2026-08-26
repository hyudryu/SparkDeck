import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { NodeSelector } from './NodeSelector'

const inventory = [
  { id: 'local', name: 'Coordinator', local: true, online: true, docker_ready: true },
  { id: 'spark-2', name: 'Studio Spark', online: true, docker_ready: true },
  { id: 'offline', name: 'Offline node', online: false, docker_ready: true },
]

afterEach(cleanup)

function Harness() {
  const [selected, setSelected] = useState(['local'])
  return <NodeSelector nodes={inventory} selectedIds={selected} onChange={setSelected} />
}

describe('NodeSelector', () => {
  it('shows the local default and supports accessible multi-selection', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    expect(screen.getByRole('checkbox', { name: /Coordinator/ })).toBeChecked()
    expect(screen.getByText('Target:').parentElement).toHaveTextContent('Target: Coordinator')

    await user.click(screen.getByRole('checkbox', { name: /Studio Spark/ }))
    expect(screen.getByRole('checkbox', { name: /Studio Spark/ })).toBeChecked()
    expect(screen.getByText('Targets:').parentElement).toHaveTextContent('Targets: Coordinator, Studio Spark')
    expect(screen.getByRole('checkbox', { name: /Offline node/ })).toBeDisabled()
  })

  it('keeps a required coordinator selected', () => {
    render(<NodeSelector nodes={inventory} selectedIds={['local']} onChange={() => undefined} requiredIds={['local']} />)

    const coordinator = screen.getByRole('checkbox', { name: /Coordinator/ })
    expect(coordinator).toBeChecked()
    expect(coordinator).toBeDisabled()
    expect(coordinator).toHaveAccessibleName(/Required/)
  })
})
