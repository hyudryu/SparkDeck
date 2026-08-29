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

  it('allows an asynchronously discovered required coordinator to be selected', async () => {
    const user = userEvent.setup()
    function RequiredHarness() {
      const [selected, setSelected] = useState<string[]>([])
      return <NodeSelector nodes={inventory} selectedIds={selected} onChange={setSelected} requiredIds={['local']} />
    }
    render(<RequiredHarness />)

    const coordinator = screen.getByRole('checkbox', { name: /Coordinator/ })
    expect(coordinator).not.toBeDisabled()
    await user.click(coordinator)
    expect(coordinator).toBeChecked()
    expect(coordinator).toBeDisabled()
  })

  it('labels the controller and first replicated target in worker context', () => {
    render(<NodeSelector nodes={inventory} selectedIds={['spark-2', 'local']} onChange={() => undefined} localLabel="Controller" primaryId="spark-2" />)

    expect(screen.getByRole('checkbox', { name: /Controller/ })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: /Studio Spark.*Primary/ })).toBeChecked()
    expect(screen.getByText('Targets:').parentElement).toHaveTextContent('Targets: Studio Spark, Controller · Primary: Studio Spark')
  })

  it('explains why a running node is disabled', () => {
    render(<NodeSelector nodes={[{
      id: 'spark-4', name: 'Spark Four', online: true, docker_ready: false, selectable: false,
      status_message: "SparkDeck's service user cannot access Docker. Add this user to the docker group, then restart the user session.",
    }]} selectedIds={[]} onChange={() => undefined} />)

    const checkbox = screen.getByRole('checkbox', { name: /Spark Four/ })
    expect(checkbox).toBeDisabled()
    expect(checkbox.closest('label')).toHaveAttribute('title', expect.stringContaining('service user cannot access Docker'))
    expect(screen.getByText(/service user cannot access Docker/)).toBeInTheDocument()
  })
})
