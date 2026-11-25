// Runs desktop/plugin.js against a stand-in @hermes/plugin-sdk with no build step: a custom
// node:module loader hands back virtual SDK/React modules, and React renders to an inspected element tree.

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { pathToFileURL } from 'node:url'

const scenario = process.argv[2] || 'all'

const REACT_SOURCE = `
export function useState(initial) {
  const s = globalThis.__STORE
  const i = s.Index++
  if (!(i in s.slots)) s.slots[i] = typeof initial === 'function' ? initial() : initial
  const set = next => { s.slots[i] = typeof next === 'function' ? next(s.slots[i]) : next }
  return [s.slots[i], set]
}
`

const JSX_SOURCE = `
export function jsx(type, props, key) { return { $$el: true, type, props: props || {}, key } }
export const jsxs = jsx
`

const HERMES_SDK_SOURCE = `
export const SETTINGS_PLUGINS_AREA = 'settings.plugins'
export const PALETTE_AREA = 'command-palette.items'
export const pluginSettingsHref = id => '/settings?tab=plugins&plugin=' + encodeURIComponent(id)
export const host = {
  state: { profile: { get: () => globalThis.__STORE.profile } },
  navigate: to => globalThis.__STORE.navigated.push(to),
}
export const queryClient = {
  setQueryData: (key, data) => { globalThis.__STORE.queryState.data = data },
  invalidateQueries: opts => globalThis.__STORE.invalidated.push(opts),
  removeQueries: opts => globalThis.__STORE.removed.push(opts),
}
export const useValue = cell => cell.get()
export function useQuery(opts) {
  globalThis.__STORE.lastQueryOpts = opts
  const q = globalThis.__STORE.queryState
  return { ...q, refetch: () => { q.refetchCalls++ } }
}
export function useMutation(opts) {
  const m = {
    isPending: false,
    mutate: arg => {
      globalThis.__STORE.mutations.push({ opts, arg })
      return Promise.resolve().then(() => opts.mutationFn(arg)).then(
        res => opts.onSuccess?.(res),
        err => opts.onError?.(err),
      )
    }
  }
  return m
}
`

const loaderSource = `
const stubs = new Map([
  ['@hermes/plugin-sdk', ${JSON.stringify(HERMES_SDK_SOURCE)}],
  ['react', ${JSON.stringify(REACT_SOURCE)}],
  ['react/jsx-runtime', ${JSON.stringify(JSX_SOURCE)}],
])
export async function resolve(specifier, context, nextResolve) {
  if (stubs.has(specifier)) return { url: 'virtual:' + specifier, shortCircuit: true }
  return nextResolve(specifier, context)
}
export async function load(url, context, nextLoad) {
  if (url.startsWith('virtual:')) {
    const key = url.slice('virtual:'.length)
    return { format: 'module', source: stubs.get(key), shortCircuit: true }
  }
  return nextLoad(url, context)
}
`

globalThis.__STORE = {
  slots: [],
  Index: 0,
  navigated: [],
  invalidated: [],
  removed: [],
  queryState: { data: undefined, error: null, isLoading: false, isFetching: false, refetchCalls: 0 },
  mutations: [],
  profile: 'default'
}

register('data:text/javascript,' + encodeURIComponent(loaderSource))

const pluginUrl = new URL('../desktop/plugin.js', import.meta.url).href
const mod = await import(pluginUrl)
const plugin = mod.default

function expand(node) {
  if (node == null || typeof node === 'boolean') return []
  if (typeof node === 'string' || typeof node === 'number') return [String(node)]
  if (Array.isArray(node)) return node.flatMap(expand)
  if (node.$$el) {
    if (typeof node.type === 'function') return expand(node.type(node.props))
    const kids = expand(node.props.children)
    return [{ tag: node.type, props: node.props, children: kids }]
  }
  return []
}

function render(component) {
  globalThis.__STORE.Index = 0
  return expand({ $$el: true, type: component, props: {} })
}

function text(nodes) {
  return nodes.map(n => typeof n === 'string' ? n : text(n.children)).join(' ')
}

function findAll(nodes, pred, out = []) {
  for (const n of nodes) {
    if (typeof n === 'object' && n !== null) {
      if (pred(n)) out.push(n)
      findAll(n.children, pred, out)
    }
  }
  return out
}

function makeCtx(restImpl) {
  const pages = []
  const contributions = []
  const events = new Map()
  const disposers = []
  return {
    pages,
    contributions,
    events,
    disposers,
    ctx: {
      pluginId: 'decider',
      rest: restImpl,
      registerSettingsPage: page => pages.push(page),
      register: c => contributions.push(c),
      onEvent: (topic, fn) => events.set(topic, fn),
      onDispose: fn => disposers.push(fn)
    }
  }
}

const sampleState = {
  mode: 'decider',
  mode_label: 'System-1 Decider',
  route_default: 'FastSub',
  routes: [
    { name: 'FastSub', description: 'Fast worker', summary: 'openai-codex/gpt-5.4-mini', decider_eligible: true },
    { name: 'VisualProwess', description: 'Screenshots', summary: 'solstice/gemini-4-hydrogen', decider_eligible: true },
    { name: 'UltraFastPaid', description: 'Paid per token', summary: 'openrouter/deepseek', decider_eligible: false }
  ],
  decider: {
    enabled: true,
    url: 'http://127.0.0.1:11434/v1/systemone',
    primary_model: 'pplx-decider:27b-nvfp4',
    has_api_key: false,
    allow_fallback: false,
    fallback_url: '',
    fallback_model: 'jev-latest',
    has_fallback_api_key: false,
    specialist_threshold: 0.7,
    confidence_threshold: 0.6,
    timeout: 18,
    excluded_options: ['UltraFastPaid']
  },
  decisions: [
    { timestamp: 1, action: 'upgraded', route: 'VisualProwess', needs_specialist: 0.99, confidence: 0.99, latency_ms: 812, fallback_used: false, goal: 'Check screenshot.png' }
  ]
}

const flush = () => new Promise(r => setTimeout(r, 0))

const scenarios = {
  async registration() {
    const h = makeCtx(async () => sampleState)
    plugin.register(h.ctx)
    assert.equal(plugin.id, 'decider')
    assert.equal(h.pages.length, 1)
    assert.equal(h.pages[0].id, 'main')
    assert.equal(h.pages[0].title, 'System-1 Decider')
    assert.equal(h.contributions.length, 1)
    assert.equal(h.contributions[0].area, 'command-palette.items')
    h.contributions[0].data.run()
    assert.equal(globalThis.__STORE.navigated.at(-1), '/settings?tab=plugins&plugin=decider')

    assert.ok(h.events.has('plugin.decider.updated'))
    assert.ok(h.events.has('gateway.ready'))
    h.events.get('plugin.decider.updated')()
    assert.deepEqual(globalThis.__STORE.invalidated.at(-1), { queryKey: ['decider'] })
    h.events.get('gateway.ready')()
    assert.deepEqual(globalThis.__STORE.removed.at(-1), { queryKey: ['decider'] })

    h.disposers.forEach(fn => fn())
    await assert.rejects(() => mod.api.state(), /not loaded/)
  },

  async 'path-guard'() {
    assert.equal(mod.assertBackendPath('/state?limit=30'), '/state?limit=30')
    assert.equal(mod.assertBackendPath('/decider'), '/decider')
    assert.equal(mod.assertBackendPath('/probe'), '/probe')
    assert.equal(mod.assertBackendPath('/decisions'), '/decisions')
    assert.throws(() => mod.assertBackendPath('/routes'), /Unsupported backend path/)
    assert.throws(() => mod.assertBackendPath('http://evil.example/state'), /Unsupported backend path/)
  },

  async 'fallback-default-off'() {
    const f = mod.toForm({})
    assert.equal(f.allow_fallback, false, 'Secondary fallback must default to false in UI')
  },

  async 'happy-path'() {
    const calls = []
    const h = makeCtx(async (path, opts) => {
      calls.push({ path, opts })
      if (path === '/decider') {
        return { ...sampleState, decider: { ...sampleState.decider, ...opts.body } }
      }
      if (path === '/probe') {
        return {
          ok: true,
          effective_route: 'VisualProwess',
          candidate_route: 'VisualProwess',
          upgraded: true,
          needs_specialist: 0.99,
          confidence: 0.99,
          latency_ms: 640,
          model: 'pplx-decider:27b-nvfp4',
          fallback_used: false
        }
      }
      if (path === '/decisions') return { ok: true, cleared: 1 }
      return sampleState
    })
    plugin.register(h.ctx)
    globalThis.__STORE.slots = []
    globalThis.__STORE.queryState = { data: structuredClone(sampleState), error: null, isLoading: false, isFetching: false, refetchCalls: 0 }

    let tree = render(mod.DeciderSettings)
    const body = text(tree)
    assert.match(body, /System-1 Decider/)
    assert.match(body, /Routes it may pick \(2\/3\)/)
    assert.doesNotMatch(body, /Needs Model routes/, 'decider mode must not display a mode warning')

    // Toggle Paid route on -> PUT /decider with excluded_options: []
    const paidSwitch = findAll(tree, n => n.props?.['aria-label'] === 'Let the Decider pick UltraFastPaid')[0]
    assert.ok(paidSwitch)
    assert.equal(paidSwitch.props.checked, false)
    paidSwitch.props.onChange({ currentTarget: { checked: true } })
    await flush()
    assert.equal(calls.at(-1).path, '/decider')
    assert.deepEqual(calls.at(-1).opts.body.excluded_options, [])

    // Run live probe -> POST /probe
    tree = render(mod.DeciderSettings)
    const testBtn = findAll(tree, n => n.tag === 'button' && text(n.children).trim() === 'Test')[0]
    testBtn.props.onClick()
    await flush()
    assert.equal(calls.at(-1).path, '/probe')
    tree = render(mod.DeciderSettings)
    assert.match(text(tree), /640 ms/)

    // Clear decisions -> DELETE /decisions
    const clearBtn = findAll(tree, n => n.tag === 'button' && text(n.children).trim() === 'Clear')[0]
    clearBtn.props.onClick()
    await flush()
    assert.equal(calls.at(-1).path, '/decisions')
    h.disposers.forEach(fn => fn())
  },

  async 'field-width'() {
    const h = makeCtx(async () => sampleState)
    plugin.register(h.ctx)
    globalThis.__STORE.slots = []
    const withFallback = structuredClone(sampleState)
    withFallback.decider.allow_fallback = true
    globalThis.__STORE.queryState = { data: withFallback, error: null, isLoading: false, isFetching: false, refetchCalls: 0 }

    const tree = render(mod.DeciderSettings)
    const rem = label => {
      const input = findAll(tree, n => n.props?.['aria-label'] === label)[0]
      assert.ok(input, `missing input ${label}`)
      const m = /^(\d+(?:\.\d+)?)rem$/.exec(String(input.props.style?.width || ''))
      return m ? Number(m[1]) : 0
    }
    for (const label of ['Endpoint', 'Fallback endpoint', 'Fallback API key']) {
      assert.ok(rem(label) >= 16, `${label} width ${rem(label)}rem cannot show a URL/key`)
    }
    for (const label of ['Specialist threshold', 'Route confidence', 'Timeout']) {
      assert.ok(rem(label) > 0 && rem(label) <= 6, `${label} width ${rem(label)}rem is not sized for a short number`)
    }
    // Verify write-only payload omission when blank and inclusion when typed
    const blankPayload = mod.toPayload(mod.toForm(withFallback.decider))
    assert.equal('fallback_api_key' in blankPayload, false)
    const keyedPayload = mod.toPayload({ ...mod.toForm(withFallback.decider), fallback_api_key: '  sk-fb-test  ' })
    assert.equal(keyedPayload.fallback_api_key, 'sk-fb-test')
    h.disposers.forEach(fn => fn())
  }
}

const toRun = scenario === 'all' ? Object.keys(scenarios) : [scenario]
for (const name of toRun) {
  await scenarios[name]()
}
console.log(`ok (${toRun.join(', ')})`)
