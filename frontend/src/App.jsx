import { useState } from 'react'
import './App.css'

// Single combined backend (generation + EMI). Override at build time with VITE_API_BASE.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const TOPICS = [
  'Healthcare reform',
  'Immigration policy',
  'Tax legislation',
  'Climate and energy policy',
  'Gun control',
  'National defense spending',
  'Education funding',
  'Social Security',
  'Infrastructure investment',
  'Trade and tariffs',
  'AI regulation'
]

const PARTIES = ['Democratic', 'Republican', 'Independent']

const METHODS = [
  { key: 'w2v_emi', name: 'Word2Vec' },
  { key: 'bert_emi', name: 'BERT' },
  { key: 'gpt2_emi', name: 'GPT-2' },
]

// Classify an EMI score as evidence-driven (high), intuition-driven (low), or neutral.
function classifyEmi(v) {
  if (v === null || v === undefined) return { label: 'No Data', kind: 'na' }
  if (v > 0.1) return { label: 'High · Evidence', kind: 'high' }
  if (v < -0.1) return { label: 'Low · Intuition', kind: 'low' }
  return { label: 'Neutral', kind: 'neutral' }
}

export default function App() {
  const [topic, setTopic] = useState(TOPICS[0])
  const [party, setParty] = useState(PARTIES[0])
  const [creativity, setCreativity] = useState(0.9)
  const [support, setSupport] = useState(true)

  const [speech, setSpeech] = useState('')
  const [meta, setMeta] = useState(null) // {party, topic, support} captured at generation time
  const [emi, setEmi] = useState(null)
  const [genLoading, setGenLoading] = useState(false)
  const [emiLoading, setEmiLoading] = useState(false)
  const [error, setError] = useState('')

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  async function generate() {
    setError('')
    setEmi(null)
    setSpeech('')
    setGenLoading(true)
    try {
      // The model prompt is "...in support of {topic}", so encode opposition in the topic.
      const stance = support ? topic : `opposing ${topic}`
      const params = new URLSearchParams({
        party,
        topic: stance,
        max_new_tokens: '200',
        temperature: String(creativity),
      })
      const res = await fetch(`${API_BASE}/generate_speech?${params}`, { method: 'POST' })
      if (!res.ok) throw new Error(`Generation failed (${res.status})`)
      const data = await res.json()
      setSpeech(data.speech || '')
      setMeta({ party, topic, support })
    } catch (e) {
      setError(e.message)
    } finally {
      setGenLoading(false)
    }
  }

  async function calculateEmi() {
    setError('')
    setEmiLoading(true)
    try {
      const params = new URLSearchParams({ w2v: 'true', bert: 'true', gpt2: 'true', text: speech })
      const res = await fetch(`${API_BASE}/calculate_emi?${params}`, { method: 'POST' })
      if (!res.ok) throw new Error(`EMI calculation failed (${res.status})`)
      setEmi(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setEmiLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="flagrule" aria-hidden="true" />

      <header className="masthead">
        <div className="edition-bar">
          <span>Vol. I · No. 1</span>
          <span className="edition-mid">{today}</span>
          <span>Washington Edition</span>
        </div>
        <h1 className="title">The Congressional Record</h1>
        <div className="subbar">
          <span>Price · Free</span>
          <span className="tagline">All the Rhetoric Fit to Print</span>
          <span>Est. MMXXVI</span>
        </div>
        <p className="dek">
          A machine-authored congressional address engine. Compose a floor speech with a
          fine-tuned GPT-2 model, then submit it to the <strong>Evidence-Motivation Index</strong>
          {' '}— a three-method analysis measuring whether the rhetoric leans on evidence or instinct.
        </p>
      </header>

      <main className="grid">
        {/* LEFT — controls */}
        <section className="col-left newsprint-texture">
          <span className="kicker red">The Composing Desk</span>

          <div className="field">
            <label htmlFor="topic">Topic</label>
            <select id="topic" value={topic} onChange={(e) => setTopic(e.target.value)}>
              {TOPICS.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="party">Party</label>
            <select id="party" value={party} onChange={(e) => setParty(e.target.value)}>
              {PARTIES.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Stance</label>
            <div className="segmented">
              <button
                className={support ? 'seg active support' : 'seg'}
                aria-pressed={support}
                onClick={() => setSupport(true)}
              >
                Support
              </button>
              <button
                className={!support ? 'seg active oppose' : 'seg'}
                aria-pressed={!support}
                onClick={() => setSupport(false)}
              >
                Oppose
              </button>
            </div>
          </div>

          <div className="field">
            <label htmlFor="creativity">
              Creativity <span className="mono-val">[ {creativity.toFixed(1)} ]</span>
            </label>
            <input
              id="creativity"
              type="range"
              min="0.2"
              max="1.5"
              step="0.1"
              value={creativity}
              onChange={(e) => setCreativity(Number(e.target.value))}
            />
            <div className="rangeends">
              <span>Focused</span>
              <span>Creative</span>
            </div>
          </div>

          <button className="btn primary" onClick={generate} disabled={genLoading}>
            {genLoading ? 'Setting Type…' : 'Generate Speech'}
          </button>
        </section>

        {/* RIGHT — article + analysis */}
        <section className="col-right">
          {error && <div className="error">{error}</div>}

          {!speech && !genLoading && !error && (
            <div className="placeholder">
              <div>
                <span className="kicker">Press Wire</span>
                <p>Awaiting dispatch. Set the desk and run the press.</p>
              </div>
            </div>
          )}

          {genLoading && (
            <div className="placeholder">
              <div>
                <span className="kicker red">On The Wire</span>
                <p>The press is running…</p>
              </div>
            </div>
          )}

          {speech && meta && (
            <article className="article">
              <span className="kicker blue">Floor Address</span>
              <h2 className="headline">
                {meta.party} Remarks {meta.support ? 'in Support of' : 'Opposing'} {meta.topic}
              </h2>
              <div className="byline">
                By the {meta.party} Desk · Special to the Record · {today}
              </div>
              <p className="article-body">{speech}</p>

              <button className="btn secondary" onClick={calculateEmi} disabled={emiLoading}>
                {emiLoading ? 'Analyzing…' : 'Calculate EMI Scores'}
              </button>
            </article>
          )}

          {emi && (
            <section className="emi">
              <span className="kicker red">Editorial Analysis · The Index</span>
              <p className="emi-help">
                Positive ▸ evidence-driven · Negative ▸ intuition-driven · Near zero ▸ neutral
              </p>
              {METHODS.map((m) => {
                const v = emi[m.key]
                const c = classifyEmi(v)
                const pct = Math.min(Math.abs(v ?? 0), 1) * 50
                return (
                  <div className="emi-row" key={m.key}>
                    <div className="emi-head">
                      <span className="emi-name">{m.name}</span>
                      <span className={`badge ${c.kind}`}>{c.label}</span>
                      <span className="emi-val">{v == null ? '——' : v.toFixed(3)}</span>
                    </div>
                    <div className="emi-track">
                      <div className="emi-center" />
                      {v != null && (
                        <div
                          className={`emi-fill ${v >= 0 ? 'pos' : 'neg'}`}
                          style={v >= 0 ? { left: '50%', width: `${pct}%` } : { right: '50%', width: `${pct}%` }}
                        />
                      )}
                    </div>
                  </div>
                )
              })}
            </section>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Printed in Washington · No Trees Harmed</span>
        <span>GPT-2 · BERT · Word2Vec</span>
        <span>Edition Vol. 1.0</span>
      </footer>
    </div>
  )
}
