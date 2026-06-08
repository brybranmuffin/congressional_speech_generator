// Interpretations for the 27 EMI score combinations (Word2Vec, BERT, GPT-2),
// each classified High / Neutral / Low.
// Key = `${w2v}${bert}${gpt2}` using H / N / L (in that order).
// Source: emi_combinations_interpretations.md

export const INTERPRETATIONS = {
  HHH: {
    title: 'Genuine evidence-based across all layers',
    text: 'Evidence-based at every layer — empirical vocabulary, deployed in genuinely epistemic contexts, with an evidence-first sequential structure. The cleanest case of evidence-based rhetoric: the speaker cites data, frames claims as empirical, and builds toward conclusions through reasoned argumentation. All three methods independently confirm the same finding, giving maximal confidence.',
  },
  NNN: {
    title: 'Procedurally neutral or balanced',
    text: 'No strong commitment to either evidence- or intuition-based rhetoric at any level. Likely procedural (introducing legislation, yielding time, acknowledging colleagues) or substantively mixed. All three methods agree there is no dominant epistemic register — common in floor management, ceremonial remarks, or balanced policy discussion.',
  },
  LLL: {
    title: 'Genuine intuition-based across all layers',
    text: 'Intuition-based at every layer — belief and emotional vocabulary, contextually deployed in stance-taking ways, with an argument that unfolds through appeals to values, feelings, or shared assumptions rather than empirical reasoning. The speaker emphasizes conviction, moral urgency, or persuasion over data. All three methods agree, giving maximal confidence.',
  },
  HLL: {
    title: 'Vocabulary masking — evidence words used non-epistemically',
    text: 'Contains evidence-related vocabulary but uses it in non-epistemic ways. The contextual models detect that words like "evidence," "data," or "facts" are deployed rhetorically rather than substantively — through negation ("there is no evidence"), accusation ("the so-called facts"), or as decorative intensifiers. The argument structure is intuition-driven despite the surface vocabulary.',
  },
  LHH: {
    title: 'Evidence reasoning without seed vocabulary',
    text: 'Makes evidence-based arguments without the standard seed vocabulary. The speaker reasons empirically — citing authorities, presenting quantitative claims, building structured arguments — but with words the seed dictionary does not capture. A limitation of the dictionary approach: the original 49 seed words capture only a subset of how empirical argument actually appears.',
  },
  HHL: {
    title: 'Empirical content lacking argumentative structure',
    text: 'Uses evidence vocabulary in contextually appropriate ways at the sentence level, but does not organize those sentences into a coherent argumentative structure. The speaker may present scattered facts, list statistics without drawing inferences, or reference data in isolation. The components of evidence-based reasoning are present but not assembled into a sequential argument.',
  },
  HLH: {
    title: 'Imitative empirical structure without contextual evidence',
    text: 'An unusual pattern: evidence vocabulary appears alongside an evidence-shaped argumentative arc, but the sentence-level contextual usage is not evidence-based. The speaker makes structured arguments that mimic evidence-based form, deploying seed vocabulary non-epistemically while organizing the speech premise-to-conclusion — sophisticated persuasion that imitates empirical reasoning structurally without substantive content.',
  },
  LHL: {
    title: 'Analytical style without empirical substance',
    text: 'Uses analytical, formally precise language sentence by sentence — adopting the tone of evidence-based reasoning — but contains no evidence vocabulary and builds no evidence-shaped argument. The speaker may use technical policy language, bureaucratic precision, or a formal analytical register without making empirical claims. Analytical style without empirical substance.',
  },
  LLH: {
    title: 'Structured argument with intuition-based content',
    text: 'Intuition-based vocabulary and contextual usage, but an evidence-shaped argumentative structure. The speaker builds a logically structured argument from intuition-based premises — assertions, beliefs, and value claims arranged premise-to-conclusion without empirical grounding. The rhetorical form of evidence-based reasoning without its substantive content.',
  },
  NHH: {
    title: 'Evidence-driven beyond the seed vocabulary',
    text: 'Vocabulary balance is mixed, but both contextual models agree the speech is evidence-based at the sentence and argument levels. The speaker makes empirical claims using language that does not heavily favor either seed dictionary. The seed vocabulary misses how this speaker reasons empirically, but the underlying epistemic structure is genuinely evidence-driven.',
  },
  NLL: {
    title: 'Intuition-driven beyond the belief vocabulary',
    text: 'Vocabulary is balanced, but both contextual models detect intuition-based rhetoric at the sentence and argument levels. The speaker makes belief-based or emotional claims using language outside either seed dictionary, contextually arguing from intuition and values — intuition-based reasoning manifesting through tone, framing, and structural emphasis on conviction over evidence.',
  },
  NHL: {
    title: 'Analytical style without structural follow-through',
    text: 'Balanced vocabulary that reads as analytical sentence-by-sentence but lacks evidence-based argumentative structure. The speaker uses formal, precise language and contextually empirical framing, but the speech does not build a sequential evidence-driven argument. Analytical components are present locally but not integrated into a coherent empirical argument.',
  },
  NLH: {
    title: 'Structured persuasion on normative premises',
    text: 'Balanced vocabulary, intuition-based sentence-level meaning, but an evidence-shaped argumentative structure. The speaker sequences a logical argument from non-empirical premises — beliefs, values, or assumptions arranged premise-to-conclusion despite intuition-driven sentence content. Structured persuasion that mimics empirical reasoning architecturally while drawing on normative content.',
  },
  HNH: {
    title: 'Evidence arc with inconsistent sentence-level usage',
    text: 'Evidence vocabulary and an evidence-shaped argument structure, but mixed contextual usage at the sentence level — some sentences deploy evidence words epistemically, others rhetorically. The overall arc is evidence-based and the vocabulary signals empirical intent, but the contextual execution is inconsistent.',
  },
  HNL: {
    title: 'Decorative evidence vocabulary, no empirical argument',
    text: 'Uses evidence vocabulary but does not organize it into an evidence-based argument, with mixed sentence-level usage. Seed words are deployed inconsistently — some epistemically, some rhetorically — and the overall argument is not structured around empirical reasoning. Vocabulary use is largely decorative or topical rather than substantively epistemic.',
  },
  LNH: {
    title: 'Empirical architecture without conventional vocabulary',
    text: 'Builds an evidence-shaped argument without evidence vocabulary, with mixed sentence-level register. The speaker structures premise-to-conclusion empirical reasoning but uses non-standard vocabulary and inconsistent contextual framing. The argumentative architecture is empirical even though the surface language varies — rigorous reasoning without the conventional empirical vocabulary.',
  },
  LNL: {
    title: 'Intuition-driven with occasional analytical moments',
    text: 'Intuition-based in vocabulary and argument structure, with mixed sentence-level register. The speaker uses belief-oriented language and constructs the speech around appeals rather than evidence, though some sentences are contextually neutral or empirically framed. Genuine intuition-driven rhetoric with occasional analytical moments that do not change its overall character.',
  },
  HNN: {
    title: 'Evidence vocabulary without epistemic positioning',
    text: 'Contains evidence vocabulary that the contextual models do not strongly classify either way. Seed words appear in mixed or ambiguous contexts, possibly as topical references rather than substantive epistemic claims. The vocabulary surface signals empirical intent, but deeper analysis neither confirms nor denies it — common when evidence-related topics are mentioned without strong epistemic positioning.',
  },
  LNN: {
    title: 'Intuition vocabulary without sustained argument',
    text: 'Contains intuition vocabulary that the contextual models do not strongly classify. Belief and feeling words appear in mixed contexts, possibly referencing emotional or normative concepts without committing to intuition-based argumentation. The vocabulary signals subjective framing, but deeper analysis finds no dominant intuition-based strategy — topical mention without sustained intuitive argument.',
  },
  NHN: {
    title: 'Analytical sentence-level tone only',
    text: 'Reads as evidence-based at the sentence level only — vocabulary is mixed and the argument structure is not strongly evidence-shaped. Individual sentences resemble empirical reasoning contextually but are not integrated into a sustained empirical argument. Analytical contextual framing without strong vocabulary signals or argumentative follow-through.',
  },
  NLN: {
    title: 'Subjective sentence-level framing only',
    text: 'Sentence-level contextual meaning reads as intuition-based even though vocabulary is mixed and the argument structure is not strongly intuition-shaped. Individual sentences are framed through feelings, beliefs, or value-claims, but this is not sustained into a full intuition-driven argument — subjective framing without consistent vocabulary or structural reinforcement.',
  },
  NNH: {
    title: 'Empirical argument form on non-empirical content',
    text: 'Mixed vocabulary and sentence-level usage, but a distinctly evidence-shaped argumentative structure. The speech is built as a sequential premise-to-conclusion argument even though local content is not clearly empirical. Rhetorical architecture borrowed from evidence-based reasoning applied to content that is not itself strongly empirical.',
  },
  NNL: {
    title: 'Persuasive structure regardless of topic',
    text: 'Mixed vocabulary and sentence-level usage, but a distinctly intuition-shaped argumentative flow. The speech is structured around appeals, assertions, and value claims arranged for emotional or normative impact rather than logical inference — intuition-driven architecture applied regardless of topic.',
  },
  HHN: {
    title: 'Empirical reporting rather than empirical argument',
    text: 'Uses evidence vocabulary contextually in epistemic ways but does not organize the speech around a clear evidence-based argumentative arc. The speaker makes contextually empirical claims sentence by sentence — citing data, framing claims analytically — but the speech does not build sequentially toward conclusions. Evidence-based local content without coherent argumentative integration: reporting rather than argument.',
  },
  LLN: {
    title: 'Intuition-based content without sustained architecture',
    text: 'Uses intuition vocabulary contextually in stance-taking ways but does not organize around a clearly intuition-shaped arc. The speaker makes belief-based claims sentence by sentence — appealing to feelings, values, or convictions — but the speech does not build through emotional escalation or value-based argumentation. Intuition-based local content without sustained rhetorical architecture.',
  },
  HLN: {
    title: 'Surface evidence vocabulary masking weak content',
    text: 'Evidence vocabulary deployed in non-empirical ways at the sentence level, with no clear argumentative direction. Seed words are used rhetorically — through negation, accusation, or decoration — without contextually empirical claims and without a coherent argument. Surface-level evidence vocabulary masking content that is neither substantively empirical nor structured as empirical argument.',
  },
  LHN: {
    title: 'Local analytical framing without coherence',
    text: 'Reads as contextually empirical at the sentence level without standard evidence vocabulary or a clear evidence-shaped argument structure. The speaker uses non-seed vocabulary to make analytically framed claims sentence by sentence, but does not sustain these into a sequential empirical argument — local analytical framing that does not aggregate into argumentative coherence.',
  },
}

// Map an EMI value to its H / N / L code (same thresholds as classifyEmi).
export function emiCode(v) {
  if (v === null || v === undefined) return null
  if (v > 0.1) return 'H'
  if (v < -0.1) return 'L'
  return 'N'
}

// Look up the interpretation for a full {w2v_emi, bert_emi, gpt2_emi} result.
// Returns null if any score is missing (e.g. W2V had no vocabulary overlap).
export function interpretationFor(emi) {
  const w = emiCode(emi.w2v_emi)
  const b = emiCode(emi.bert_emi)
  const g = emiCode(emi.gpt2_emi)
  if (!w || !b || !g) return null
  return INTERPRETATIONS[`${w}${b}${g}`] || null
}
