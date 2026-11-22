# Manual Audit Worksheet: 30 Stratified Qwen3-1.7B Neuron Poles

**Purpose:** Manual raw-pole reading worksheet. Read the actual marked snippets (`<<token>>`) across activation bands, compare against the generated label, Q3 double-negative flags, rule baselines, and judge verdicts, and record short observations.
**Primary judge:** `gpt-6.1-terra@none` | **Additional judges shown:** `gpt-6-luna@none` (secondary judge), `gpt-6.2-sol@low` (third judge)

## Selection Summary
| # | Pole ID | Cohort | Layer | Bucket | `p99.9+` own | `p90–97` own | `p75–90` own | `p99.9+` no_label |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | `L04_N3614_neg` | `heldout_80` | L4 | Sub-p97 Collapse | 1.00 | 0.17 | 0.17 | 0.67 |
| 2 | `L11_N0664_pos` | `heldout_80` | L11 | Sub-p97 Collapse | 1.00 | 0.50 | 0.17 | 0.67 |
| 3 | `L18_N4045_neg` | `heldout_80` | L18 | Sub-p97 Collapse | 1.00 | 0.00 | 0.67 | 0.50 |
| 4 | `L24_N4188_pos` | `heldout_80` | L24 | Sub-p97 Collapse | 0.83 | 0.17 | 0.17 | 0.50 |
| 5 | `L04_N5913_neg` | `heldout_80` | L4 | Sub-p97 Collapse | 0.83 | 0.17 | 0.33 | 0.67 |
| 6 | `L24_N3502_neg` | `heldout_80` | L24 | Sub-p97 Collapse | 0.83 | 0.00 | 0.50 | 0.17 |
| 7 | `L04_N3295_neg` | `heldout_80` | L4 | Sub-p97 Collapse | 1.00 | 0.50 | 0.33 | 0.00 |
| 8 | `L04_N3614_pos` | `heldout_80` | L4 | Sub-p97 Collapse | 1.00 | 0.50 | 0.33 | 0.67 |
| 9 | `L11_N1576_neg` | `heldout_80` | L11 | Sub-p97 Collapse | 1.00 | 0.17 | 0.67 | 0.67 |
| 10 | `L11_N4088_neg` | `pilot_20` | L11 | Sub-p97 Collapse | 1.00 | 0.50 | 0.33 | 0.33 |
| 11 | `L04_N3295_pos` | `heldout_80` | L4 | Clean Weak-Band Survivor | 1.00 | 0.83 | 0.83 | 0.17 |
| 12 | `L18_N0224_pos` | `heldout_80` | L18 | Clean Weak-Band Survivor | 1.00 | 0.83 | 0.83 | 0.50 |
| 13 | `L11_N0278_pos` | `heldout_80` | L11 | Clean Weak-Band Survivor | 0.83 | 0.83 | 0.83 | 0.67 |
| 14 | `L11_N1224_neg` | `heldout_80` | L11 | Clean Weak-Band Survivor | 0.83 | 0.83 | 0.83 | 0.33 |
| 15 | `L24_N2945_pos` | `pilot_20` | L24 | Clean Weak-Band Survivor | 0.83 | 0.83 | 0.83 | 0.83 |
| 16 | `L18_N1327_neg` | `heldout_80` | L18 | Clean Weak-Band Survivor | 0.67 | 0.67 | 1.00 | 0.50 |
| 17 | `L18_N0809_pos` | `pilot_20` | L18 | Clean Weak-Band Survivor | 0.50 | 0.83 | 0.83 | 0.33 |
| 18 | `L04_N5325_pos` | `heldout_80` | L4 | Clean Weak-Band Survivor | 1.00 | 0.83 | 0.67 | 0.67 |
| 19 | `L18_N2188_pos` | `heldout_80` | L18 | Clean Weak-Band Survivor | 1.00 | 0.67 | 0.83 | 0.83 |
| 20 | `L24_N3118_pos` | `heldout_80` | L24 | Clean Weak-Band Survivor | 1.00 | 0.83 | 0.67 | 0.83 |
| 21 | `L04_N3979_pos` | `pilot_20` | L4 | High-Q3 Double-Negative Pole | 0.83 | 0.67 | 0.83 | 0.50 |
| 22 | `L04_N6043_pos` | `heldout_80` | L4 | High-Q3 Double-Negative Pole | 0.67 | 0.00 | 0.50 | 0.67 |
| 23 | `L04_N5913_pos` | `heldout_80` | L4 | High-Q3 Double-Negative Pole | 0.50 | 0.67 | 0.67 | 0.83 |
| 24 | `L04_N3437_pos` | `heldout_80` | L4 | High-Q3 Double-Negative Pole | 0.83 | 0.83 | 0.50 | 0.83 |
| 25 | `L11_N4363_pos` | `pilot_20` | L11 | High-Q3 Double-Negative Pole | 0.50 | 0.50 | 0.50 | 1.00 |
| 26 | `L24_N3193_pos` | `heldout_80` | L24 | High-Floor / Judge-Disagreement Notable | 1.00 | 0.67 | 0.50 | 1.00 |
| 27 | `L04_N0920_pos` | `pilot_20` | L4 | High-Floor / Judge-Disagreement Notable | 0.83 | 0.33 | 0.50 | 1.00 |
| 28 | `L24_N3118_neg` | `heldout_80` | L24 | High-Floor / Judge-Disagreement Notable | 1.00 | 0.67 | 0.67 | 1.00 |
| 29 | `L24_N4175_neg` | `heldout_80` | L24 | High-Floor / Judge-Disagreement Notable | 1.00 | 0.67 | 0.50 | 1.00 |
| 30 | `L11_N4693_neg` | `heldout_80` | L11 | High-Floor / Judge-Disagreement Notable | 0.83 | 0.50 | 0.50 | 0.67 |

---
## Pole 1/30: `L04_N3614_neg` (Sub-p97 Collapse) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3614 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N528 (gate cos = `0.12154695391654968`)
- **Top-3 trigger tokens:** `[' such', ',', ' Such']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Formal and Quantifying Determiners and Connectives: Triggers include determiners and pronouns like 'such', 'those', and 'as', often introducing examples, categories, or comparisons, as well as connective or quantifying terms marking relationships or sets. Activation is strongest when these tokens appear at the start of noun phrases or clauses, especially in formal or explanatory contexts.
- **Wrong label** (donor `L24_N4175_neg`, L24, sim=0.458414): Proper Name Fragments and Abbreviations: Activations cluster on tokens forming parts of proper names, especially surname or given name fragments, initials, and abbreviations, as well as publisher or organization name segments and technical acronyms. Triggers include capitalized or period-adjacent substrings, often at name or title boundaries.
- **Token-only label:** `Activates on tokens such as: " such", ",", " Such"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.17 | 0.17 | 0.83 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 1.00 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.50 | 0.33 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.50 | 0.50 | 0.50 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.33 | 0.67 | 1.00 |
| `rarity` | 0.17 | 0.50 | 0.33 | 0.33 |
| `top_token` | 0.42 | 0.50 | 0.50 | 0.67 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3614_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.812` `quad=Q2` `gate_open=True` | dist `act_dir=0.000` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Roughly three in 10 have lived in their community 20 years or longer (32% vs. 20% of others), and thus not surprisingly are more likely to know all of their neighbors. This is driven largely by local news enthusiasts age 40 and older.<< As>> a whole, local news enthusiasts are also`
  - **DISTRACTOR:** `ica. ⏎ - Frye, Richard Nelson, Greater Iran, ISBN 1-56859-177-2 p.xi: "... Iran means all lands and people where Iranian languages were and are spoken, and where in the past, multi-faceted Iranian cultures existed.<< ..." ⏎ >>- "Amazons in the Sc`
- **`p97–99.9` (`L04_N3614_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.332` `quad=Q2` `gate_open=True` | dist `act_dir=-0.027` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` to a 12-foot-high cottonwood tree and is quietly ecstatic. "You can see the terminal bud scars," the bespectacled Ripple says, bending the limber tree over to show lines that mark a year's growth of a foot or more on the broom-handle-size trunk.<< ">>You can see that elk haven't brows`
  - **DISTRACTOR:** `ala and Zambezia provinces. Some NGOs provided health care, counseling, and training in other vocations to children engaged in prostitution. ⏎ Traffickers were principally citizens or South African. Trafficking groups included small networks of citizens based in Maputo and Nampula, and there were reports that organized<< crime>> groups were involved. Traffickers often l`
- **`p90–97` (`L04_N3614_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.132` `quad=Q2` `gate_open=True` | dist `act_dir=0.028` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` – the failed General Strike of 1922 by the socialists was referred to by Mussolini as the "Caporetto of Italian Socialism". Many years after the war, Caporetto was still being used to destroy the credibility of the liberal state. ⏎ Popular culture  ⏎ The Battle of<< Cap>>oretto has been the subject of a`
  - **DISTRACTOR:** `-making rights, responsibilities, and authority related to the health, education and welfare of the child. Physical custody designates where the child will reside and who provides the care and supervision. ⏎ Involving the nonresidential parent in school activities ⏎ Nonresidential parents often feel excluded by institutions and organizations,<< including>> schools. Invite nonresidential parents to`
- **`p75–90` (`L04_N3614_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.032` `quad=Q2` `gate_open=True` | dist `act_dir=0.029` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `0:50 mix ⏎ applied at 350 lbs. per acre and an untreated control. Dr. Elmore's weed data ⏎ showed excellent results with Midas when it was used at 50 or 67 percent of the ⏎ fumigant mix but slightly<< reduced>> control when used at only 33`
  - **DISTRACTOR:** ` and develop and the way in which contexts can support learning ⏎ - knowledge of the principles underpinning policy and practice in primary schools ⏎ - an awareness of theories and research that underpin primary practice ⏎ - knowledge of the curriculum for primary children ⏎ - knowledge of how policies relating to children’s status,<< welfare>> and learning impact on children(and adults)`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 2/30: `L11_N0664_pos` (Sub-p97 Collapse) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L11 / N664 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N5160 (gate cos = `0.314024955034256`)
- **Top-3 trigger tokens:** `[' was', ' wet', ' maize']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.62, p90–97: Q3=0.37, p97–99.9: Q3=0.01, p99.9+: Q3=0.00
- **Own label:** Common Nouns, Adjectives, and Auxiliary Verbs: Activations cluster on frequent English nouns (e.g., months, countries, materials, foods, religions), adjectives describing physical or abstract qualities, and auxiliary or linking verbs such as 'was' and 'are', often in mid-sentence contexts. The pole also responds to plural possessive suffixes and some number tokens.
- **Wrong label** (donor `L04_N3614_pos`, L4, sim=0.521599): Sentence-Initial Discourse Markers and Deictics: Activations cluster on sentence-initial adverbs and discourse markers such as 'here', 'now', 'then', and 'sometimes', as well as deictic or referential terms like 'that' and 'of course', often following punctuation or at the start of a new clause.
- **Token-only label:** `Activates on tokens such as: " was", " wet", " maize"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.17 | 0.50 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.67 | 0.17 | 0.33 | 0.33 |
| `gpt-6.1-terra@none` `no_label` | 0.33 | 0.50 | 0.33 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.67 | 0.50 | 0.33 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.17 | 0.50 | 1.00 |
| `rarity` | 0.50 | 0.33 | 0.83 | 1.00 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N0664_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=2.953` `quad=Q1` `gate_open=True` | dist `act_dir=-0.000` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` music produces a powerful form of alert relaxation, a state in which the body functions more efficiently on less energy, making more energy available for the brain. ⏎ Some "slow movement" Baroque music: ⏎ * Adagio in G Minor for Strings ⏎ * Largo from Harpsichord Concerto in<< F>> Minor, BWV 105`
  - **DISTRACTOR:** `unctious or rebellious than another. No objective criteria exist to accurately confirm the presence of ADHD. ADHD often goes undiagnosed if not caught at an early age, and it affects many adults who may not be aware of their condition. ⏎ ADHD is generally recognized by a pattern of inattention<<,>> distractibility, impulsivity, and hyper`
- **`p97–99.9` (`L11_N0664_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.195` `quad=Q1` `gate_open=True` | dist `act_dir=-0.026` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` affirms, for example, the creation of Adam in the image of God (James 3:9). ⏎ Paul wrote to Timothy, "Adam was formed first, then Eve. And Adam was not deceived, but the woman being deceived, fell into transgression" (1 Timothy 2:1<<3>>-14). In 1 Corinthians`
  - **DISTRACTOR:** ` of events is performed until the user stops the process. ⏎ 1. A data processor implemented method of recording events, the steps comprising: ⏎ loading an application; ⏎ adding listeners for each event type produced in the application; ⏎ capturing user generated events; ⏎ recording the user generated events to a data structure; and ⏎ <<re>>playing the user generated events. ⏎ 2.`
- **`p90–97` (`L11_N0664_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.314` `quad=Q3` `gate_open=False` | dist `act_dir=0.001` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` each patrol in a tent. Lay down (face-up) each Ace in a row. Then, on top of the Aces, flip the Kings, then the Queens, then the Jacks. Now, there are 4 piles with one of each card. ⏎ - Flip the piles over and say<< the>> Scoutmaster called lights out and its time`
  - **DISTRACTOR:** ` program highlights the gap between what children know about bullying and how they actually behave in school around peers. ⏎ "We're educating the kids that this is not acceptable and to become leaders for others," Miyairi said. "We want to teach empathy as a core value. We tell the students: Next year<<,>> new students will come here and copy what`
- **`p75–90` (`L11_N0664_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.170` `quad=Q3` `gate_open=False` | dist `act_dir=0.009` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `ove developed into a reclusive and uncommunicative man whose sole passion was books. For so long they were the only meaning to his existence. But when fate eventually intervened to have the outside world intrude upon his life, he began to discover emotions that he never knew he had. ⏎ A story for<< the>> genuine booklover, penned by an Irish`
  - **DISTRACTOR:** ` the molecular level, in order to create new materials that mimic and exceed the properties of biological ones by utilizing material concepts discovered in biological materials. ⏎ We envision that our work can lead to the development of a new set of tools that can be applied, together with synthetic biological and self-assembly methods, to<< select>>, design, and produce a new class`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 3/30: `L18_N4045_neg` (Sub-p97 Collapse) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L18 / N4045 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N3079 (gate cos = `0.3218512237071991`)
- **Top-3 trigger tokens:** `[' ‘', ' “', ',']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Quotation Marks, Definitions, and Metalinguistic Terms: Triggers include quotation marks or typographic variants around words, dictionary or definition contexts, explicit mention of terms or words, and parenthetical or phonetic explanations of terminology. Activation also occurs for punctuation immediately following or preceding such metalinguistic references.
- **Wrong label** (donor `L04_N0920_neg`, L4, sim=0.303359): Proper Noun Suffixes and Rare Morphological Endings: Activations cluster on tokens ending with uncommon suffixes in proper nouns (e.g., -au, -aucon, -amba, -ap, -int, -en, -ica, -er, -ell, -ed, -ward, -stein, -mus, -par, -N) and on rare or irregular morphological endings in both names and technical terms.
- **Token-only label:** `Activates on tokens such as: " ‘", " “", ","`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.00 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.00 | 0.33 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.17 | 0.17 | 0.50 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.17 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 0.00 | 0.67 | 1.00 |
| `rarity` | 0.50 | 0.17 | 0.25 | 0.67 |
| `top_token` | 0.50 | 0.42 | 0.50 | 0.58 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L18_N4045_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=17.250` `quad=Q2` `gate_open=True` | dist `act_dir=-0.005` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` once for both hits and misses. For each trial, the key word or words represented in the photograph were tabulated. Some images had only one key word, but others had two or more. For example, a picture of a couple hugging in a pool of water involved two key words, ‘<<‘>>water’’ and ‘‘hug`
  - **DISTRACTOR:** ` person may not be able to learn other things, as fast as others. That explains; why some students excel in mathematics, some in science, some in languages, some in dramatics, some in art, some in sports etc. Learning also is a process; which we can improve, with certain<< set>> or unset rules. In our life;`
- **`p97–99.9` (`L18_N4045_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.914` `quad=Q4` `gate_open=False` | dist `act_dir=0.041` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.5; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` and adults with disabilities. ⏎ The researchers are charged with three outcomes: ⏎ - Identifying strategies to overcome barriers that impede access to routine health care. ⏎ - Identifying interventions that promote health and wellness, such as exercise, nutrition, pain management, or complementary and alternative therapies. ⏎ - Developing a method for measuring<< the>> health status of people with disabilities. ⏎ People`
  - **DISTRACTOR:** ` thesis (your controlling idea) clearly. You might also give an overview of the main parts of the paper. ⏎ The body of the paper contains your basic information and the points of your argument. Each paragraph in the body develops a single point in detail. Each paragraph is connected to the one before it and<< the>> one to follow it. ⏎ Your conclusion rest`
- **`p90–97` (`L18_N4045_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.547` `quad=Q4` `gate_open=False` | dist `act_dir=-0.043` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.0; rule:top_token=0.0; rule:window_has_ge_p97=0.5
  - **FIRING:** ` and eat it as he goes. ⏎ I used to get so upset when my sister ate with her fingers in company. I used to try to make her use her fork and for a while I didn’t invite anyone round. But then I thought about it and I decided that it really didn’t do anyone<< any>> harm. She wasn’t worried and my`
  - **DISTRACTOR:** `. ⏎ In conclusion, many children experience parental divorce and live in a stepfamily for part of their lives. Schools can support children through family transitions by taking a partnership approach, having informed attitudes, creating an atmosphere that welcomes all types of families, and taking action that supports all children and parents. ⏎ - Berger<<,>> E. H. (200`
- **`p75–90` (`L18_N4045_neg__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.095` `quad=Q4` `gate_open=False` | dist `act_dir=-0.024` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `, 671-684. ⏎ Gilbert, L. & Moore, D.R. (1998). Building interactivity into web courses: Tools for social and instructional interaction. Educational Technology, 38(3), 29-35. ⏎ Hannaf<<in>>, M.J. (198`
  - **DISTRACTOR:** ` and then joined Lawrence Livermore National Laboratory as a postdoctoral researcher in the Earth Science Department. She was hired in a staff position at LLNL in 1994 and worked as a research geophysicist until 2002, when she assumed her current position. ⏎ Her<< research>> at LLNL focused on how things like`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 4/30: `L24_N4188_pos` (Sub-p97 Collapse) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N4188 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N5992 (gate cos = `0.12999185919761658`)
- **Top-3 trigger tokens:** `[' by', ' said', ' delivery']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.84, p90–97: Q3=0.38, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Common English Verbs and Prepositions: Activations cluster on frequent English verbs in various tenses (be, take, say, said, says, traveled, analysed, known, together as a participle, delivery as a nominalization) and on high-frequency prepositions (by, but, hour, bar). The pattern favors morphologically simple, semantically general function words and verb forms.
- **Wrong label** (donor `L18_N0164_neg`, L18, sim=0.524396): Adverbs and Present-Tense Verbs of Continuity or Assertion: Triggers include adverbs such as 'always', 'still', 'never', and present-tense verbs expressing ongoing states or repeated actions (e.g., 'continues', 'repeated', 'presented', 'stated', 'known', 'affirmed', 'matter', 'persistent'). These tokens often appear in contexts emphasizing persistence, regularity, or factual assertion.
- **Token-only label:** `Activates on tokens such as: " by", " said", " delivery"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.17 | 0.17 | 0.33 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.00 | 0.17 | 0.50 | 0.83 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.83 | 0.33 | 0.50 |
| `gpt-6.1-terra@none` `token_only` | 0.17 | 0.67 | 0.67 | 0.67 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.17 | 0.17 | 0.67 | 0.83 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.25 | 0.17 | 0.58 | 0.83 |
| `rarity` | 0.17 | 0.83 | 0.50 | 0.50 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N4188_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=34.000` `quad=Q1` `gate_open=True` | dist `act_dir=0.076` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `, when comparing front-seat occupants and rear-seat passengers—both belted and unrestrained—in the same crash, those in the back seat more often than not sustain less severe injuries than those in the front seat. ⏎ "The rear seat is a much safer area in frontal crashes than the front," Huelke<< says>>. "Lap-belted rear`
  - **DISTRACTOR:** ` court's discretion. ⏎ A final, inter-systemic, explanation involves the dynamic relation between systems that serve youth. During the 1990s, most states saw a reduction in the availability of public mental health services for children, especially inpatient services.36 It is possible that less adequate<< treatment>> contributed to increased delinquencies among youth`
- **`p97–99.9` (`L24_N4188_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=6.812` `quad=Q1` `gate_open=True` | dist `act_dir=0.085` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` reaching the bigoted and fanatical Mohammedan Malays has led to the concentration of efforts upon the Chinese and the Tamils, as they represent the way of least resistance. There is good reason to believe, however, that aggressive work among the more primitive peoples of the jungles would soon yield a large<< harvest>>.” ⏎ “The first difficulty to be met in`
  - **DISTRACTOR:** ` million years ago may have doomed large carnivores ⏎ - Tiny bits of burned plants and bone from a South African cave show that humans had tamed fire by 1 million years ago–some 600,000 year earlier than had previously been documented. ⏎ - Our ancestors began making mult<<icom>>ponent tools in the form of deadly stone`
- **`p90–97` (`L24_N4188_pos__p90_97__dead_zone__3`, `pos_slot=B`):** pos `act_dir=1.531` `quad=Q3` `gate_open=False` | dist `act_dir=0.043` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=MISS(A); gpt-6.2-sol@low/own[rev]=MISS(B); gpt-6.2-sol@low/token_only[fwd]=MISS(A); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` involved at a high level, acting as active partners and educational leaders at home and school, whereas others are involved at lower levels, acting as a recipients of education and support from the school. For parents experiencing transitions such as divorce or remarriage, lower levels of involvement may be all that are possible. ⏎ <<Att>>itudes are often shaped by our own experiences`
  - **DISTRACTOR:** ` post-secondary school earnings and higher-education outcomes as well). The richer the measures used, the less weight there is on the psychometric concerns involving test scores. The alternative is to rest teacher compensation on factors that have little to do with student learning. It is now well established, for example, that<< the>> number of degrees teachers possess and the number`
- **`p75–90` (`L24_N4188_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.602` `quad=Q3` `gate_open=False` | dist `act_dir=-0.002` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `, engineering and math - or STEM, in education terms. ⏎ So while these kids may think they're getting out of class - and out of their shoes into bare feet - to mess around with a submersible boat in a pool, the real lesson in the project is that science and all those related<< subjects>> don't have to be dull. They`
  - **DISTRACTOR:** ` the money from Congress ran out. Countries that had been told they could wipe out malaria in four years--and had diverted much of their health budgets to that effort--grew disillusioned as the years dragged on and eradication never materialized. Soon, they put their money back into areas that seemed<< equally>> pressing, like maternal and child health.`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 5/30: `L04_N5913_neg` (Sub-p97 Collapse) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N5913 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N1090 (gate cos = `0.24552294611930847`)
- **Top-3 trigger tokens:** `[' up', ' of', ' device']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Technical Nouns and Prepositional Phrases: Activations cluster on technical or scientific nouns, often with plural or adjectival forms, and on prepositions like 'of' and 'up' within prepositional phrases. Morphological triggers include common noun suffixes such as -ion, -ing, -ity, -ence, and -ance, as well as device- or technology-related terms.
- **Wrong label** (donor `L11_N0664_neg`, L11, sim=0.593024): Numbers, percentages, and noun suffixes: Activations cluster on numerical tokens (years, percentages, quantities), punctuation following numbers (commas), and common noun or nominalizing suffixes such as -tion, -ers, -e, -s, and -days, especially at word endings or in plural forms.
- **Token-only label:** `Activates on tokens such as: " up", " of", " device"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.33 | 0.17 | 0.83 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.83 | 0.50 | 0.83 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.67 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.33 | 0.83 | 0.50 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 0.17 | 0.83 | 0.83 |
| `rarity` | 0.33 | 0.50 | 0.42 | 0.83 |
| `top_token` | 0.50 | 0.42 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N5913_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.463` `quad=Q4` `gate_open=False` | dist `act_dir=-0.003` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `? ⏎ Spinach leaves are green because they maximally absorb blue and red wavelengths, and green wavelengths are absorbed the least and are therefore reflected back the most, resulting in the green colour ⏎ At what wavelength is the action spectrum measured at? ⏎ The wavelength that photosynthesis occurs at maximally ⏎ Where do light<< reactions>> take place in the chloroplast? Re`
  - **DISTRACTOR:** `, as he did in the fish's belly. Then, when the seaweed was twined around his head, and he seemed to be already buried until the sea should give up her dead, "God prepared the fish to swallow Jonah" . "God could as easily have kept Jonah alive in the<< sea>> as in the fish's belly, but`
- **`p97–99.9` (`L04_N5913_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.183` `quad=Q4` `gate_open=False` | dist `act_dir=-0.009` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` 35 000 genes - high molecular weight DNA) ⏎ Can be used to analyze small amount of plasmid DNA - DNA is not very pure and maxi prep must be used for further analysis as it is a larger quantity of very pure DNA - separates plasmid DNA from bacterial genomic<< DNA>> based on size and conformation ⏎ How`
  - **DISTRACTOR:** `, Giuliano returned to Rome, where he attached himself to Cardinal Branda. Suggestions of wide reform were rife, and the principles of the outward unity of the Church and its reformation from within became the ideals of his life. In 1419 he accompanied Branda on his difficult<< mission>> to Germany and Bohemia, where the`
- **`p90–97` (`L04_N5913_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.092` `quad=Q4` `gate_open=False` | dist `act_dir=0.005` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.0; rule:top_token=0.0; rule:window_has_ge_p97=0.5
  - **FIRING:** ` D. B. Woodbridge, "Footnotes," J. Natl. Class. Mgmt. Soc. 12 (2), 120-124 (1977), p.122. ⏎  R. J. Boberg, "Panel--Classification<< Management>> Today," J. Natl. Class`
  - **DISTRACTOR:** `, a canon of Soissons, appealed to the middle classes of the cities to save Catholicism. A secret society was formed. Rocheblond and five other leaguers carried on a propaganda, gradually organizing a little army at Paris, and establishing relations with the Guises. The combination<< of>> these two movements — the aristocratic and`
- **`p75–90` (`L04_N5913_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.036` `quad=Q4` `gate_open=False` | dist `act_dir=0.011` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `. It was children and grandchildren of European Mennonites who gathered here along the towns only major street, in the first of their peoples congregations to endure in America. The location was the site of a homestead bought by a Dutch-speaking paper-maker four years after the towns founding in 16<<8>>3. ⏎ Willem Rittinghuys`
  - **DISTRACTOR:** ` forensic evidence is almost entirely one-sided, Garrett said. ⏎ "Defense counsel rarely cross-examined analysts concerning invalid testimony and rarely retained experts, since courts routinely deny funding for defense experts." Only 19 of the eventual exonerees whose cases were examined had defense experts. "Prosecutors, moreover<<,>> presented erroneous accounts of the forensic evidence during`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 6/30: `L24_N3502_neg` (Sub-p97 Collapse) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N3502 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N393 (gate cos = `0.133186474442482`)
- **Top-3 trigger tokens:** `[' to', ' be', ' just']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Infinitive Markers, Modals, and Adverbs: Activations cluster on the infinitive 'to', modal verbs like 'can', 'must', and 'be' in passive constructions, as well as adverbs and certain suffixes such as '-ally' and '-ological'. The pattern highlights grammatical markers introducing verb phrases, auxiliary/modal usage, and select adverbial or adjectival forms.
- **Wrong label** (donor `L04_N1888_neg`, L4, sim=0.517345): Government, Academic, and Institutional Nouns and Suffixes: Activations cluster on tokens forming or ending institutional, governmental, or academic nouns, including those with suffixes like -ion, -al, -ial, -ll, -log, and -ce, as well as capitalized proper nouns denoting organizations, events, or titles. Triggers include both full words and partial morphemes at word boundaries, often in formal or informational contexts.
- **Token-only label:** `Activates on tokens such as: " to", " be", " just"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.00 | 0.67 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.67 | 0.33 | 0.33 |
| `gpt-6.1-terra@none` `no_label` | 0.00 | 0.83 | 0.50 | 0.17 |
| `gpt-6.1-terra@none` `token_only` | 0.67 | 0.33 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 0.17 | 0.67 | 0.83 |
| `rarity` | 0.67 | 0.67 | 0.33 | 0.33 |
| `top_token` | 0.50 | 0.50 | 0.58 | 0.75 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N3502_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=76.000` `quad=Q2` `gate_open=True` | dist `act_dir=0.021` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` and a picture of the Centre Block of Parliament. On the reverse side is a wilderness scene with a Snowy Owl. It also had a holographic sticker showing the amount in the top left side, which changes from gold to green when tilted. The front has a wavy background of extremely small but<< still>> clear numeral 50s. This`
  - **DISTRACTOR:** ` as is the case in what is now called an appeal; but Paul, knowing the deadly enmity of the Jews, and the corruption of the governors, elected to be judged at the court of Caesar, which, as a Roman, he had the right to do. Acts 25:<<1>>1. There is One who "com`
- **`p97–99.9` (`L24_N3502_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=37.000` `quad=Q2` `gate_open=True` | dist `act_dir=0.032` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `. Longer life treatments. Substrates that have been atmospheric plasma-treated hold their treatment levels longer than corona-treated surfaces. Longer treatment life allows product converters to take advantage of economies of scale during production, increase inventory life and provide enhanced manufacturing flexibility. ⏎ 2. Higher treat levels allow for treatment of difficult<<-to>>-treat surfaces. Atmospheric plasma treatment is`
  - **DISTRACTOR:** ` with racial prejudice. For some, experiences with discrimination only motivated them to work harder to achieve their goals. For others, the Jim Crow system thwarted their efforts to receive a quality education and pursue their chosen career. Types of discrimination experienced ranged from segregated movie theaters and hostility from whites to relatives privileging<< lighter>> skinned children over those with dark complex`
- **`p90–97` (`L24_N3502_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=2.719` `quad=Q2` `gate_open=True` | dist `act_dir=-0.065` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` the type of bonding. Some ⏎ bonds are quite strong The following are covalent bond strengths. ⏎ Bond Bond Strength(kJ/mole); Bond Bond Strength (kJ/mole) ⏎ Cl-Cl 239 H-Cl 427 ⏎ H-H 432 C-H<< >>413 ⏎ N N 9`
  - **DISTRACTOR:** ` when a virus such as H5N1 comes along, it ⏎ may merge with a flu that a human body already has and develop (or mutate) into ⏎ a whole other subtype for which our body has not built immunity. This, then, ⏎ is what has prompted the concerns of the public health<< community>>. ⏎ The H5N1 virus is`
- **`p75–90` (`L24_N3502_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.377` `quad=Q4` `gate_open=False` | dist `act_dir=-0.033` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `, and I found masking tape to work very well. It’s especially helpful for working on vertical surfaces, of course, but even on a flat horizontal surface, it is a good idea. Be aware, though, that depending on the kind of paper you are using, the area with the masking tape<< may>> need to be trimmed away. This is`
  - **DISTRACTOR:** ` back one hour, following the adage of “spring forward, fall back.” The “extra” hour gives you a chance to catch up on the sleep you lost in March when the clocks moved forward and Daylight Saving Time began. ⏎ The federal government doesn’t require states to adopt Daylight Saving Time<<,>> so Arizona doesn’t. Indiana used to`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 7/30: `L04_N3295_neg` (Sub-p97 Collapse) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3295 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N6025 (gate cos = `0.16478919982910156`)
- **Top-3 trigger tokens:** `[' long', ' or', ' is']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Common Function Words and Short Connectives: Triggers include high-frequency function words such as prepositions, conjunctions, auxiliary verbs, pronouns, and determiners, as well as short adverbs and adjectives in neutral or connective grammatical contexts. Activation is strongest on tokens like 'in', 'long', 'most', 'far', 'or', 'is', 'are', 'then', 'more', 'said', 'the', and punctuation, typically linking or modifying surrounding content.
- **Wrong label** (donor `L11_N4088_neg`, L11, sim=0.513674): Abstract and Existential Nouns, Identity and Being: Triggers include abstract nouns and pronouns denoting identity, existence, or collective being (e.g., self, we, them, oneness, identity, being, home, all), as well as words relating to existence, essence, or conceptual categories (such as nor, such, as, of, an, federal, national).
- **Token-only label:** `Activates on tokens such as: " long", " or", " is"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.33 | 0.50 | 0.17 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.67 | 0.17 | 0.33 | 0.83 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.33 | 0.50 | 0.00 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.17 | 0.33 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 0.33 | 0.33 | 1.00 |
| `rarity` | 0.33 | 0.33 | 0.17 | 0.33 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3295_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.922` `quad=Q2` `gate_open=True` | dist `act_dir=0.017` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `History%252C%2BGovernment%2B%2526%2BSocial%2BStudies&c=1&searchTerm=Holidays&ste=5&tbst=tsrch&tab=1&bConts=55 ⏎ More of<< the>> Best Holiday Crafts http://ebooks.lv`
  - **DISTRACTOR:** ` as might have been shed from the Weka Pass or Amuri limestone beds, and this suggests that they are not unconformable to the beds containing those limestones, a conclusion which is supported by general stratigraphical evidence. These Motunau beds are capped unconformably by gravels<< containing>> numerous fragments of limestone, and from their`
- **`p97–99.9` (`L04_N3295_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.381` `quad=Q2` `gate_open=True` | dist `act_dir=-0.002` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` goats. Humans can also suffer from Brucellosis. In dogs, the disease is caused by a bacterium named Brucella canis, while goats and sheep are infected with Brucella melitensis, pigs with Brucella ovis, and so on. ⏎ Brucel<<los>>is transmission in dogs ⏎ In dogs,`
  - **DISTRACTOR:** `Soft-curd by homogenization. Softness of curd can be imparted to a milk by homogenizing it. This procedure consists in pumping milk under very great pressure through a special valve with small clearance so that the, butterfat globules are broken up and uniformly distributed. The hom<<ogen>>ization of skimmed milk does not impart`
- **`p90–97` (`L04_N3295_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.137` `quad=Q4` `gate_open=False` | dist `act_dir=-0.020` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` Naturwissenschaften. (ANI) ⏎ - Oldest mammalian hair found in France - May 20, 2010 ⏎ - Midair collision kills four in France - Oct 18, 2009 ⏎ - Fossils, mineral rocks attract huge crowds at<< expo>> (With Images) - Jan 0`
  - **DISTRACTOR:** ` vaccine or treatment once an animal gets the disease. If detected early in free-ranging populations, i.e. when prevalence is low, then eradication may be an achievable goal. This is not currently considered possible in the core endemic area; Wisconsin, however, has initiated an aggressive eradication program in the<< portion>> of the state where CWD has been`
- **`p75–90` (`L04_N3295_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.076` `quad=Q4` `gate_open=False` | dist `act_dir=0.013` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` any number of groups. More interestingly, groups can be nested, creating a hierarchy of groups. For example, the "staff" group could encompass all employees, while the "managers" group could be a subset of "staff," and "executives" could be a subset of "<<man>>agers," and so on. ⏎ Setting an`
  - **DISTRACTOR:** ` that has been subjected to BNR treatment processes contains such low concentrations of N and P that the reclaimed water can be applied at rates sufficient to supply a crop’s water needs without risk of surface or ground water contamination. ⏎ The Virginia Water Reclamation and Reuse Regulations require that irrigation with reclaimed water shall be<< limited>> to supplemental irrigation. Supplemental irrigation is`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 8/30: `L04_N3614_pos` (Sub-p97 Collapse) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3614 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N528 (gate cos = `0.12154695391654968`)
- **Top-3 trigger tokens:** `[' here', ' Then', ' Now']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.96, p90–97: Q3=0.95, p97–99.9: Q3=0.77, p99.9+: Q3=0.00
- **Own label:** Sentence-Initial Discourse Markers and Deictics: Activations cluster on sentence-initial adverbs and discourse markers such as 'here', 'now', 'then', and 'sometimes', as well as deictic or referential terms like 'that' and 'of course', often following punctuation or at the start of a new clause.
- **Wrong label** (donor `L11_N0278_pos`, L11, sim=0.395422): Technical and Scientific Terminology Suffixes: Activations cluster on noun and adjective suffixes common in technical and scientific contexts, including -able, -tools, -implemented, -method, -algorithm(s), -model, -end, -implementation, -credit, -virus, -authenticity, -tolerant, -onic, and -new, often marking specialized vocabulary or formal register in descriptions of processes, systems, or attributes.
- **Token-only label:** `Activates on tokens such as: " here", " Then", " Now"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.33 | 0.50 | 0.33 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.67 | 0.50 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.17 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.17 | 0.33 | 0.67 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.33 | 0.50 | 1.00 |
| `rarity` | 0.50 | 0.50 | 0.75 | 0.33 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3614_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.453` `quad=Q1` `gate_open=True` | dist `act_dir=-0.011` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` trying to incite a nearby male into action. Onlookers are watching – just to see what happens. ⏎ However, this is no ordinary territorial scuffle. This is bird research. The sparrow on the perch is a robot, and the chief, hip wader-clad onlooker –<< who>> is also in control of the robot’s`
  - **DISTRACTOR:** `DBYSERVER NT, REPLISTOR 5.0 ⏎ Legato Systems * 650-210-7000 * http://www.legato.com ⏎ BRIGHTSTOR HIGH-AVAILABILITY MANAGER 7.0 ⏎ Computer Associates * 6<<3>>1-342-60`
- **`p97–99.9` (`L04_N3614_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.459` `quad=Q3` `gate_open=False` | dist `act_dir=0.011` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.5; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` of five years and eight months in prison. There were no developments on the prime minister's May 2006 plan to redistribute 494,210 acres of land to 50,000 farmers in Sihanoukville. ⏎ Living conditions worsened at two<< of>> the resettlement sites for former residents of`
  - **DISTRACTOR:** ` a functional whole. The overall objective of better integration between fisheries and other sectors in policy development and management is to maximize synergistic interactions and minimize antagonistic ones. ⏎ Awareness of inter-sectoral dependencies, through the environment and the global economy, is rapidly rising, stimulated inter alia by growing media coverage<< of>> ecological catastrophes, advances in data`
- **`p90–97` (`L04_N3614_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.226` `quad=Q3` `gate_open=False` | dist `act_dir=-0.004` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `‘phone’’ three times, and in another he said ‘‘flower’’ ten times, and in the tabulation of data the numbers of times he said these words are shown in parentheses as: phone (3); flower (10). For most of the statistical analyses, repetitions were ignored<<,>> but in one analysis the numbers of words`
  - **DISTRACTOR:** ` cliff. What effect will gravity have upon the motion of the cannonball? Will gravity affect the cannonball's horizontal motion? Will the cannonball travel a greater (or shorter) horizontal distance due to the influence of gravity? The answer to both of these questions is "No!" Gravity will act downwards<< upon>> the cannonball to affect its vertical motion`
- **`p75–90` (`L04_N3614_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.185` `quad=Q3` `gate_open=False` | dist `act_dir=0.023` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` century boom years. No comprehensive plan was adopted until 1960. 18 Ironically, its proposals were very similar to Nolen and Draper's in concept. During the 1970s the city finally completed a belt road, dubbed "Charlotte 4", and in<< the>> 1980s is struggling`
  - **DISTRACTOR:** ` money out of the pockets of taxpayers rather than from the businesses that benefit from the credit, creating, in effect, a federal subsidy for low-wage employers. A higher minimum wage would help ensure that the earned income tax credit works more effectively and efficiently. ⏎ Raising the minimum wage is about reducing inequality,<< but>> it is also about restoring the true value`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 9/30: `L11_N1576_neg` (Sub-p97 Collapse) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L11 / N1576 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N1636 (gate cos = `0.20067213475704193`)
- **Top-3 trigger tokens:** `[' their', ' to', ' light']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Units of Measurement, Quantities, and Numeric Expressions: Triggers include tokens denoting units (e.g., pressure, currency, length, mass, time, distance), numeric fractions, and words marking quantities or measures, often following numbers or quantifiers. Also activates on prepositions and determiners directly preceding or following such expressions.
- **Wrong label** (donor `L24_N4188_pos`, L24, sim=0.340322): Common English Verbs and Prepositions: Activations cluster on frequent English verbs in various tenses (be, take, say, said, says, traveled, analysed, known, together as a participle, delivery as a nominalization) and on high-frequency prepositions (by, but, hour, bar). The pattern favors morphologically simple, semantically general function words and verb forms.
- **Token-only label:** `Activates on tokens such as: " their", " to", " light"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.17 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.83 | 0.50 | 0.00 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.33 | 0.83 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.67 | 0.17 | 0.17 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.33 | 0.33 | 0.83 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.75 | 0.42 | 0.50 | 1.00 |
| `rarity` | 0.50 | 0.50 | 0.67 | 0.17 |
| `top_token` | 0.58 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N1576_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.938` `quad=Q2` `gate_open=True` | dist `act_dir=0.042` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=MISS(A); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` of bone tissue. ⏎ Normal cycle of hair growth. ⏎ About 10 % of the hair on the scalp is in a resting phase at any given time. The resting hair falls after 2 to 3 months and new hair starts growing in its place. The growing phase continues for 2.2<<5>> to 6 yrs. During this phase`
  - **DISTRACTOR:** ` They feared that such an organization, supported by the authorities and the Liberal Jews, would diminish the communities' independence and undermine the status of the rabbi as supreme religious authority and sole arbiter. After all attempts at compromise failed, the Orthodox delegates walked out of the congress. The conflict between the camps<< reached>> its peak, causing a rupture between the`
- **`p97–99.9` (`L11_N1576_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.330` `quad=Q4` `gate_open=False` | dist `act_dir=-0.036` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=MISS(A); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` ⏎ A new study led by researchers from Harvard School of Public Health (HSPH) and the Wellcome Trust Sanger Institute in the UK has, for the first time, used genome sequencing technology to track the changes in a bacterial population following the introduction of a vaccine. The study follows how th<<...>> Read More ⏎ Certain types of anti-de`
  - **DISTRACTOR:** ` and educational environments, and learning how to make decisions for themselves. Adults can offer a helping and supportive hand to young adults when dealing with peer pressure by considering the following: ⏎ Separation: Understanding that this is a time for the child to separate and learn how to be his own individual is important. It<< is>> hard to let go and allow the child`
- **`p90–97` (`L11_N1576_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.151` `quad=Q4` `gate_open=False` | dist `act_dir=-0.016` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=OK(B); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` in children. Middle ear infections can also cause a hole (perforation) in the eardrum or spread to nearby areas, such as the mastoid bone. Adults also can get middle ear infections. ⏎ Children in day care have an increased risk of middle ear infections. The increased exposure to other<< infected>> children increases your child's chances of getting`
  - **DISTRACTOR:** `, non-Newtonian physical behavior. However, cats and dogs that are especially long-haired can exhibit a wavelike appearance. Furthermore, animals clearly interfere with each other, sometimes destructively (as with cats and dogs, or with cats and humans that are attempting to read a newspaper), sometimes construct<<ively>> (as with rabbits and rabbits). ⏎ Er`
- **`p75–90` (`L11_N1576_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.051` `quad=Q4` `gate_open=False` | dist `act_dir=-0.027` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` was chewing our freshly-built cabin at a woods edge in the Catskills. The animal looked surreal and wild, but I had no doubt about its identification. It had quills, therefore it was a porcupine. ⏎ But the quills that give porcupines their easy identification and shape their natural<< histories>> are themselves the source of endless mystery and`
  - **DISTRACTOR:** ` addition to their symbolic religious meanings, these places are characterized by a ⏎ remarkable beauty thanks to the careful integration of architectural elements into the surrounding natural landscapes: hills, woods ⏎ and lakes. Furthermore they boast very important works of art like frescos and statues. ⏎ UNESCO World Heritage Committee has put the<< Site>> on its List for the following reasons: ⏎ `

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 10/30: `L11_N4088_neg` (Sub-p97 Collapse) **[PILOT_20]**
- **Layer / Neuron / Sign:** L11 / N4088 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N1850 (gate cos = `0.13034017384052277`)
- **Top-3 trigger tokens:** `[' self', ' we', ' them']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Abstract and Existential Nouns, Identity and Being: Triggers include abstract nouns and pronouns denoting identity, existence, or collective being (e.g., self, we, them, oneness, identity, being, home, all), as well as words relating to existence, essence, or conceptual categories (such as nor, such, as, of, an, federal, national).
- **Wrong label** (donor `L24_N4212_pos`, L24, sim=0.30256): Adverbs and Past Tense Verbs in Factual Narration: Strong activation occurs on adverbs indicating degree, frequency, or approximation, as well as on past tense verbs in narrative or explanatory contexts, especially when describing historical events, processes, or quantitative changes.
- **Token-only label:** `Activates on tokens such as: " self", " we", " them"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.33 | 0.50 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.17 | 0.83 | 0.33 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.33 | 0.67 | 0.33 |
| `gpt-6.1-terra@none` `token_only` | 0.17 | 0.50 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.67 | 0.50 | 1.00 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.25 | 0.50 | 0.42 | 1.00 |
| `rarity` | 0.67 | 0.33 | 0.58 | 0.17 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N4088_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=3.672` `quad=Q2` `gate_open=True` | dist `act_dir=0.006` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` ought to be said. After the Holocaust, for example, a number of Jewish theologians have felt the covenant was so compromised that it is no longer binding. ⏎ All that notwithstanding, the spiritual question most of us encounter is whether life is little more than a series of coincidences. To some, our<< existence>> is the outgrowth of luck, possibly`
  - **DISTRACTOR:** ` her doctorate, Miyairi worked for four years at Avalon Hills in Logan, where she helped treat those with eating disorders. ⏎ "I'm hoping more school districts and researchers can work together," said Miyairi, who will get her doctorate sometime this summer. "My next plan is to<< expand>> this project to educateÂ parents, teachers`
- **`p97–99.9` (`L11_N4088_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.484` `quad=Q2` `gate_open=True` | dist `act_dir=-0.021` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=MISS(B); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=OK(B); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Sciences| ⏎ Investigations are being made on cellular rhythms including those in lymphatics, blood vessels, gastrointestinal tract, heart and specific mood-associated brain nuclei. We have discovered a new mechanism that it is driven by intracellular Ca2+ stores that allow groups of cells to self pace and hence become rhyth<<mic>>. We are now exploring the relevance of`
  - **DISTRACTOR:** ` that countries treat the disease largely through the health-care system--through elimination of the parasite--but many anti-malarial drugs are no longer effective. In the past thirty years, there have been outbreaks in India, Sri Lanka, Brazil, and South Korea, among other places. "Our troubles with<< mosquitoes>> are getting worse," Spielman concludes in`
- **`p90–97` (`L11_N4088_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.262` `quad=Q4` `gate_open=False` | dist `act_dir=0.011` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=MISS(B); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=MISS(A); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `'s Gate was constructed - a cluster of different sized towers designed to command all angles of attack. In the 18th century, Colonel William Twiss added extra gun positions, and constructed Constable's Bastion, Canon's Gateway and numerous barracks and storehouses. Further additions were made in the<< >>1850s, and Dover`
  - **DISTRACTOR:** `. No further consultation is necessary for those wetlands. ⏎ 6) The eastern prairie fringed orchid may be present in your action area. You may either assume that the eastern prairie fringed orchid is present and proceed to Step 2 of the Section 7 Consultation Process or conduct<< a>> field search during the bloom date of the`
- **`p75–90` (`L11_N4088_neg__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.198` `quad=Q4` `gate_open=False` | dist `act_dir=-0.020` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=MISS(A); gpt-6.2-sol@low/own[rev]=MISS(B); gpt-6.2-sol@low/token_only[fwd]=MISS(A); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `. ⏎ It's unclear how o... Read More ⏎ Through the serendipity of science, researchers at the University of Pittsburgh have discovered a potential treatment for deadly, drug-resistant bacterial infections that uses the same approach that HIV uses to infect cells. ⏎ The National Institutes of Health-supported discovery will be descr...<< Read>> More ⏎ Researchers from UNSW have cautioned`
  - **DISTRACTOR:** ` to imbed our sample sites within areas that were both recently disturbed by man and others which had been relatively untouched for quite some time. Because Rio Bravo is a preserve and PfB encourages research activities, we felt confidant that human impact on both the ecosystem and our sampling devices would be minimal.<< Thus>>, it seemed probable that any potential differences`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 11/30: `L04_N3295_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3295 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N6025 (gate cos = `0.16478919982910156`)
- **Top-3 trigger tokens:** `[' says', ' say', ' said']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.94, p90–97: Q3=0.93, p97–99.9: Q3=0.55, p99.9+: Q3=0.00
- **Own label:** Speech and Thought Reporting Verbs: Triggers include present and past tense reporting verbs such as 'say', 'said', 'says', 'believe', 'hope', 'knew', 'reveals', 'confirmed', and 'thought', typically used to attribute statements, beliefs, or discoveries to people or groups in journalistic or narrative contexts.
- **Wrong label** (donor `L18_N2188_pos`, L18, sim=0.236903): Etymology and Naming Explanations: Activates on tokens introducing explanations of word origins, meanings, or naming—especially in phrases like 'the name', 'the term', 'named after', and when defining or translating words from other languages or contexts.
- **Token-only label:** `Activates on tokens such as: " says", " say", " said"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.33 | 0.67 | 0.50 |
| `gpt-6.1-terra@none` `no_label` | 0.17 | 0.17 | 0.50 | 0.17 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.67 | 0.50 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.83 | 0.67 | 1.00 |
| `rarity` | 0.33 | 0.50 | 0.67 | 0.50 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.58 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3295_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=2.750` `quad=Q1` `gate_open=True` | dist `act_dir=-0.027` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=1.0; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` any Third Party Site. Never disregard the medical advice of your physician or health professional, or delay in seeking such advice, because of something you read on this Site. We offer this Site AS IS and without any warranties. By using this Site you agree to the following Terms and Conditions. If you think<< you>> may have a medical emergency, call your`
  - **DISTRACTOR:** ` week in the Springer journal Behavioral Ecology and Sociobiology. ⏎ The bearded vulture is the only vertebrate with a bone-dominated diet and can ingest bones up to 28 cm (11 inches) long and up to 4 cm (1 inches) wide. This makes sound ecological sense<< because>> due to their high fat content, mamm`
- **`p97–99.9` (`L04_N3295_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.324` `quad=Q3` `gate_open=False` | dist `act_dir=0.002` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Australia. Palm oil can be found in everything from baked goods, ⏎ confectionery and ice-cream, to body products, cosmetics and cleaning ⏎ agents. It can even be used in toothpaste, paint, printer ink and car fuel! ⏎ Why is palm oil so popular and widely used? ⏎ - High<< yielding>> vegetable oil crop ⏎ - Large economic benefit`
  - **DISTRACTOR:** `400 – 750||500 – 1000||90 – 180||6 – 8| ⏎ |Sub humid||750 – 1200||1000 – 1500||180<< –>> 270||4–6`
- **`p90–97` (`L04_N3295_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.218` `quad=Q3` `gate_open=False` | dist `act_dir=-0.027` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` food and dietary supplements. Green vegetables, fruits, and juices have natural folate, and other foods, such as cereal and bread, have folic acid added to them. CDC is looking at NHANES data to see how people get folic acid and to see if they are getting the recommended<< amount>>. This information will help determine if adding`
  - **DISTRACTOR:** ` Spanish 2||3| ⏎ |Computer Information System Elective||1||Baptist Heritage||3| ⏎ |Christian Life 2||1||Bible Elective||2| ⏎ |Principles of Bible Study||2||Minor||3| ⏎ |Writing Elective||3||Literary<< Crit>>icism||3| ⏎ |Period Literature Elect`
- **`p75–90` (`L04_N3295_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.093` `quad=Q3` `gate_open=False` | dist `act_dir=0.011` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` the slaves was indentured to farmer Thomas Smith who had sold the property for the new facility to the Philadelphia Board of Health. ⏎ Moreover, their story reflects in miniature the history of the Philadelphia Lazaretto itself: integral to our nation’s demographic and political landscape and yet often overlooked. Stories like those of<< the>> Ganges Africans and the thousands of others`
  - **DISTRACTOR:** `1992 ⏎ - Evelyn Czech 1926 - 1981 ⏎ - K E Czech 1956 - 1990 ⏎ - Michael J Czech 1921 - 1999 ⏎ - John Czech 1896<< ->> 1976 ⏎ - Ag`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 12/30: `L18_N0224_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L18 / N224 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N5865 (gate cos = `0.48630785942077637`)
- **Top-3 trigger tokens:** `[' and', '.', ' with']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.07, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Coordinating Conjunctions and Prepositions: Triggers include coordinating conjunctions (and, or, while, as, versus, that, who) and prepositions (with, from, versus, anti-) at the start of or within multi-word phrases, often linking clauses, items, or contrasting elements in formal or expository contexts.
- **Wrong label** (donor `L04_N6043_neg`, L4, sim=0.324356): Suffixes -ing, -ion, -ive, -al, -mal, -op, -aw, -res: Activations cluster on tokens ending with common English suffixes such as -ing, -ion, -ive, -al, -mal, -op, -aw, and -res, as well as some inflected verb forms and technical or scientific terms. The pattern includes both nominalizations and adjectives, often marking morphological boundaries.
- **Token-only label:** `Activates on tokens such as: " and", ".", " with"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.83 | 0.33 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.33 | 0.33 | 0.50 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.83 | 0.67 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 1.00 | 0.67 | 1.00 |
| `rarity` | 0.67 | 0.25 | 0.67 | 0.00 |
| `top_token` | 0.50 | 0.58 | 0.58 | 0.67 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L18_N0224_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=8.688` `quad=Q1` `gate_open=True` | dist `act_dir=0.075` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` the classroom. 20. Find ways to engage students in your classroom and school. 21. When possible, find incentives for high school graduation. Yes, even dollars. 22. Offer college credit to at-risk high school kids for taking a class that helps them understand the importance<< of>> having a high school diploma. 2`
  - **DISTRACTOR:** `ful. ⏎ The confusion and forgetfulness caused by emotions usually are temporary and go away when the feelings fade. The emotional problems can be eased by supportive friends and family, but if these feelings last for a long time, it is important to get help from a doctor or counselor. Treatment may include counseling,<< medication>>, or both. ⏎ More serious memory problems`
- **`p97–99.9` (`L18_N0224_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=3.297` `quad=Q1` `gate_open=True` | dist `act_dir=0.004` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` This way, the software can definitely apply to user needs and interests as well as draw closer to perfection in the most efficient way possible. Many organizations have adopted the open source philosophy in order to produce the premier software of their markets and many nonprofit organizations have coalesced in support of open source software.<< Even>> governments have heralded their support of open`
  - **DISTRACTOR:** `When you lose weight, you not only feel better, but your fat cells are much healthier. ⏎ So says endocrinologist Andrew Greenberg, director of the Obesity Metabolism Laboratory at Tufts University in Boston. ⏎ Your body needs fat cells to be healthy, but in obese individuals when fat cells get<< very>> big, those cells are at risk of`
- **`p90–97` (`L18_N0224_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.195` `quad=Q1` `gate_open=True` | dist `act_dir=-0.002` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` Having this around your neck, you are involving God's presence and protection against harm." ⏎ Dr. Esther Eshel, a professor of the Bible at Bar-Ilan and an authority on Hebrew inscriptions, said this was the earliest example of amulets from Israel. But she noted that the language<< of>> the benediction was similar to a`
  - **DISTRACTOR:** `, effective government, religious freedom, the possibility of meaningful work, access to the arts, freedom and other things required for a society where individuals can flourish, where others can can live into God's design for them. ⏎ As the noted author Richard Foster observes, “Work places us into the stream of divine<< action>>. We are ‘subcreators,’`
- **`p75–90` (`L18_N0224_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.396` `quad=Q3` `gate_open=False` | dist `act_dir=0.078` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` can do it with an iPad. The result is spooky 3D letters that seem to hover in midair. Casper would love it. ⏎ We've seen iPads used to create touch-screen art, but this goes a step beyond. The folks at communications agency Dentsu London and design consultancy<< Berg>> teamed up to produce an series of other`
  - **DISTRACTOR:** `, J. A. J. Breeuwer, & G. D. D. Hurst (1999). WOLBACHIA PIPIENTIS: Microbial Manipulator of Arthropod Reproduction Annual Review of Microbiology DOI: 10.11<<4>>6/annurev.micro.5`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 13/30: `L11_N0278_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L11 / N278 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N2775 (gate cos = `0.17099517583847046`)
- **Top-3 trigger tokens:** `[' implemented', 'able', ' tools']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.98, p90–97: Q3=0.97, p97–99.9: Q3=0.87, p99.9+: Q3=0.00
- **Own label:** Technical and Scientific Terminology Suffixes: Activations cluster on noun and adjective suffixes common in technical and scientific contexts, including -able, -tools, -implemented, -method, -algorithm(s), -model, -end, -implementation, -credit, -virus, -authenticity, -tolerant, -onic, and -new, often marking specialized vocabulary or formal register in descriptions of processes, systems, or attributes.
- **Wrong label** (donor `L18_N0783_neg`, L18, sim=0.587207): Abstract Nouns, Modals, and Infinitives: Activations cluster on abstract nouns ending in -ion, -ity, -ism, or -al, as well as on modal verbs and infinitive constructions, especially in formal or explanatory contexts. Occasional triggers include pronouns and prepositions in set phrases or idiomatic expressions.
- **Token-only label:** `Activates on tokens such as: " implemented", "able", " tools"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.67 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.17 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.50 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.50 | 0.33 | 0.50 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 0.67 | 0.83 | 0.83 |
| `rarity` | 0.83 | 0.75 | 0.33 | 0.83 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N0278_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.906` `quad=Q1` `gate_open=True` | dist `act_dir=0.042` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Portable application suites like PortableApps.com, CodySafe, or Lupo PenSuite, each have a main menu providing access to the programs installed into the suite. ⏎ However, there may be reasons why you need to create shortcuts to programs on your USB drive. You may be using a program that does not<< integrate>> into the suite’s main menu. Or`
  - **DISTRACTOR:** ` strangle the baby during a vaginal delivery. ⏎ Fetal distress or medical problems: if the mother has diabetes, genital herpes, hypertension, cardiac disease, toxemia, or ovarian or uterine cysts, vaginal delivery may be hazardous for both mother and child. ⏎ If a C-section may be necessary<< for>> you, it is important to plan ahead`
- **`p97–99.9` (`L11_N0278_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.352` `quad=Q3` `gate_open=False` | dist `act_dir=0.037` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` with Hepatitis C virus infection. Neurology 1999;53:861-864. ⏎ 22. Bolay H, Soylemezoglu F, Nurlu G, et al. PCR PCR polymerase chain reaction. ⏎ polymerase chain reaction<< ⏎ >>Polymerase chain reaction (PCR)`
  - **DISTRACTOR:** `A play in how bacteria respond to stress and other physiological stimuli may help scientists create new antibacterial drugs that target mechanisms specific and unique to harmful bacteria. ⏎ "Conventional antibiotics aimed at killing bacteria also put immense pressure on bacteria to survive, and to ultimately develop resistance to these drugs," Artsimovitch<< said>>. "Forcing harmful bacteria into a`
- **`p90–97` (`L11_N0278_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.201` `quad=Q3` `gate_open=False` | dist `act_dir=0.029` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `olphe Ernst-Metzmaier, prepared the final drawings. ⏎ Though the project was far more advanced than the two first French tanks about to enter production, the Schneider CA1 and the heavy St. Chamond, Renault had at first great trouble getting it accepted. Even after the first British use of<< tanks>>, on 15 September 1`
  - **DISTRACTOR:** ` to protect children from serving as soldiers or being trafficked into prostitution. ⏎ But Anthony adds there is still a long way to go. He says as many as one billion children still live without basic necessities such as clean water and food. And he says more needs to be done in the fields of protection and<< participation>> - the right of all children to have`
- **`p75–90` (`L11_N0278_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.134` `quad=Q3` `gate_open=False` | dist `act_dir=0.005` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.0; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` waters and had a major advantage over its prey. ⏎ "The little that we do know about the contemporaries of anomalocaris, they either had fairly poor eyesight or they were completely blind," says Paterson. ⏎ He says the presence of anomalocaris would have driven the development of protective adaptations in prey<< animals>>, and in turn, counter adaptations in`
  - **DISTRACTOR:** ` sanction teachers. But teachers are the dominant school input, in terms of both spending and impact on student learning. Excluding them essentially leaves the education system without accountability. ⏎ The main concern with value-added assessment is that the technique exacerbates the amount of random error involved in measuring student performance. The risk is<< that>> teachers and schools may be wrongfully rewarded`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 14/30: `L11_N1224_neg` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L11 / N1224 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N1043 (gate cos = `0.2416980117559433`)
- **Top-3 trigger tokens:** `[' town', ' window', "'s"]`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Commonplace Nouns and Locative Endings: Activations cluster on frequent English nouns, especially those denoting places, people, or objects, as well as on common noun suffixes like -s, -er, -or, -ion, and -ed, and prepositions such as 'of'.
- **Wrong label** (donor `L04_N1366_pos`, L4, sim=0.45451): Function Words and Pronouns in Contextual Phrases: Triggers include prepositions, conjunctions, and pronouns such as 'with', 'for', 'about', 'his', 'he', 'it', 'there', 'to', 'on', and 'that', especially when embedded in explanatory, comparative, or reporting constructions. These tokens often appear at clause boundaries or as connectors within complex sentences.
- **Token-only label:** `Activates on tokens such as: " town", " window", "'s"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.67 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.50 | 0.33 | 0.83 |
| `gpt-6.1-terra@none` `no_label` | 0.17 | 0.50 | 0.33 | 0.33 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.50 | 0.83 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.33 | 0.50 | 0.83 |
| `rarity` | 0.83 | 0.50 | 0.42 | 0.67 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N1224_neg__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=2.781` `quad=Q2` `gate_open=True` | dist `act_dir=-0.011` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` the Yankee warships stationed there. By the time Buchanan ordered his ship to sail on March 8, 1862, Heywood was a captain commanding the USS Cumberland’s Marine contingent. ⏎ Brilliant late-winter sunshine and warm air engulfed Hampton Roads as the CSS Virginia stood down<<river>> that Saturday. Buchanan had already selected his`
  - **DISTRACTOR:** `4% of all working youth during the 1990s were employed in agriculture, but they experienced more than 40% of the youth occupational fatalities. The Bureau of Labor Statistics reported that agricultural workers ages 15 to 17 have a risk of dying that’s 4<<.>>4 times greater than the average 1`
- **`p97–99.9` (`L11_N1224_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.984` `quad=Q2` `gate_open=True` | dist `act_dir=-0.006` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Google as places to go. Wrong. That's the old media model. In reality, every click is a command for some computer somewhere in the world. (...) Indeed, CEO Eric Schmidt says that Google essentially is a huge, distributed supercomputer "doing all sorts of things over a fiber-optic network<< that>> eventually are services available to end-users."`
  - **DISTRACTOR:** `592. ⏎ Stevens, E. J. and Villalta, P., 1983. Proc. 14th Int. Grassl. Congr., p. 176. ⏎ Susetyo, S., Whiteman, P. C. and<< Humph>>reys, L. R., `
- **`p90–97` (`L11_N1224_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.404` `quad=Q2` `gate_open=True` | dist `act_dir=-0.015` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` and the elimination of pinholing and backside treatment. ⏎ Until recently, low-pressure vacuum plasmas have been used to treat two- and three-dimensional plastic objects and polymer films. The recent introduction of high-density atmospheric plasma for film, foil, paper, foam, nonwoven and woven substrate<< applications>> has had a significant impact on the surfaces`
  - **DISTRACTOR:** ` Bottled water is expensive - if you apply the costs of bottled water toward the purchase of a filtration system, it will pay for itself in only 8-14 months. ⏎ The Environmental Protection Agency regulates public water supplies, but has no authority over bottled water. For example, there are no specific<< requirements>> regarding proximity of bottled water sources to industrial`
- **`p75–90` (`L11_N1224_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.087` `quad=Q4` `gate_open=False` | dist `act_dir=0.009` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` is to remove an electron. The ionization energy may be an indicator of the reactivity of an element. Elements with a low ionization energy tend to be reducing agents and form cations, which in turn combine with anions to form salts. ⏎ Electron Affinity  ⏎ |Electron affinity is<< the>> opposite of ionization energy. It is`
  - **DISTRACTOR:** ` on condition that Antiochus should put away his wife Laodice and marry Berenice the daughter of Ptolemy. This was done, but on the death of Philadelphus he restored Laodice; but she, fearing another divorce, poisoned her husband, and then<< caused>> the death of Berenice and her`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 15/30: `L24_N2945_pos` (Clean Weak-Band Survivor) **[PILOT_20]**
- **Layer / Neuron / Sign:** L24 / N2945 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N1924 (gate cos = `0.22281494736671448`)
- **Top-3 trigger tokens:** `[' http', '(equal', 'ssh']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.98, p90–97: Q3=0.94, p97–99.9: Q3=0.69, p99.9+: Q3=0.00
- **Own label:** Abbreviations, URLs, and Directional or Relational Terms: Strong activation occurs on tokens forming abbreviations (e.g., 'BI', 'ag', 'al'), URL prefixes ('http'), time or number markers, and words denoting direction, relation, or position (e.g., 'northwest', 'Middle', 'light', 'school', 'book', 'create', 'interested', 'ssh').
- **Wrong label** (donor `L18_N0224_neg`, L18, sim=0.550731): Coordinating Conjunctions and Prepositional Phrases: Strong activation occurs on coordinating conjunctions (and, or, as) linking noun or verb phrases, as well as on common prepositions (with, for, by, of, on) introducing prepositional phrases, especially in formal or expository contexts. Occasional activation also appears on punctuation following list items and on determiners in technical explanations.
- **Token-only label:** `Activates on tokens such as: " http", "(equal", "ssh"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.67 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.50 | 0.00 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.50 | 1.00 | 0.83 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.50 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 0.83 | 0.50 | 1.00 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.58 | 0.83 | 0.92 | 0.92 |
| `rarity` | 0.83 | 0.67 | 0.67 | 1.00 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N2945_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=11.188` `quad=Q1` `gate_open=True` | dist `act_dir=0.009` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` registered nurses, therapists, and doctors. Keep in mind that technologists are senior to technicians, typically earn higher salaries, and often need a bachelor's degree and credentials. ⏎ Expected Openings: 23,800 ⏎ Expected Openings: 23,400 ⏎ Expected Open<<ings>>: 210,20`
  - **DISTRACTOR:** ` its cultivars because it has the most attractive foliage in the entire genus. Its horizontal, long, leathery, pleated foliage is green with a striking purple cast. As the blossom first opens, it is nestled on top of the attractive foliage; the flowering stems continue to grow. It pushes<< the>> large, spherical umbels up and away`
- **`p97–99.9` (`L24_N2945_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=2.859` `quad=Q1` `gate_open=True` | dist `act_dir=-0.006` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=MISS(A); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `3 ⏎ N N 941 N-H 391 ⏎ Bond strength can be overcome by adding heat...The GC and AT bonds in DNA ⏎ are hydrogen bonds (non-covalent) and can also be overcome with heat. ⏎ these hydrogen bonds are about 71 kilojou<<les>> per mole...relatively weak (but`
  - **DISTRACTOR:** ` the milk. The scientific literature leaves the subject in a very con-fused state. Whatever improved digestibility there is seems to result entirely from the speeding of the passage of the milk from the stomach and not from any increased food value or degree of assimilation.' The whole subject is excellently reviewed by<< Do>>an in the Journal of Dairy Science,`
- **`p90–97` (`L24_N2945_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.234` `quad=Q3` `gate_open=False` | dist `act_dir=-0.011` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=MISS(A); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `. On five kinds of warriors: those who go down into the thick of the fight where they are overpowered, those who are wounded and die on the way to their home, those who survive for some time but die of their wounds, those who are cured of their wounds, those that are victorious<< in>> battle and continue to fight. There are`
  - **DISTRACTOR:** `ges and palms. ⏎ Phytoliths are difficult to identify, but have the advantage of surviving in both charred and non-charred contexts, so we can learn about the presence and use of plants in areas where seeds and grains don’t survive. ⏎ At the moment, I am processing sediments to<< extract>> phytoliths, which includes the`
- **`p75–90` (`L24_N2945_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.068` `quad=Q3` `gate_open=False` | dist `act_dir=0.013` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `fection. Their need to appear or be perfect perfectionism is so intense that it's exhausting if not painful. ⏎ But striving for perfection, while accepting that perfection rarely can be achieved, can lead to growth and development and a feeling of satisfaction. It can be a powerful motivator as long as it is<< based>> on reasonable standards and expectation. For example`
  - **DISTRACTOR:** ` breast and other cancers in so many women. And natural harmony, as her mother knows, is not what it used to be, especially now that pollutants are acting like even more estrogen in our bodies. Raymond-Whish can only hope that Darby's cells have the normal number of receptors, and<< that>> her genes and her environment haven't somehow`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 16/30: `L18_N1327_neg` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L18 / N1327 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N1440 (gate cos = `0.2554132640361786`)
- **Top-3 trigger tokens:** `[',', ' rather', ' and']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Coordinating and Contrasting Conjunctions and Commas: Triggers include coordinating conjunctions such as 'and', 'or', 'but', 'rather', 'instead', and 'than', as well as commas used in lists or appositive phrases. Activation often occurs at the conjunction or comma itself, especially when linking clauses, contrasting alternatives, or enumerating items.
- **Wrong label** (donor `L11_N0278_neg`, L11, sim=0.404474): Technical Terms and User-Directed Language: Activations cluster on computer-related nouns, software and hardware terminology, and user-directed possessive or imperative constructions, especially with 'your' and 'to'. Triggers include technical vocabulary, file types, software names, and instructions or references to user actions or possessions.
- **Token-only label:** `Activates on tokens such as: ",", " rather", " and"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 1.00 | 0.67 | 0.17 | 0.67 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.50 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.17 | 0.00 | 0.50 |
| `gpt-6.1-terra@none` `token_only` | 1.00 | 1.00 | 0.33 | 0.67 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 0.33 | 0.33 | 0.67 |
| `rarity` | 0.33 | 0.33 | 0.42 | 0.33 |
| `top_token` | 0.67 | 0.58 | 0.42 | 0.58 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L18_N1327_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=7.031` `quad=Q2` `gate_open=True` | dist `act_dir=-0.074` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` bacterial vaginosis. While it's more common in women who are sexually active, it also occurs in women who are not sexually active. It's not usually necessary for your sex partner to be treated. ⏎ Your doctor will examine your vagina and use a cotton swab to get a sample of the discharge<<.>> This sample will be tested. ⏎ Yes.`
  - **DISTRACTOR:** ` Aglink FPP 681, N.Z. Min. Agr., Fish., Wellington. ⏎ Minson, D. J., 1982. In Grazing Animals (Ed. F. W. H. Morley), pp. 143–158,<< Else>>vier, Amsterdam. ⏎ Morley, F`
- **`p97–99.9` (`L18_N1327_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=2.750` `quad=Q2` `gate_open=True` | dist `act_dir=-0.051` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `ah, Amen and Heru-khuti is the most iconic Ancient Egyptian sight after the Pyramids. The temple has four massive statues of Ramses II and the gods Ra-Horakhty, Amun and Ptah at its entrance. Beyond lie two pillared halls and the sacred sanctuary<<.>> Every wall and ceiling is a storybook`
  - **DISTRACTOR:** `ok the Priest" to illustrate. ⏎ - West Coast Swing Ladies Syncopations ! West Coast Swing Ladies Syncopations & Variations with Laurie Schwimmer! ⏎ - Syncopation - It's Jazzy ⏎ - Dynamic Syncopation - The Essence Dynamic Syncopation - The Essence from Dynanism<< ⏎ >>- Dynamic Syncopation - The Plan`
- **`p90–97` (`L18_N1327_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.492` `quad=Q2` `gate_open=True` | dist `act_dir=-0.046` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` on the production crosssections time experimental acceptance. Hard-disk production of thesestates have been ruled out over most of the current LHC energy reach.However the LHC has said little about nonperturbative states producedin model with reduced cross sections. I will discuss some popularmodels that have not<< yet>> been ruled out by the LHC experiments`
  - **DISTRACTOR:** ` to the land, began to notice troubling shifts in the environment such as intense drought, frequent wildfires, scarcer fish flows and erratic rainfall. ⏎ There are over 100 government recognized tribes in California, which represent more than 700,000 people. The Karuk is the<< second>> largest Native American tribe in California and has`
- **`p75–90` (`L18_N1327_neg__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.645` `quad=Q2` `gate_open=True` | dist `act_dir=0.078` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=1.0; rule:top_token=1.0; rule:window_has_ge_p97=0.5
  - **FIRING:** `ected mother. Many regions of the world are gaining increased access to complex antiretroviral drug regimens for preventing HIV transmission from a mother to her child. However, these strategies have not yet been directly compared with simpler antiretroviral drug regimens in terms of their safety, efficacy<<,>> feasibility and cost-effectiveness. ⏎ On January`
  - **DISTRACTOR:** `1.98%) fall on the inverse banana curve (figure taken from Ref. #3). ⏎ In contrast, metals such as copper are very robust; however, they do typically not resist large forces. Yet, these materials allow for large deformation, and even the existence of cracks in the material does<< not>> lead to a sudden breakdown. Yet,`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 17/30: `L18_N0809_pos` (Clean Weak-Band Survivor) **[PILOT_20]**
- **Layer / Neuron / Sign:** L18 / N809 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N4784 (gate cos = `0.12328632175922394`)
- **Top-3 trigger tokens:** `[' to', ' of', ' has']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Prepositions and Function Words in Contextual Phrases: Triggers include common prepositions (to, in, since, for, of, through), conjunctions (and), and other function words (has, end, body, planets, He) appearing within idiomatic, temporal, spatial, or attributive phrases, often linking clauses or specifying relationships between entities or events.
- **Wrong label** (donor `L04_N1888_pos`, L4, sim=0.469143): Proper Nouns, Decade Suffixes, and Prepositions: Activations cluster on proper nouns (personal names, place names, organizations), tokens marking decades with '0' (e.g., '1950s', '1870-1900'), and common prepositions ('on', 'of', 'from'). Occasional firing also occurs on month names, demonyms, and rare morphemes within longer words.
- **Token-only label:** `Activates on tokens such as: " to", " of", " has"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.83 | 0.83 | 0.50 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.33 | 0.17 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.33 | 0.00 | 0.17 | 0.33 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.83 | 1.00 | 0.50 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.50 | 0.83 | 0.67 | 0.50 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.83 | 1.00 | 0.67 | 0.58 |
| `rarity` | 0.50 | 0.33 | 0.67 | 0.50 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L18_N0809_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=8.250` `quad=Q1` `gate_open=True` | dist `act_dir=-0.077` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=MISS(A); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` the effect on sponge growth and survival is unknown. Significant declines in sponge health and biomass would be catastrophic to coral reefs, reducing water quality and severely impacting thousands of species from symbiotic microbes to foraging hawksbill turtles. A major loss of sponges would not only negatively impact marine life,<< but>> also local communities that depend on reefs for`
  - **DISTRACTOR:** `, a process that began during each man’s own lifetime. Lincoln Portrait gives us the chance to consider them together, with all the resonance each man’s life and work have for us today. ⏎ - James David Jacobs ⏎ READ MORE AND HEAR LINCOLN PORTRAIT ON-DEMAND ⏎ Comment<< on>> This Article ⏎ News updates from WGB`
- **`p97–99.9` (`L18_N0809_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=3.250` `quad=Q1` `gate_open=True` | dist `act_dir=-0.024` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=OK(B); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Architects of Philadelphia and architect Johnpaul Jones (Cherokee/Choctaw). Disagreements during construction led to Cardinal being removed from the project, but the building retains his original design intent, and his continued input enabled its completion. The museum's project architects are Jones & Jones Architects and Landscape<< Architects>> Ltd. of Seattle and SmithGroup of`
  - **DISTRACTOR:** `3 (2003), ⏎ - Horton, James Oliver. "Alexander Hamilton: Slavery and Race in a Revolutionary Generation" New-York Journal of American History 2004 65(3): 16-24. ISSN 1551-54<<8>>6 ⏎ - Roger G. Kennedy,`
- **`p90–97` (`L18_N0809_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.914` `quad=Q1` `gate_open=True` | dist `act_dir=-0.103` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` the hands. ⏎ Hand-washing is a familiar Jewish ritual: it is, in fact, the first act performed by pious Jews upon awakening in the morning (some people even keep a cup of water next to their beds, so that they may wash their hands before taking even a single step); one<< performs>> a ritual washing of the hands before eating`
  - **DISTRACTOR:** `3 is a block diagram of a data processing system; ⏎ FIG. 4 is a block diagram of a Java virtual machine (JVM); ⏎ FIG. 5 depicts a sample user interface to an applet recorder; ⏎ FIG. 6 is a block diagram illustrating how events are currently handled within Java app<<lets>>; ⏎ FIG. 7 is a block`
- **`p75–90` (`L18_N0809_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.859` `quad=Q1` `gate_open=True` | dist `act_dir=0.091` `quad=Q1`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` 04:07 AM ⏎ Don't worry about it gothic_type, I can see where I sort of rambled on in my post and just messed up what I was trying to point out. ⏎ I'll work off of your code since it has the basic framework and I want to<< save>> typing... ⏎ *Opens up DevC++`
  - **DISTRACTOR:** ` subdividing it into panels, directing the reader-viewer powerfully right-to-left across the image. In Inko’s view, "The beauty of asymmetry is much admired in Japanese artistic traditions, so ukiyo-e compositions create various ‘eye streams’ to avoid a symmetrical layout." In<<ko>> sees parallels here with kabuki theatre.`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 18/30: `L04_N5325_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N5325 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N1345 (gate cos = `0.18108201026916504`)
- **Top-3 trigger tokens:** `[' won', ' for', ' gain']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.95, p90–97: Q3=0.93, p97–99.9: Q3=0.67, p99.9+: Q3=0.00
- **Own label:** Verbs of Acquisition, Benefit, and Exchange: Triggers include verbs denoting obtaining, receiving, or achieving something (earn, gain, win, avail, gain, gained, earnings, won, asked), as well as prepositional phrases with 'for' indicating purpose or exchange. Activation is strongest on present and past tense verb forms related to acquisition or transactional contexts.
- **Wrong label** (donor `L11_N4693_pos`, L11, sim=0.340622): Abbreviations, Ranges, and Domain-Specific Terms: Activations cluster on abbreviations, single-letter or hyphenated range notations, and domain-specific nouns or suffixes (e.g., -ry, -art, -arm, -ology) in scientific, academic, or technical contexts, as well as on list or navigation delimiters such as vertical bars or double pipes.
- **Token-only label:** `Activates on tokens such as: " won", " for", " gain"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.83 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.67 | 0.00 | 0.00 | 0.33 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.33 | 0.67 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.83 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 1.00 | 0.67 | 1.00 |
| `rarity` | 0.67 | 0.50 | 0.67 | 0.83 |
| `top_token` | 0.50 | 0.50 | 0.42 | 0.67 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N5325_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.766` `quad=Q1` `gate_open=True` | dist `act_dir=-0.018` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` support organizations and communities that are working to regain control of and sustainably manage their traditional territories through land title initiatives, community education, development of sustainable economic alternatives, and grassroots resistance to destructive industrial activities. ⏎ PAA is an alternative to “buy-an-acre” programs that seek to provide rainforest protection by<< buying>> tracts of land, but which often`
  - **DISTRACTOR:** ` particles were small and soft. Breast milk registered 0 curd tension, natural cows' milk 50-100 grams, and homogenized milk (processed at 3500 pounds) 15 grams. The patients reported that the milk tasted better (because of the<< minute>> division of the milkfat globules)`
- **`p97–99.9` (`L04_N5325_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.412` `quad=Q1` `gate_open=True` | dist `act_dir=-0.023` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` out of season, that this love and passionate longing for his native country burst forth in words and musings; the narratives of his most trustworthy biographers are full of it. The most severe penance which he could have imagined for the guiltiest sinners who came to confess to him, was to<< impose>> upon them the same fate which he had`
  - **DISTRACTOR:** ` and it is integrated as part of basic education in secondary school curricula across the world. ⏎ The primary intention of EHL is to help young people embrace the principles of humanity in their daily lives and in the way they assess events at home and abroad. In particular, outcomes should take account of positive changes<< in>> levels of: ⏎ The EHL curriculum is`
- **`p90–97` (`L04_N5325_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.170` `quad=Q3` `gate_open=False` | dist `act_dir=-0.015` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `From 2001 to 2004, scientists collected and tested the soil from 35 different sites in North Carolina. According to the North Carolina scientists, the collected soil samples were representative of millions of hectares in agricultural production in the southeastern USA. Corn was planted at each site<< with>> a range of nitrogen fertilizer rates, and`
  - **DISTRACTOR:** ` (81 to 95 °F), and in winter 0 to -15 °C (32 to 5 °F), with extremes ranging from about 42 °C (108 °F) in summer to −29 °C (−20 °<<F>>) in winter. Average yearly rainfall is`
- **`p75–90` (`L04_N5325_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.118` `quad=Q3` `gate_open=False` | dist `act_dir=0.003` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` on to food manufacturers across the country. In February 2006, 20 kg of chicken tongues from China were found by customs in Rio de Janeiro, Brazil, and 21 tonnes of (mainly) poultry meat from China were confiscated in southern Spain. These indicate continuing laps<<es>> in border controls, despite the widely public`
  - **DISTRACTOR:** `our, 928.5 miles/hour or 806.9 knots.The table below shows the Moon position compared to the time and date above: ⏎ |Time||Longitude difference||Latitude difference||Total| ⏎ |1 minute||0° 14' 29.7<<">>||15.48 mi||`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 19/30: `L18_N2188_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L18 / N2188 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N1276 (gate cos = `0.4372830390930176`)
- **Top-3 trigger tokens:** `[' name', ' the', ' a']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.99, p90–97: Q3=0.98, p97–99.9: Q3=0.69, p99.9+: Q3=0.00
- **Own label:** Etymology and Naming Explanations: Activates on tokens introducing explanations of word origins, meanings, or naming—especially in phrases like 'the name', 'the term', 'named after', and when defining or translating words from other languages or contexts.
- **Wrong label** (donor `L24_N1157_neg`, L24, sim=0.402922): Tokens with 'pen', 'hen', 'hem', or similar letter clusters: Strong activation occurs on tokens containing the letter sequence 'pen', 'hen', 'hem', or similar clusters, whether as standalone words, within names, or embedded in longer words; additional activations appear for tokens with 'pel', 'geo', 'ud', and some numerical or grammatical function words.
- **Token-only label:** `Activates on tokens such as: " name", " the", " a"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.67 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.67 | 0.33 | 0.83 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.17 | 0.33 | 0.17 | 0.83 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.33 | 0.67 | 0.50 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 0.50 | 0.50 | 1.00 |
| `rarity` | 0.33 | 0.33 | 0.33 | 1.00 |
| `top_token` | 0.58 | 0.50 | 0.42 | 0.58 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L18_N2188_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=22.750` `quad=Q1` `gate_open=True` | dist `act_dir=0.101` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` to stop the beating of an animal.) (Asimov 535)Due partly to his belief in metempsychosis, he opposed the taking of life, the eating of flesh, and association with those who benefit by the slaughter business. He supposedly coined the term "Philosophy<<">> first as a word to signify the love`
  - **DISTRACTOR:** ` of its surface. On the opposite borders of the largest of these basins, much shrunk in its dimensions since the days of the Aztecs, stood the cities of Mexico and Tezcuco, the capitals of the two most potent and flourishing states of Anahuac, whose history,<< with>> that of the mysterious races that preceded them`
- **`p97–99.9` (`L18_N2188_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.125` `quad=Q3` `gate_open=False` | dist `act_dir=0.069` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.0; rule:top_token=0.0; rule:window_has_ge_p97=1.0
  - **FIRING:** ` screen. ⏎ - Awareness of the logical thinking and problem solving involved in using the application. ⏎ Participants will learn how to ⏎ - Program and sequence instructions. ⏎ - Use Scratch in the classroom. ⏎ Course content and methodology ⏎ - Demonstration of the application. ⏎ - Practical exploration of the application. ⏎ - Discussion of how this<< can>> be used to support learning. ⏎ - Planning`
  - **DISTRACTOR:** ` that projected massive changes in plant communities across nearly half of the earth’s land surface, with “the conversion of nearly 40 percent of land-based ecosystems from one major ecological community type – such as forest, grassland or tundra – toward another.” ⏎ The NASA announcement added: ⏎ The model projections paint<< a>> portrait of increasing ecological change and stress in`
- **`p90–97` (`L18_N2188_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.707` `quad=Q3` `gate_open=False` | dist `act_dir=-0.046` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `From the Atlas and Cyclopedia of Ireland (1900) ⏎ It was to the rugged and desolate Hebrides that Columba turned his face when he accepted the terrible penance of Molaise. He bade farewell to his relatives, and, with a few monks who insisted on accompany<< him>> whithersoever he might go`
  - **DISTRACTOR:** ` track the changes in a bacterial population following the introduction of a vaccine. The study follows how th... Read More ⏎ Certain types of anti-depressants have been linked to an increase in the risk of Clostridium difficile infection (CDI) finds a study in BioMed Central's open access<< journal>> BMC Medicine. Awareness of this link should`
- **`p75–90` (`L18_N2188_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.527` `quad=Q3` `gate_open=False` | dist `act_dir=0.022` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=1.0; rule:top_token=1.0; rule:window_has_ge_p97=0.5
  - **FIRING:** ` Salvage and Utilization ⏎ Timber salvage and utilization in the inland northwest. [ ⏎ Tech Notes No. ⏎ 8 (PDF, 194k) ] ⏎ by S. Willits, C. Brunner, and J.J. Morrell ⏎ This technical note discusses the use of salvage harvest as<< a>> tool ⏎ to accomplish forest restoration objectives in`
  - **DISTRACTOR:** `is". International Journal of Immunogenetics 34 (6): 457–463. doi:10.1111/j.1744-313X.2007.00723.x. PMID 18<<0>>01303. ⏎ - Enc`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 20/30: `L24_N3118_pos` (Clean Weak-Band Survivor) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N3118 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N584 (gate cos = `0.11658117920160294`)
- **Top-3 trigger tokens:** `[' S', ' B', '-b']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.98, p90–97: Q3=0.95, p97–99.9: Q3=0.62, p99.9+: Q3=0.00
- **Own label:** Capitalized S, B, and Z at Word Start: Activations are triggered by tokens representing uppercase S, B, or Z at the beginning of proper nouns, surnames, place names, or abbreviations, often following whitespace or punctuation. Occasional activation also occurs on hyphenated forms where these letters follow a hyphen.
- **Wrong label** (donor `L18_N0783_pos`, L18, sim=0.399451): Common English Function Words and Past Tense Verbs: Strongest activations occur on high-frequency function words such as 'to', 'of', 'they', 'the', and on past tense verb forms ending in '-ed' or irregular forms like 'was', 'became', 'sold', 'hit', 'reached', 'succeed', often in narrative or expository contexts.
- **Token-only label:** `Activates on tokens such as: " S", " B", "-b"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.83 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.83 | 0.00 | 0.17 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.33 | 0.83 | 1.00 | 0.83 |
| `gpt-6.1-terra@none` `token_only` | 0.67 | 0.83 | 0.67 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.67 | 1.00 | 1.00 |
| `rarity` | 0.17 | 0.67 | 0.50 | 0.50 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.83 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N3118_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=52.000` `quad=Q1` `gate_open=True` | dist `act_dir=0.007` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=1.0; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` by Zimbabweans (28,548) ⏎ A report by the SAHRC notes that xenophobia underpins police action against foreigners. People are apprehended for being “too dark” or “walking like a black foreigner”. Police also regularly destroy documents of black non-South Africans. ⏎ <<S>>udanese refugee James Diop is`
  - **DISTRACTOR:** ` we have seen, in a purely theoretical way, denying that Atman or Brahman underlies the flux of events. They also see this theory as leading people to depreciate that actual events of the world viewing them as appearance and the appearance as illusory. The Buddhist seeks to realize the existential<< meaning>> of the fact that there is no At`
- **`p97–99.9` (`L24_N3118_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.562` `quad=Q3` `gate_open=False` | dist `act_dir=0.001` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` The gas builds up under the ice until it is highly pressurised and finally breaks through the surface layer. ⏎ As the gas escapes, it carries the smaller, finer particles of the soil along with it, forming grooves under the ice. This "spider" effect indicates a spot where a g<<ey>>ser is established, and will form again`
  - **DISTRACTOR:** ` While people can clearly see that the nation, at least geographically, was divided, I wonder how many are actually unable to see that the people within those areas were divided. A fine example, I think, is the way that some want to say that the “Southern perspective of the war” equ<<ates>> to the “Confederate perspective of`
- **`p90–97` (`L24_N3118_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.316` `quad=Q3` `gate_open=False` | dist `act_dir=-0.007` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` scales, from nano to macro, by using systematic experimental, theoretical or computational methods6,7. ⏎ In order to provide a bottom-up description of materials behavior, our interdisciplinary team of undergraduate students, graduate students and postdocs applies an experimentally validated atomistic based multi-scale simulation approach that considers the structure<<-process>>-property paradigm of materials science and the architecture`
  - **DISTRACTOR:** ` understands something about heaving, but early yesterday afternoon he didn't think so. In fact, he did not think much of himself. If he did think it couldn't have been much, for the champions slammed him for eight runs and six hits in the first two innings. This was naturally all he<< wanted>>, and he was glad to get under`
- **`p75–90` (`L24_N3118_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.101` `quad=Q3` `gate_open=False` | dist `act_dir=-0.003` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` danger (except perhaps from falling off my chair), forty-six years later, I experienced similar problems. If I moved my arm one way, my virtual body spun the other. The normal rules of movement that we are accustomed to on Earth do not apply in space. Imagine the simple act of tightening a<< bolt>> – without something to push against, as`
  - **DISTRACTOR:** `ly considered, perhaps, as deeply, as finally, staked on the experiment entrusted to the hands of the American people. ⏎ Besides the ordinary objects submitted to your care, it will remain with your judgment to decide how far an exercise of the occasional power delegated by the fifth article of the Constitution is rendered<< exped>>ient at the present juncture by`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 21/30: `L04_N3979_pos` (High-Q3 Double-Negative Pole) **[PILOT_20]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3979 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N6094 (gate cos = `0.13262277841567993`)
- **Top-3 trigger tokens:** `['1', '9', '3']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=1.00, p97–99.9: Q3=1.00, p99.9+: Q3=0.99
- **Own label:** Numbers, Ordinals, and Scientific Suffixes: Activations cluster on tokens within multi-digit numbers, ordinal or code-like digit sequences, and scientific or technical suffixes such as -ble, -obe, -oph, and -con, as well as common prepositions and conjunctions in close proximity to numbers or lists.
- **Wrong label** (donor `L11_N2071_pos`, L11, sim=0.582731): Abstract Nouns and Technical Terms Ending in -ion, -ing, -age, -x, -use: Activations cluster on abstract nouns and technical terms, especially those ending with suffixes like -ion, -ing, -age, -x, -use, and -ness, often denoting concepts, processes, or qualities in formal or explanatory contexts.
- **Token-only label:** `Activates on tokens such as: "1", "9", "3"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.83 | 0.67 | 0.83 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.50 | 0.50 | 0.33 |
| `gpt-6.1-terra@none` `no_label` | 0.33 | 0.67 | 1.00 | 0.50 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.50 | 0.83 | 0.67 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.83 | 0.83 | 0.83 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.92 | 0.67 | 1.00 | 0.67 |
| `rarity` | 0.25 | 0.33 | 0.50 | 0.50 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.58 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3979_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.480` `quad=Q3` `gate_open=False` | dist `act_dir=0.010` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=MISS(A); gpt-6.2-sol@low/own[rev]=MISS(B); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=MISS(B); gpt-6.2-sol@low/wrong[fwd]=OK(B); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` David Sequeira from Chorotegan pottery excavated by him in Nicaragua will be shown at the Brooklyn Museum from December 1 through February 4 in an exhibition arranged by Fr. Herbert J. Spinden, Curator of American Indian Art and Primitive Cultures. Sequira is of Nicar<<ag>>uan birth and primarily a musican whose`
  - **DISTRACTOR:** ` laws are established by acts of congress. Laws are also called statutes and are enforced by regulations. ⏎ Federal discrimination laws make it illegal for employers to discriminate against employees and job applicants in any aspect of employment based on age, disability, national origin, race, religion, genetics or sex (gender). ⏎ The U<<.S>>. Equal Employment Opportunity Commission (EEO`
- **`p97–99.9` (`L04_N3979_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.229` `quad=Q3` `gate_open=False` | dist `act_dir=-0.010` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=OK(B); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` oscillate about the primary optic axis to maintain focus despite vibration inherent from the Be scan mirror. This engineering solution allowed the US to develop LANDSAT at least five years ahead of French SPOT which first used CCD arrays to stare without need for a scanner. ⏎ The MSS FPA, or Focal Plane<< Array>> consisted of 24 square optical fibers`
  - **DISTRACTOR:** ` sanctioned, and comparison groups are selected carefully, the technical psychometric points raised by critics should not swamp the incentive and information benefits of performance-based compensation plans for teachers. If we are ready to determine reading programs, language labs, class sizes, and the use of computerized learning on the basis of value<<-added>> assessments, we should be ready to reward`
- **`p90–97` (`L04_N3979_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.166` `quad=Q3` `gate_open=False` | dist `act_dir=0.015` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=MISS(A); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=0.0; rule:top_10_token=1.0; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` could travel with her husband and brothers to Texas on the eve of the Mexican War. They settled in Corpus Christi until Taylor moved his army to the Rio Grande valley at the beginning of the war. By this time Mary was working as a hospital matron. Matthew and Tom Nolan were at the battles<< of>> Palo Alto and Resaca de la Pal`
  - **DISTRACTOR:** ` in Java. According to a preferred embodiment, an automator is attached to a Java applet. Responsive to selection by a user, listeners are added for each event type produced in the Java applet. Each time a specified event occurs, that event is captured and saved to a data structure. The<< recording>> of events is performed until the user stops`
- **`p75–90` (`L04_N3979_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.105` `quad=Q3` `gate_open=False` | dist `act_dir=0.009` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `pin primary education policy and classroom practice ⏎ - an awareness of current theories and research that support primary school practice ⏎ - insights into issues relating to race equality and social inclusion, and the implications for practice ⏎ - an appreciation of the complementary nature of home and school learning and the role of parents and the family<< in>> children’s education ⏎ - academic skills related`
  - **DISTRACTOR:** ` materials along with our increasing ability to tailor the composition and structure of materials in a cost-effective manner. Today's technological revolution would not be possible without the continuing increase in our scientific understanding of materials and phenomena, along with the processing and synthesis required for high-volume, low-cost manufacturing. This article examines selected<< examples>> of the scientific and technological impact of electronic`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 22/30: `L04_N6043_pos` (High-Q3 Double-Negative Pole) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N6043 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N3154 (gate cos = `0.14220476150512695`)
- **Top-3 trigger tokens:** `[' to', ' entered', ' press']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=1.00, p97–99.9: Q3=0.99, p99.9+: Q3=0.68
- **Own label:** Inflected Verbs, Technical Nouns, and Proper Nouns: Triggers include English verbs in past participle or infinitive forms (entered, press, inserted, provide, enter, sorted, supply, quit), technical or scientific nouns (keyboard, pills, glutamate, fac, register), and capitalized proper nouns or names (Friendly, Rhode, Okello).
- **Wrong label** (donor `L24_N3118_neg`, L24, sim=0.427974): Mid-token Capital Letters and Scientific Abbreviations: Activations cluster on tokens with internal or initial uppercase letters, especially in scientific names, abbreviations, and proper nouns, including chemical symbols, acronyms, and names with mid-token capitalization or uncommon capitalization patterns.
- **Token-only label:** `Activates on tokens such as: " to", " entered", " press"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.00 | 0.50 | 0.67 |
| `gpt-6.1-terra@none` `wrong` | 0.17 | 0.17 | 0.33 | 0.50 |
| `gpt-6.1-terra@none` `no_label` | 0.33 | 0.00 | 0.67 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.67 | 0.50 | 0.50 | 0.67 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.17 | 0.00 | 0.33 | 0.67 |
| `rarity` | 0.33 | 0.33 | 0.33 | 0.58 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N6043_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.383` `quad=Q3` `gate_open=False` | dist `act_dir=-0.004` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` development, says new UNFPA report. – UNFPA REPORT ⏎ TAKE A BOW Secretary Clinton, who began this push in 1995 as first lady when she declared human rights are women’s rights, which begins with planning your family. The UNFPA report is non<<-binding>> and won’t impact law internationally, but`
  - **DISTRACTOR:** `LGs provides a spectrum (shown above) similar to that produced by the WMAP project. The peak in the spectrum gives "smoking gun" evidence that the clumpiness seen in the WMAP data grows into the galaxies and galaxy clusters imaged by SDSS by straightforward gravitational processes, as<< predicted>> by big bang models. Further, the`
- **`p97–99.9` (`L04_N6043_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.130` `quad=Q3` `gate_open=False` | dist `act_dir=-0.013` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `3 2000 17.74% done, estimate finish Wed Sep 13 21:28:00 2000 23.64% done, estimate finish Wed Sep 13 21:28:0<<3>> 2000 ... 8`
  - **DISTRACTOR:** ` be true. However, since the bird allegedly had a vocabulary of some 950 words at the time of the test, omitting sessions where the bird said nothing or said something not on the key list, is unjustifiable. Furthermore, there is no evidence that it is reasonable to assume that<< when>> the parrot is by itself uttering`
- **`p90–97` (`L04_N6043_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.087` `quad=Q3` `gate_open=False` | dist `act_dir=-0.010` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` where all of human knowledge is written, edited, and re-drafted by somewhat anonymous individuals the world over. On Wikipedia, information is created, destroyed, and refined by millions of contributors. ⏎ You know what the crazy thing is? It works. ⏎ Yes, not all information is perfect, but the resource<< that>> Wikipedia has become equates into billions of`
  - **DISTRACTOR:** ` infection in fig wasp communities from two continents. Proc. R. Soc. B 272, 421-429. ⏎ Hatcher MJ, Dunn AM & Tofts C. 2000. Co-existence of hosts and sex ratio distorters in structured<< populations>>. Evolutionary Ecology Research 2,`
- **`p75–90` (`L04_N6043_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.069` `quad=Q3` `gate_open=False` | dist `act_dir=0.010` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` higher in extremely obese girls. ⏎ The same association was not found in boys. ⏎ Once considered rare in children, multiple sclerosis ⏎ has become more common, especially in teenage girls. Researchers say an increase in childhood obesity may be one cause. ⏎ I'm Dr. Cindy Haines of HealthDay TV with the news<< doctors>> are reading health news for healthier living.`
  - **DISTRACTOR:** ` was suppressed, nationally based anti-Semitism increased over time. The Slovakian intelligentsia, which advocated a Slovakian national revival, regarded the Jews as tools of the hated Hungarian regime and, therefore, one of the causes of the plight of the Slovakian people. After Hungarian Jewry attained equal rights<< in>> 1867, anti-Semitic`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 23/30: `L04_N5913_pos` (High-Q3 Double-Negative Pole) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N5913 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N1090 (gate cos = `0.24552294611930847`)
- **Top-3 trigger tokens:** `[' feeding', 'chr', 'ipp']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=1.00, p97–99.9: Q3=0.99, p99.9+: Q3=0.88
- **Own label:** Proper Nouns, Technical Terms, and Function Words: Activations cluster on proper nouns (names, places, surnames), technical or domain-specific terms (e.g., memchr, TMGB, TALENs), and common function words ('of', 'actually'). Morphological triggers include suffixes like -ing, -ias, -able, and -bar, as well as capitalization patterns typical of names and acronyms.
- **Wrong label** (donor `L24_N5570_pos`, L24, sim=0.595588): Uppercase G and Suffixes with 'g', 'air', 'ats', 'as', 'y': Strong activation occurs on uppercase 'G' at the start of names or words, as well as on tokens containing or ending with 'g', 'air', 'ats', 'as', or 'y', often within proper nouns, surnames, or scientific/technical terms.
- **Token-only label:** `Activates on tokens such as: " feeding", "chr", "ipp"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.67 | 0.50 | 0.50 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.67 | 0.83 | 0.50 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.33 | 0.50 | 0.83 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.33 | 0.67 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.67 | 0.83 | 0.50 |
| `rarity` | 0.50 | 0.75 | 0.83 | 1.00 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N5913_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.531` `quad=Q3` `gate_open=False` | dist `act_dir=0.005` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` At least at first, patients should be treated in a hospital. ⏎ Treatment often lasts for several months, and can have serious side effects. ⏎ As of Oct. 9, the CDC reports deaths and cases in these states: ⏎ SOURCES:Curtis Allen, public information officer, CDC, email interview.C<<DC>> web site.FDA web site.David`
  - **DISTRACTOR:** ` are tremendous opportunities for improvements in productivity.” ⏎ Several provisions of the 2010 health care law aim to reduce inefficiencies by rewarding higher-value care and reducing some of the incentives for doctors to deliver more testing and care with unproven benefits. Such policies may serve to reduce health care employment,<< the>> authors argue, by “allowing us`
- **`p97–99.9` (`L04_N5913_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.179` `quad=Q3` `gate_open=False` | dist `act_dir=-0.008` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` delays, check out these resources from teachmetotalk.com: ⏎ For a more structured way to teach a child to understand and use more words, I highly recommend Teach Me To Talk: The Therapy Manual. While this book was written for therapists, many moms who are committed to providing the bulk of their<< child>>’s therapy at home have told me that`
  - **DISTRACTOR:** `. Having decided that Sir Politic will believe anything anyone tells him, as his name indicates, Peregrine proceeds to let him tell some more improbable stories for his and the audience's amusement, including the one about Mas' Stone, the supposed drunken illiterate who Politic is convinced was a dangerous<< spy>>. According to Politic, Stone had`
- **`p90–97` (`L04_N5913_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.099` `quad=Q3` `gate_open=False` | dist `act_dir=0.011` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `- Who or what factors do European observers, British officials, and Indian opponents of sati hold to be responsible for the continuance of the practice of sati? ⏎ - What were the reasons that widows gave for committing sati? Were they religious, social or material motives? What is the evidence<< that>> the widows were voluntarily committing sati`
  - **DISTRACTOR:** `. While police claimed he was "trying to escape" and received his injuries after falling, Mandlate stated he fled after police threatened him with a gun. Police gave chase, beat Mandlate, and threw him in a cell with other inmates for several hours before he was rescued by colleagues and taken to<< the>> hospital for treatment of head injuries. There`
- **`p75–90` (`L04_N5913_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.081` `quad=Q3` `gate_open=False` | dist `act_dir=0.006` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `kills provide the majority of New York City's drinking water supply; CERM forums help coordinate research and identify research agendas to protect these resources. ⏎ In November, Cary Institute will hold a two-day conference examining the effects of climate change on plant, animal, and microbial species. The invitation-only event is being<< organized>> by Richard Ostfeld, Shannon LaDe`
  - **DISTRACTOR:** ` This will ensure that as Asia achieves green and sustainable development, our people will enjoy comprehensive development. ⏎ Ladies and Gentlemen, ⏎ We demonstrated solidarity as we rose up together to the international financial crisis in 2009. Let us carry forward this great spirit, build up consensus, strengthen unity and<< cooperation>> and explore a path of green and sustainable`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 24/30: `L04_N3437_pos` (High-Q3 Double-Negative Pole) **[HELDOUT_80]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N3437 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N4183 (gate cos = `0.20746928453445435`)
- **Top-3 trigger tokens:** `['a', 'ium', 'den']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=1.00, p97–99.9: Q3=0.99, p99.9+: Q3=0.49
- **Own label:** Chemical Elements and Scientific Suffixes: Triggers include chemical element names and symbols (e.g., 'ium', 'den', 'U', 'Uran', 'Y', 'Meredith'), scientific or technical suffixes ('-on', '-ant', '-ing', '-by', '-ium', '-den'), and terminology from chemistry, physics, or biology contexts. Activation is strongest on tokens forming or completing element names, scientific terms, or technical jargon.
- **Wrong label** (donor `L11_N1066_neg`, L11, sim=0.52892): Common English Suffixes and Function Words: Activations cluster on tokens with frequent English suffixes such as -ness, -ary, -ors, -ed, -own, -loss, and -views, as well as high-frequency function words like 'as', 'a', 'into', and 'another', often at the start or end of noun or verb phrases.
- **Token-only label:** `Activates on tokens such as: "a", "ium", "den"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.83 | 1.00 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.33 | 0.50 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.67 | 0.67 | 0.83 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 1.00 | 0.50 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.83 | 0.83 | 0.83 |
| `rarity` | 0.33 | 0.83 | 0.67 | 0.83 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N3437_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.484` `quad=Q3` `gate_open=False` | dist `act_dir=0.010` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `. He successfully used the subsequent trial to gain publicity for himself and his ideas. Following his trial, he was sentenced to five years in Landsberg prison, but served only six months. ⏎ During his prison term, Hitler began to dictate his thoughts and philosophy to co-conspirator and cellmate Rudolf<< Hess>>, which became the book Mein Kampf`
  - **DISTRACTOR:** `thers. ⏎ In his 30 years of life, Kamrul has never been well-treated by people, except for his mother. When he first came in touch with the Compassion staff, he was amazed by their compassionate attitude. Now Kamrul is astonished by your love for him and his<< family>>. ⏎ “From my childhood I got used`
- **`p97–99.9` (`L04_N3437_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.344` `quad=Q3` `gate_open=False` | dist `act_dir=0.018` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` offshore facilities and 10,000 miles of pipelines were in the direct path of Ivan. Seven platforms were destroyed and 24 others damaged. The oil and natural gas industry submitted numerous damage reports to MMS, including for mobile drilling rigs, offshore platforms, producing wells, topside<< systems>> including wellheads and production and processing equipment`
  - **DISTRACTOR:** `led numerals. ⏎ - Have him add up the circled numbers. ⏎ - Have him unfold your answer - 15! ⏎ Jump the Coin ⏎ - Bet your friend that he will not be able to jump over this coin when you set it down. ⏎ - Have him stand with his arms stright down<< at>> his sides and then place the coin on`
- **`p90–97` (`L04_N3437_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.144` `quad=Q3` `gate_open=False` | dist `act_dir=-0.002` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` server as a node. In addition to providing a heartbeat, BrightStor High-Availability Manager lets cluster nodes ping other network servers or routers to verify network connectivity. BrightStor High-Availability Manager doesn't provide application monitoring, but you can use other systems management applications to alert BrightSt<<or>> High-Availability Manager if an application`
  - **DISTRACTOR:** ` out of reach. ⏎ ___ Store dishwasher detergent and other cleaning supplies in their original containers and out of reach. ⏎ ___ When children are present, install safety latches on cabinets and drawers within a child's reach. ⏎ ___ Keep chairs and step stools away from counters and stoves. ⏎ ___ Always turn pot handles inward<< when>> cooking on the stove. Use back burn`
- **`p75–90` (`L04_N3437_pos__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.066` `quad=Q3` `gate_open=False` | dist `act_dir=0.018` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` located a few miles south of San Clemente, California. If San Onofre were to melt down, it would contaminate Los Angeles, San Diego, Orange County, and make large regions of southern California uninhabitable forever. As background, read Shut down San Onofre: The<< continuing>> nuclear threat to southern California and Bad Vibr`
  - **DISTRACTOR:** ` child, a 504 Plan or ESE program should be addressed. A physical and occupational therapist would be beneficial in evaluating and planning for accessibility and adaptation needs in the school. ⏎ Below is a list of some accommodation that other EDS students have found to be beneficial. ⏎ Physical education/sports needs<<: ⏎ >>Modified physical education ⏎ Alternative health related credit`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 25/30: `L11_N4363_pos` (High-Q3 Double-Negative Pole) **[PILOT_20]**
- **Layer / Neuron / Sign:** L11 / N4363 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N271 (gate cos = `0.22967343032360077`)
- **Top-3 trigger tokens:** `['1', '||', ' reaction']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=1.00, p97–99.9: Q3=0.99, p99.9+: Q3=0.85
- **Own label:** Common English Nouns and Numbers in Context: Activations cluster on frequent English nouns (e.g., 'reaction', 'waves', 'exercise', 'games', 'hair', 'mother', 'schedule', 'the', 'with', 'your', 'open') and on numerals or years (e.g., '2', '4', '15', '103', '10', 'thirty'), often in contexts involving lists, headings, or factual statements.
- **Wrong label** (donor `L04_N2467_neg`, L4, sim=0.395127): Prepositions and Supply/Provision Verbs: Triggers include prepositions such as 'with', 'of', 'from', and 'for', as well as verbs and nouns related to supplying, providing, or furnishing resources, information, or materials. Activation often occurs on the preposition or the verb/noun denoting provision in contexts of distribution or assignment.
- **Token-only label:** `Activates on tokens such as: "1", "||", " reaction"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.50 | 0.67 | 0.50 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.00 | 0.17 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.17 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.33 | 0.83 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.83 | 0.67 | 0.67 | 0.67 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.50 | 0.42 | 0.67 | 0.42 |
| `rarity` | 0.33 | 0.17 | 0.50 | 0.67 |
| `top_token` | 0.50 | 0.50 | 0.58 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N4363_pos__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.664` `quad=Q3` `gate_open=False` | dist `act_dir=-0.008` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` for those interested in the presentation of mathematics at the K-9 levels. Topics emphasized are problem solving, patterns, sequences, set theory, logic, numeration systems, number bases, arithmetic operations, and number theory. Various manipulative and problem solving strategies are used. Prerequisite: MTH <<9>>5 or higher, and WR 1`
  - **DISTRACTOR:** ` public perception of climate change and the reality of the change that has already begun. ⏎ All of these gaps are interrelated. ⏎ In 2009 countries made pledges to reduce greenhouse gas emissions in order to keep global average temperature from increases below 2 degrees Celsius. Last week the United Nations Environment Programme<< issued>> “The Emissions Gap Report: Are`
- **`p97–99.9` (`L11_N4363_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.340` `quad=Q3` `gate_open=False` | dist `act_dir=-0.005` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=MISS(A); gpt-6.2-sol@low/own[rev]=MISS(B); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` it was an Ark for scrolls of the Law. The whole floor of the building is paved with mosaics. ⏎ Two inscriptions were found at the entrance to the hall. One, in Aramaic, states that the mosaics were made during the reign of Emperor Justin (518<<-->>527). The other, in`
  - **DISTRACTOR:** `0 percent minimum tax on people with million-dollar incomes — known as the "Buffett rule" — would do much to reduce the deficit or boost the economy. But the Obama administration says the proposal is necessary to make the tax code more equitable. ⏎ The Buffett rule would apply to wealthy people, but not<< all>> wealthy people. Instead, it would affect`
- **`p90–97` (`L11_N4363_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.243` `quad=Q3` `gate_open=False` | dist `act_dir=-0.019` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `anir, Ph.D., a Howard Hughes Medical Institute investigator and chair of the Solomon H. Snyder Department of Neuroscience at Hopkins, "we’ve long known that LTD underlies responses like the vestibulo-ocular reflex. ⏎ This study gets at the heart of how LTD occurs, specifically how PICK1 controls<< the>> Purkinje cell’s response to the`
  - **DISTRACTOR:** `, agreed to pay a royalty of ten per cent of all crude petroleum removed from the ground and fifty dollars a year for each gas well, as long as it was used. The royalty was to be based on the market value at the place of production, and was to be paid to the National Treasurer<< of>> the Osage Nation. Foster further agreed`
- **`p75–90` (`L11_N4363_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.101` `quad=Q3` `gate_open=False` | dist `act_dir=-0.018` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=MISS(B); gpt-6.2-sol@low/no_label[rev]=MISS(A); gpt-6.2-sol@low/own[fwd]=MISS(B); gpt-6.2-sol@low/own[rev]=MISS(A); gpt-6.2-sol@low/token_only[fwd]=MISS(B); gpt-6.2-sol@low/token_only[rev]=MISS(A); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` modeled monsoon depressions and published their findings in the Quarterly Journal of the Royal Meteorological Society. Future studies will incorporate more simulations and more advanced models to test the ability of Doppler radar to track monsoon processes. Niyogi said the techniques and tools being developed also could help predict landfall<< of>> tropical storm systems that affect the Caribbean and`
  - **DISTRACTOR:** ` “That is the paradox,” she said. ⏎ Web sites can disappear in a matter of days or change repeatedly in a matter of hours. Files can become lost or corrupted, formats and hardware change, and physical media such as tapes and disks deteriorate far more quickly than anticipated. ⏎ So Congress charged the LOC<< in>> 2000 with preserving the`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 26/30: `L24_N3193_pos` (High-Floor / Judge-Disagreement Notable) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N3193 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N4773 (gate cos = `0.24910426139831543`)
- **Top-3 trigger tokens:** `['bl', 'Inv', 'orn']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.99, p90–97: Q3=0.97, p97–99.9: Q3=0.87, p99.9+: Q3=0.00
- **Own label:** Foreign Proper Nouns and Loanwords: Activations cluster on tokens within or at the start of foreign-derived proper nouns, publisher names, and loanwords, especially those with initial capital letters or uncommon letter sequences (e.g., 'bl', 'omb', 'ie', 'ne', 'Ade', 'Jan', 'gen'). These tokens often appear in bibliographic, academic, or encyclopedic contexts.
- **Wrong label** (donor `L11_N1576_neg`, L11, sim=0.352064): Units of Measurement, Quantities, and Numeric Expressions: Triggers include tokens denoting units (e.g., pressure, currency, length, mass, time, distance), numeric fractions, and words marking quantities or measures, often following numbers or quantifiers. Also activates on prepositions and determiners directly preceding or following such expressions.
- **Token-only label:** `Activates on tokens such as: "bl", "Inv", "orn"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.67 | 0.83 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.67 | 0.83 | 0.67 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 1.00 | 0.67 | 1.00 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.83 | 0.67 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.67 | 0.83 | 1.00 |
| `rarity` | 0.17 | 1.00 | 0.33 | 0.33 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N3193_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=9.438` `quad=Q1` `gate_open=True` | dist `act_dir=0.000` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `ab Republic with Egypt in 1958. It was decided that the national anthem of ⏎ the UAR would be a combination of the then-Egyptian anthem and "Homat elDiyar" When Syria seceded from the union in 1961, the ⏎ <<anth>>em was completely restored. ⏎ Translation of Syrian`
  - **DISTRACTOR:** ` The constitution also prohibits discrimination based on religion, and minority religions experienced little or no official discrimination. Buddhism is the state religion, and more than 93 percent of the population is Buddhist. Ethnic Cham Muslims constitute most of the remaining population. ⏎ The law requires all religious groups, including Buddhists, to<< submit>> applications to the Ministry of Cults and`
- **`p97–99.9` (`L24_N3193_pos__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=1.594` `quad=Q3` `gate_open=False` | dist `act_dir=0.008` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` (ARS) scientists have found environmentally and economically sound uses for the byproducts of biofuel production. ⏎ Animal scientist William Dozier, formerly with the ARS Poultry Research Unit in Mississippi State, Miss., has been working with colleagues at the ARS Swine Odor and Manure Management Research Unit<< in>> Ames, Iowa, and Iowa State University`
  - **DISTRACTOR:** `999) who claimed to show that prehension involves the independent targeting of the thumb and forefinger, and thus you don't need object size. ⏎ - He then suggests that DF is generally able to reach successfully because she has access to the necessary egocentric information (in hand-centred<< coordinates>>) about the location of object edges.`
- **`p90–97` (`L24_N3193_pos__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.934` `quad=Q3` `gate_open=False` | dist `act_dir=0.003` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` For example, if one identical twin acquires schizophrenia, the other stands just a 50 percent chance of also suffering from the disease. Indeed, abundant data suggest that psychiatric ailments typically result from a complex interplay between the environment and a number of different genes [see ¿The Character Code,¿<< by>> Turhan Canli; Scientific American Mind`
  - **DISTRACTOR:** ` and inconvenient, economic activity is impeded. ⏎ Digital financial services ⏎ In countries where digital payment systems have taken hold in poor and rural communities, we work with banks, insurance companies, and other providers to increase the range of financial services that people can access in digital form. Many of these services are designed<< to>> meet the specific household management needs of low`
- **`p75–90` (`L24_N3193_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.176` `quad=Q3` `gate_open=False` | dist `act_dir=0.002` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(B); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` digital signatures to identify what principal is responsible ⏎ for the program. This principal is mapped to a security policy. After ⏎ that, we have identified three different ways to enforce the policy: ⏎ - A number of traditional operating systems were based ⏎ on unforgeable pointers which could be safely given to user code<<. ⏎ >>Java provides a perfect environment for implementing capabilities`
  - **DISTRACTOR:** `din Farms in Viola, Wisconsin, does want to stir the pot. He’s a produce farmer participating in the pilot project who, along with his brother, decided to remake the family farm according to a vision for what he calls “the next evolution of food.” Although he has only a few employees<<,>> Hauke was excited to offer them`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 27/30: `L04_N0920_pos` (High-Floor / Judge-Disagreement Notable) **[PILOT_20]** **[LAYER-4 POSITIVE CONTROL]**
- **Layer / Neuron / Sign:** L4 / N920 / `pos` | **Nearest Same-Layer Neuron (Descriptive):** N2563 (gate cos = `0.11053336411714554`)
- **Top-3 trigger tokens:** `['sh', 'ant', 'dt']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=1.00, p90–97: Q3=0.99, p97–99.9: Q3=0.97, p99.9+: Q3=0.32
- **Own label:** Proper Noun and Place Name Suffixes: Activations cluster on tokens marking the ends of proper nouns and geographic or institutional names, especially those ending in -sh, -ant, -dt, -er, -und, -wig, -ah, -es, -anc, -as, -ass, -stead, -el, -ena, and similar suffixes, often following a capitalized stem. Occasional activations also occur on common verbs or adjectives in infinitive or past participle form, but these are less frequent.
- **Wrong label** (donor `L18_N0809_pos`, L18, sim=0.382338): Prepositions and Function Words in Contextual Phrases: Triggers include common prepositions (to, in, since, for, of, through), conjunctions (and), and other function words (has, end, body, planets, He) appearing within idiomatic, temporal, spatial, or attributive phrases, often linking clauses or specifying relationships between entities or events.
- **Token-only label:** `Activates on tokens such as: "sh", "ant", "dt"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.33 | 0.83 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.17 | 0.17 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.50 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `token_only` | 0.67 | 0.33 | 0.67 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.67 | 1.00 | 1.00 |
| `gpt-6.2-sol@low` (third judge) `own` | 0.58 | 0.67 | 0.92 | 1.00 |
| `rarity` | 0.67 | 0.67 | 0.83 | 1.00 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L04_N0920_pos__p99.9_plus__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.496` `quad=Q3` `gate_open=False` | dist `act_dir=-0.012` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=OK(A); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=OK(A); gpt-6.2-sol@low/wrong[rev]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** `" in the action area of your proposed project. Below is guidance to help you make that determination. ⏎ Habitat: The eastern prairie fringed orchid (orchid) occurs in a wide variety of habitats, from wet to mesic prairie or wetland communities, including, but not<< limited>> to sedge meadow, fen,`
  - **DISTRACTOR:** ` individuals at a moment in time. Their best outcome is total elimination of the condition in the shortest possible period. Our first goal is to cause no outbreaks, no epidemics, to manage, to contain the infection." Bringing the absolutist attitudes of medicine to a malarious village, Spielman says<<,>> "is a good way to do a`
- **`p97–99.9` (`L04_N0920_pos__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.134` `quad=Q3` `gate_open=False` | dist `act_dir=0.011` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=MISS(B); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` had failed terribly in preventing the propagation of yellow fever and other infectious diseases. Many authorities blamed this failure on its location only a few miles from the most densely populated parts of the city and thu, facilitating the interaction between infected patients and residents. Determined to prevent another calamity, The Board of Health<< constructed>> a new lazaretto ten miles south`
  - **DISTRACTOR:** ` that gave them an advantage over adversaries. In wartime, when a nation's survival is at stake, the reasons for secrecy are most apparent, the secrecy restrictions imposed by the government are most widespread,[*] and acceptance of those restrictions by the citizens is broadest.[†] In peacetime,<< there>> are fewer reasons for secrecy in government,`
- **`p90–97` (`L04_N0920_pos__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.070` `quad=Q3` `gate_open=False` | dist `act_dir=0.010` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); gpt-6.2-sol@low/no_label[fwd]=OK(B); gpt-6.2-sol@low/no_label[rev]=OK(A); gpt-6.2-sol@low/own[fwd]=OK(B); gpt-6.2-sol@low/own[rev]=OK(A); gpt-6.2-sol@low/token_only[fwd]=OK(B); gpt-6.2-sol@low/token_only[rev]=OK(A); gpt-6.2-sol@low/wrong[fwd]=MISS(A); gpt-6.2-sol@low/wrong[rev]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` or complications. ⏎ In addition to treating qi and blood stagnation, successful treatment of pain also requires careful differential diagnosis of pain. The three main diagnostic keys are the location of the discomfort, the type of pain and the cause of pain. Location refers to the exact part of the body that is affected:<< upper>> body, lower body, external musculos`
  - **DISTRACTOR:** ` the spread of health information that is not based on the treatments that this system has to offer. ⏎ Language may not only be negative; it can also be empty (Brooks, 2003); complex issues are broken down into broad statements with little meaning. In health care information, this pattern<< of>> communication places the cause of ill health on`
- **`p75–90` (`L04_N0920_pos__p75_90__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.027` `quad=Q3` `gate_open=False` | dist `act_dir=0.004` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=OK(A); gpt-6.1-terra@none/wrong[fwd]=MISS(B); gpt-6.2-sol@low/no_label[fwd]=OK(A); gpt-6.2-sol@low/no_label[rev]=OK(B); gpt-6.2-sol@low/own[fwd]=OK(A); gpt-6.2-sol@low/own[rev]=OK(B); gpt-6.2-sol@low/token_only[fwd]=OK(A); gpt-6.2-sol@low/token_only[rev]=OK(B); gpt-6.2-sol@low/wrong[fwd]=MISS(B); gpt-6.2-sol@low/wrong[rev]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** `H1N1pdm) virus strain unfolds into 2010, substantial policy challenges will continue to present themselves for the next 12 to 18 months. These are: measuring age-specific immunity to infection; accurately quantifying severity; improving treatment outcomes for severe cases; quant<<ifying>> the effectiveness of interventions; capturing the full`
  - **DISTRACTOR:** ` to balance the amount of glucose intake with the amount of insulin that needs to be injected. Not enough insulin and the glucose levels in the blood stream start to rise; too much insulin, and they plummet. ⏎ The consequences of hypoglycemia (hypo=low, glycemia=glucose<< in>> the blood) are easy to understand.`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 28/30: `L24_N3118_neg` (High-Floor / Judge-Disagreement Notable) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N3118 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N584 (gate cos = `0.11658117920160294`)
- **Top-3 trigger tokens:** `[' S', ' X', ' Z']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Mid-token Capital Letters and Scientific Abbreviations: Activations cluster on tokens with internal or initial uppercase letters, especially in scientific names, abbreviations, and proper nouns, including chemical symbols, acronyms, and names with mid-token capitalization or uncommon capitalization patterns.
- **Wrong label** (donor `L11_N0664_pos`, L11, sim=0.463728): Common Nouns, Adjectives, and Auxiliary Verbs: Activations cluster on frequent English nouns (e.g., months, countries, materials, foods, religions), adjectives describing physical or abstract qualities, and auxiliary or linking verbs such as 'was' and 'are', often in mid-sentence contexts. The pole also responds to plural possessive suffixes and some number tokens.
- **Token-only label:** `Activates on tokens such as: " S", " X", " Z"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.67 | 0.67 | 0.83 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.33 | 0.50 | 0.17 |
| `gpt-6.1-terra@none` `no_label` | 0.50 | 0.83 | 1.00 | 1.00 |
| `gpt-6.1-terra@none` `token_only` | 0.83 | 0.83 | 0.67 | 0.83 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.33 | 0.50 | 0.83 | 1.00 |
| `rarity` | 0.33 | 0.33 | 0.67 | 0.67 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N3118_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=29.750` `quad=Q2` `gate_open=True` | dist `act_dir=-0.008` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` Gellman, David N. "Pirates, Sugar, Debtors, and Slaves: Political Economy and the Case for Gradual Abolition in New York." Slavery & Abolition (2001) 22(2): 51-68.<< Iss>>n: 0144-`
  - **DISTRACTOR:** `-emission-free energy generation, then nuclear is in the lead, followed by hydropower and then wind and solar. But with growth rates of 30 percent or better, only wind and solar have the potential to have a global impact simultaneously on emissions and energy supply. Combine those technologies with a<< simultaneous>> development of a hydrogen economy, and we`
- **`p97–99.9` (`L24_N3118_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=3.406` `quad=Q2` `gate_open=True` | dist `act_dir=0.001` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=OK(A); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=OK(A); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` to power the fuel cells will be acquired from carbon-neutral sources, such as landfills and wastewater plants, and placed into the national natural gas pipeline system. The building’s exterior is predominately a glass curtainwall system incorporating highly efficient, insulated, double-paned glass with a clear, low-em<<iss>>ive coating. ⏎ Hines views its newest`
  - **DISTRACTOR:** ` women. It was also sometimes tradition that slaves could keep the money they earned for the sale of goods during the holiday. ⏎ While the holiday season was meant to be a joyous occasion, slaves that worked inside the house would be worked hardest during Christmas, as many owners and their families would host Christmas parties<<. ⏎ >>The Christmas holiday would also be a time`
- **`p90–97` (`L24_N3118_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.457` `quad=Q4` `gate_open=False` | dist `act_dir=-0.004` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` estimating dinosaur speed are comparing to recorded speeds of modern animals of similar body size and build, and measuring distances between fossil footprints in a trackway and using these distances to calculate estimated speed. Walking-speed estimates for medium-sized bipedal (two-legged) dinosaurs vary from 4 kph to <<6>> kph, and peak running-speed estimates`
  - **DISTRACTOR:** `olith nearby. Speleothems began growing shortly after the caverns ⏎ were formed, according to dating from stalactite and stalagmite growth rings. ⏎ Moaning Cavern's spectacular speleothems are other-worldly. Photo courtesy of Underground Adventures. ⏎ California Cavern first opened to the<< public>> after its discovery in 185`
- **`p75–90` (`L24_N3118_neg__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.055` `quad=Q4` `gate_open=False` | dist `act_dir=-0.002` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` contains a small amount of water for an estuary of its size. It also has a very large area of watershed for its volume of water, and thus is quite susceptible to contamination from activities in the watershed. Originally almost entirely forested, the watershed now is the site of coal mining, industry,<< agriculture>> and, increasingly, urbanization. A`
  - **DISTRACTOR:** ` intense exercise under normal conditions without harm to a developing baby. Another myth associated with exercise was the fear that active muscles demand increased blood flow, blood flow that could be taken away from or stolen from the baby and placenta. This also has never been proven under typical exercise conditions. Lastly, is the<< fear>> that the excess jostling that occurs`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 29/30: `L24_N4175_neg` (High-Floor / Judge-Disagreement Notable) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L24 / N4175 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N3235 (gate cos = `0.15809455513954163`)
- **Top-3 trigger tokens:** `['.', '1', ' to']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Proper Name Fragments and Abbreviations: Activations cluster on tokens forming parts of proper names, especially surname or given name fragments, initials, and abbreviations, as well as publisher or organization name segments and technical acronyms. Triggers include capitalized or period-adjacent substrings, often at name or title boundaries.
- **Wrong label** (donor `L18_N2188_neg`, L18, sim=0.551043): Function Words and Morphological Suffixes: Triggers include common function words such as auxiliary verbs, pronouns, conjunctions, and prepositions, as well as morphological suffixes like -ed, -ly, and -ine at word endings. Activations often occur on or immediately after these grammatical markers within diverse syntactic contexts.
- **Token-only label:** `Activates on tokens such as: ".", "1", " to"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.67 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `wrong` | 0.33 | 0.33 | 0.33 | 0.00 |
| `gpt-6.1-terra@none` `no_label` | 0.67 | 0.67 | 0.50 | 1.00 |
| `gpt-6.1-terra@none` `token_only` | 0.33 | 0.33 | 0.50 | 0.33 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.50 | 1.00 | 0.83 |
| `rarity` | 0.67 | 0.83 | 0.50 | 0.50 |
| `top_token` | 0.58 | 0.42 | 0.50 | 0.33 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L24_N4175_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=17.875` `quad=Q2` `gate_open=True` | dist `act_dir=-0.002` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=MISS(A); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.0; rule:top_token=0.0; rule:window_has_ge_p97=1.0
  - **FIRING:** ` guardian with respect to identification, evaluation, and/or placement. ⏎ IDEA - Compliance is monitored by the State's Bureau of Education for Exceptional Students Division for Public School in that particular State Department of Education. ⏎ Complaints resolved by that same Bureau. ⏎ 504 - Enforced by the US Office<< of>> Civil Rights ⏎ Complaints resolved by`
  - **DISTRACTOR:** ` ECL only applies to external users and devices. In other words, if you have an ECL, you still have to get a CAL for all employees and contractors. ⏎ Even though Terminal Services (TS) is built into Windows Server 2003 and 2008, you need<< to>> get a separate TS CAL for each client`
- **`p97–99.9` (`L24_N4175_neg__p97_99.9__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.828` `quad=Q4` `gate_open=False` | dist `act_dir=-0.001` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=OK(A); gpt-6-luna@none/token_only[fwd]=MISS(B); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=MISS(B); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=OK(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` or having a new sexual partner or multiple sexual partners can put you at greater risk for bacterial vaginosis. While it's more common in women who are sexually active, it also occurs in women who are not sexually active. It's not usually necessary for your sex partner to be treated. ⏎ Your doctor<< will>> examine your vagina and use a cotton sw`
  - **DISTRACTOR:** ` generally received benefits, such as transportation and food, in addition to wages. ⏎ The standard legal workweek is 40 hours but can be extended to 48 hours. After 48 hours, overtime must be paid at 50 percent over the base hourly salary. Overtime is limited<< by>> law to two hours per day and `
- **`p90–97` (`L24_N4175_neg__p90_97__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.111` `quad=Q4` `gate_open=False` | dist `act_dir=0.001` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` the density of intersections and how direct paths are between places. Increased connectivity reduces the amount of circuitous travel required and often encourages shorter vehicle trips and the use of alternative modes such as biking and walking. ⏎ A simplistic measure of connectivity is the connectivity ratio which is calculated by: ⏎ A local circulation map is a<< method>> to visually examine connectivity. Streets providing local`
  - **DISTRACTOR:** `istry of Education, students are expected to meet the costs of ⏎ If a child has to travel a long distance to school, and there is ⏎ no public transport or school bus service, financial assistance ⏎ can be provided. Financial assistance and/or bus and taxi ⏎ services are provided for special education students. ⏎ <<If>> you plan to change schools, the principal`
- **`p75–90` (`L24_N4175_neg__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.035` `quad=Q4` `gate_open=False` | dist `act_dir=-0.003` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` of travelling from Europe to Australia in 2-4h. ⏎ Oxfordshire-based Reaction Engine says its A2 (a Flight artist's impression of which is pictured above), meets take-off noise restrictions and is powered by hydrogen-fuelled turbine-based combine cycle (TBCC) engines. ⏎ The other<< two>> concepts are an EADS Mach 8`
  - **DISTRACTOR:** ` higher than normal. Notify water officials of low or no water pressure. ⏎ - Stay Cool. During hot days, stay cool indoors and drink plenty of fluids. ⏎ - Check on Neighbors. Check on elderly or medically dependent neighbors. ⏎ - Life Support Equipment. If someone in your household uses life support equipment,<< make>> arrangements for a back-up power supply. ⏎ `

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._

---
## Pole 30/30: `L11_N4693_neg` (High-Floor / Judge-Disagreement Notable) **[HELDOUT_80]**
- **Layer / Neuron / Sign:** L11 / N4693 / `neg` | **Nearest Same-Layer Neuron (Descriptive):** N684 (gate cos = `0.2037220001220703`)
- **Top-3 trigger tokens:** `[' cement', 'coma', 'in']`
- **Corpus Q3 (double-negative) shares:** p75–90: Q3=0.00, p90–97: Q3=0.00, p97–99.9: Q3=0.00, p99.9+: Q3=0.00
- **Own label:** Scientific and Technical Nouns, Especially Morphological Variants: Activations cluster on scientific and technical nouns, often with suffixes like -ite, -in, -gen, -id, -ate, -its, and -ities, as well as domain-specific terms from geology, biology, chemistry, and astronomy. The trigger is strongest for tokens marking the end of such nouns or their plural or adjectival forms.
- **Wrong label** (donor `L24_N4212_neg`, L24, sim=0.51314): Numbers, Ordinals, and Large Quantities: Activations are triggered by numeric tokens (single- and multi-digit), ordinal forms, and words denoting large or indefinite quantities, as well as related suffixes and morphemes within number words or names containing number-like fragments.
- **Token-only label:** `Activates on tokens such as: " cement", "coma", "in"`

| Method | `p75–90` | `p90–97` | `p97–99.9` | `p99.9+` |
|---|---:|---:|---:|---:|
| `gpt-6.1-terra@none` `own` | 0.50 | 0.50 | 0.33 | 0.83 |
| `gpt-6.1-terra@none` `wrong` | 0.50 | 0.33 | 0.17 | 0.67 |
| `gpt-6.1-terra@none` `no_label` | 0.83 | 0.83 | 0.33 | 0.67 |
| `gpt-6.1-terra@none` `token_only` | 0.50 | 0.17 | 0.33 | 1.00 |
| `gpt-6-luna@none` (secondary judge) `own` | 0.67 | 0.67 | 0.33 | 0.83 |
| `rarity` | 0.50 | 0.67 | 0.50 | 1.00 |
| `top_token` | 0.50 | 0.50 | 0.50 | 0.50 |
| `window_has_ge_p97` | 0.50 | 0.50 | 1.00 | 1.00 |

### Representative Marked Snippets (`dead_zone` Pairs)
- **`p99.9+` (`L11_N4693_neg__p99.9_plus__dead_zone__0`, `pos_slot=B`):** pos `act_dir=1.719` `quad=Q2` `gate_open=True` | dist `act_dir=0.002` `quad=Q4`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=OK(B); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=1.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` oil. ⏎ Every hour, rainforest land equivalent to 300 football fields in size is ⏎ destroyed, burnt and replaced with palm oil plantations. To put it into ⏎ perspective - in Sumatra alone there is now more than 4 times as much land ⏎ cultivated with oil<< palms>> as there is orangutan habitat remaining. ⏎ `
  - **DISTRACTOR:** ` and it was to be pitched within and without with pitch, or bitumen, to make it water-tight. ⏎ Its proportions were to be 300 cubits long, 50 cubits broad, and 30 cubits high. If the cubit be taken at 1<<8>> inches, its length would have been `
- **`p97–99.9` (`L11_N4693_neg__p97_99.9__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.469` `quad=Q2` `gate_open=True` | dist `act_dir=0.006` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=MISS(A); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=MISS(A); gpt-6.1-terra@none/no_label[fwd]=MISS(A); gpt-6.1-terra@none/own[fwd]=MISS(A); gpt-6.1-terra@none/token_only[fwd]=MISS(A); gpt-6.1-terra@none/wrong[fwd]=MISS(A); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=1.0
  - **FIRING:** ` it as an American first strike. Both sides had practiced war games where a single high-altitude nuclear explosion from a submarine would be used to blind radar and satellites from the real attack, while the electromagnetic pulse would paralyze their defenses. ⏎ The Russian military went to full alert. Their ballistic missile submarines in<< the>> Arctic were all ordered to prepare for immediate`
  - **DISTRACTOR:** ` B.C. Persia later used cryptography for political purposes. Cryptography began its steady development in western civilization starting about the 13th century, primarily in ⏎ . By the early 16th century, Italy 's ruling Council of Ten had an elaborate organization for enciphering and decipher<<ing>> messages. Venice ⏎ Restrictions on crypt`
- **`p90–97` (`L11_N4693_neg__p90_97__dead_zone__0`, `pos_slot=A`):** pos `act_dir=0.219` `quad=Q4` `gate_open=False` | dist `act_dir=0.000` `quad=Q2`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(A); gpt-6-luna@none/own[fwd]=MISS(B); gpt-6-luna@none/token_only[fwd]=OK(A); gpt-6-luna@none/wrong[fwd]=MISS(B); gpt-6.1-terra@none/no_label[fwd]=OK(A); gpt-6.1-terra@none/own[fwd]=MISS(B); gpt-6.1-terra@none/token_only[fwd]=MISS(B); gpt-6.1-terra@none/wrong[fwd]=MISS(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` represented by Jamblichus there are present elements that are duplicable today in the most primitive regions. ⏎ Several doors open here to alluring bypaths--to inspiration, prediction, oracles, on the one side, these presuming a favoring: disposition on the part of the ghost; and<<,>> on the other, to necromancy`
  - **DISTRACTOR:** `19-College of Fine Arts organized, which consisted of offerings in music and in expression. ⏎ 1921-Terminated in favor of the present School of Music. ⏎ 1919-School of Education organized. ⏎ 1920-School of Nursing organized as a diploma-granting<< program>>. ⏎ 1921-School of`
- **`p75–90` (`L11_N4693_neg__p75_90__dead_zone__0`, `pos_slot=B`):** pos `act_dir=0.148` `quad=Q4` `gate_open=False` | dist `act_dir=-0.003` `quad=Q3`
  - *Verdicts:* gpt-6-luna@none/no_label[fwd]=OK(B); gpt-6-luna@none/own[fwd]=MISS(A); gpt-6-luna@none/token_only[fwd]=OK(B); gpt-6-luna@none/wrong[fwd]=OK(B); gpt-6.1-terra@none/no_label[fwd]=OK(B); gpt-6.1-terra@none/own[fwd]=OK(B); gpt-6.1-terra@none/token_only[fwd]=OK(B); gpt-6.1-terra@none/wrong[fwd]=OK(B); rule:rarity=0.0; rule:top_10_token=0.5; rule:top_token=0.5; rule:window_has_ge_p97=0.5
  - **FIRING:** ` follows:- ⏎ 1. Too much exposure to dust, sunlight, water and other pollutants. ⏎ 2. Too much of sweating. ⏎ 3. Irregular sleeping pattern. ⏎ 4. Anxiety, depression, insomnia. ⏎ 5. Unhygienic way of living ⏎ 7. Alcohol consumption. ⏎ Hair Loss Remedies ⏎ <<1>>. Liberal intake of vitamins. ⏎ 2.`
  - **DISTRACTOR:** `st Month ⏎ With temperatures reaching 45 degrees Celsius up north, Facebook updates have begun quoting the opening line of T.S.Eliot's The Waste Land: April is the cruellest month. ⏎ In the poem, the line is meant ironically. It is spoken from the point of view of those<< who>> feel threatened by the awakening of spring,`

### Manual Reading Notes (fill during Sunday audit)
- [ ] Top-band (`p99.9+`) label accurately matches snippets?
- [ ] Label still describes weak bands (`p90–97`, `p75–90`)?
- [ ] Apparent shortcut (token identity, rarity, formatting) vs. semantic concept?
- **Free notes:** _..._
