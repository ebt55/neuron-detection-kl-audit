# Manual Reading Sheet: 12 Selected Qwen3-4B Neurons

Twelve deterministically selected `Qwen/Qwen3-4B` neurons from `Ulto85/neurons-kinda-clean` (MIT): 4 with `top100 == 1.000` and high `sim_cal_rec` (including `info`, `k=26`), 4 with `top100 == 1.000` and negative `sim_cal_rec` (including `PCR`, `k=23`, and the minimum `-7.110`), and 4 failing top-100 detection (`top100 < 0.800`) spread across layers and causal outcomes. Snippets come from `HuggingFaceFW/fineweb` (`sample/10BT/000_00000.parquet`, ODC-By). KL shares are loaded from `results/qwen4b/band_kl_4b_all278.json` (`n = 278` neurons).

## Overview Table

| # | Group | `k` | Layer / Neuron | Polarity | `top100` | `p99.9+` | `p97–99` | `sim_cal_rec` | `active` | `strong20` | KL `p99.9+` / `p97–99.9` / `p75–97` / `<p75` |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | A (high) | `26` (info) | `L00 N9209` | unipolar | 1.000 | 0.650 | 0.800 | +0.859 | +0.958 | +0.883 | 0.148 / 0.307 / 0.205 / 0.340 |
| 2 | A (high) | `276` | `L35 N7008` | unipolar | 1.000 | 0.850 | 0.900 | +0.909 | +0.923 | +0.650 | 0.761 / 0.002 / 0.001 / 0.237 |
| 3 | A (high) | `59` | `L08 N2134` | unipolar | 1.000 | 0.900 | 0.500 | +0.584 | +0.709 | +0.759 | 0.000 / 0.161 / 0.262 / 0.578 |
| 4 | A (high) | `225` | `L32 N146` | unipolar | 1.000 | 1.000 | 1.000 | +0.429 | +0.852 | +0.318 | 0.050 / 0.854 / 0.011 / 0.084 |
| 5 | B (neg) | `23` (PCR) | `L00 N8146` | unipolar | 1.000 | 1.000 | 0.600 | -0.570 | +0.919 | +0.144 | 0.777 / 0.126 / 0.023 / 0.074 |
| 6 | B (neg) | `17` | `L00 N7325` | unipolar | 1.000 | 0.950 | 0.700 | -7.110 | +0.222 | +0.402 | 0.064 / 0.161 / 0.024 / 0.751 |
| 7 | B (neg) | `31` | `L04 N1558` | unipolar | 1.000 | 0.950 | 0.600 | -1.104 | -0.147 | +0.320 | 0.234 / 0.153 / 0.079 / 0.534 |
| 8 | B (neg) | `93` | `L12 N3077` | unipolar | 1.000 | 0.950 | 0.600 | -1.067 | +0.522 | +0.245 | 0.092 / 0.268 / 0.184 / 0.456 |
| 9 | C (fail) | `32` | `L04 N2686` | bipolar | 0.450 | 0.600 | 0.750 | +0.008 | +0.040 | +0.525 | 0.203 / 0.457 / 0.279 / 0.061 |
| 10 | C (fail) | `82` | `L08 N8930` | bipolar | 0.775 | 0.600 | 0.550 | +0.029 | +0.552 | +0.203 | 0.318 / 0.368 / 0.292 / 0.021 |
| 11 | C (fail) | `265` | `L35 N3526` | bipolar | 0.550 | 0.525 | 0.450 | -0.723 | -0.005 | +0.076 | 0.519 / 0.262 / 0.195 / 0.024 |
| 12 | C (fail) | `119` | `L16 N2492` | unipolar | 0.500 | 0.250 | 0.400 | -0.015 | -0.012 | +0.000 | 0.000 / 0.133 / 0.250 / 0.617 |

## Group A: Perfect Top-100 Detection (`1.000`) & High Causal Recovery

### 1. Neuron `k=26` (`L00_N9209` — `info`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L00_N9209_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.650`, `p97–99 = 0.800` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.859`, `sim_cal_rec_active = +0.958`, `strong20_rec = +0.883` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.148`, `p97–99.9 = 0.307`, `p75–97 = 0.205`, `< p75 = 0.340` |

**Verbatim Ulto85 Pole Label(s):**
- `L00_N9209_pos`: The token "info"/"Info" (with or without leading space, also "-info") used as a standalone word meaning information: "more info", "any info", "Background info", "study.info", "Info Box", "Info Wars". Weaker firing (~3) on the subword "inf" when it starts a longer word (inf-requent, inf-idel, Z-inf-andel, inf-raction, inf-estation, inf-using, inf-illing), and ~5 on "Infos" (Infosys). Faint (~1) on related abbreviation-like stems such as "informative", "inform", "Prof", "intro", "summary".

**Top-100 Positive Snippets (5 sampled):**
1. `L00_N9209_pos` (`pair_idx=0`) | `act_dir=0.318` (`7.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Fiction & Fantasy ⏎ Large Format Landscape, 13×11 in, 33×28 cm ⏎ - Publish Date Feb 14, 2010 ⏎ - Tags steve thorpe, nemesis 109, science fiction and fantasy illustration, beyond,<< infinity>>, steve, thorpe, nem`
2. `L00_N9209_pos` (`pair_idx=5`) | `act_dir=2.906` (`68.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` had one of these on your stand last Sunday, didnt you? I think I remember looking at it closely, and deciding whether to buy it or not. Looks v. unusual. Tell me more about it ⏎ Thanks Liga. Jacinta you would be better wait until Deb is on to get the accurate<< info>>. :-) ⏎ You really have some rare`
3. `L00_N9209_pos` (`pair_idx=10`) | `act_dir=0.336` (`8.0%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` function is provided mainly for debugging and development tuning, since it may be useful to monitor lwIP resource usage over various test scenarios when tuning an eCos lwIP configuration. ⏎ The function is a NOP unless the lwIP statistics gathering support is ⏎ configured. The main configuration ⏎ CYGDBG_LWIP<<_STATS>> controls the ⏎ availability of the statistics feature`
4. `L00_N9209_pos` (`pair_idx=14`) | `act_dir=0.344` (`8.1%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` 1/10 gram amounts to people in Nome,” the document reads. ⏎ According to a trooper’s dispatch, the investigation began on March 1 when Nome WAANT met with a person who provided information that identified Bourdon drug trafficker. WAANT signed the person up to become a confidential<< informant>>. On five different occasions, the informant`
5. `L00_N9209_pos` (`pair_idx=19`) | `act_dir=3.562` (`84.4%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `burg, West Virginia Symphony Chorus and West Virginia State University Chorus. ⏎ WANT TO GO?"Home for the Holidays"West Virginia Symphony OrchestraWHERE: ⏎ 8 p.m. Friday and SaturdayTICKETS: ⏎ Adults $12 to $64, students $10 to $52<<INFO>>: ⏎ 304-561`

**`p97–99` Positive Snippets (5 sampled):**
1. `L00_N9209_pos` (`pair_idx=0`) | `act_dir=0.035` (`0.8%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` Elementary and Wanamaker Elementary. ⏎ The openings are for the 2013-14 school year. ⏎ There is a one-time $25 tuition payment for the program, which is otherwise free. Applications are available at each of the schools and at the district’s administrative center. ⏎ In order<< to>> join the program, children must turn `
2. `L00_N9209_pos` (`pair_idx=2`) | `act_dir=0.034` (`0.8%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` (MT4/MT5).Start learning to trade Forex smart with the best Forex trading analysis software. with applied indicators,. Within our Forex trading analysis software,.Boss indicator,binary, Option, Plus, Market, Profile, Steidlmayer, Volume, Strips,Greg Shrader. ⏎ Indicator<<-ind>>icator teknical untuk trading Forex di AGE`
3. `L00_N9209_pos` (`pair_idx=4`) | `act_dir=0.042` (`1.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` File ⏎ - Prints Laser Checks or Vouchers ⏎ - Review Running Reserve Balances for Each Claim ⏎ - Integrated Document Storage and Retrieval ⏎ - Fully Integrated with Microsoft Office E-Mail ⏎ - All Reports in Adobe Acrobat format (.PDF) ⏎ - User Defined Form Letters ⏎ - All Form Letters use Microsoft Word<< for>> easy claimant specific changes. ⏎ - Create`
4. `L00_N9209_pos` (`pair_idx=7`) | `act_dir=0.043` (`1.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` sent it spiraling downward. ⏎ In this upcoming earnings report, investors are expecting Tempur-Pedic’s earnings per share to be $0.55, down from the $0.84 they were expecting during the same quarter last year. The stock has been hammered in the past four quarters as<< investors>> have lowered their expectations for earnings, revenue`
5. `L00_N9209_pos` (`pair_idx=9`) | `act_dir=0.037` (`0.9%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `. ⏎ Expressing hope of a normal monsoon this year, Singh said: “The (Southwest) monsoon has already arrived in Andaman and Nicobar islands and it is expected to come 3-4 days early in Kerala.” He also noted the growth rate in agriculture and allied sector was<< about>> 4.4 per cent. The`

### 2. Neuron `k=276` (`L35_N7008`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L35_N7008_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.850`, `p97–99 = 0.900` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.909`, `sim_cal_rec_active = +0.923`, `strong20_rec = +0.650` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.761`, `p97–99.9 = 0.002`, `p75–97 = 0.001`, `< p75 = 0.237` |

**Verbatim Ulto85 Pole Label(s):**
- `L35_N7008_pos`: The standalone subword token ' un' (lowercase, with a leading space) when it begins a word that the tokenizer splits as ' un' + stem. Examples: un|reported, un|cles, un|secured, un|-usable, un|asked, un|controlled, un|hide, un|wise, un|forgeable, un|traceable, un|movable, un|obtainable, un|plug, un|attended, un|professional. It fires whether ' un' is the negating prefix or just the first piece of a word like 'uncles' or 'unquote'. It is strongest in running mid-sentence prose. The activation sits entirely on the ' un' token. The following stem token is ~0. Similar word-initial split prefixes get weaker activation: ' er'(adicating/asing) ~3, ' se'(aring) ~3, ' ca'(utions) ~4, ' ir'(ritable) ~2. Fused prefixes like ' uns'(uitable), ' unt'(itled), ' ut', ' lib', ' so' give ~1.

**Top-100 Positive Snippets (5 sampled):**
1. `L35_N7008_pos` (`pair_idx=0`) | `act_dir=191.000` (`71.3%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `400 Kg at speeds up to 500 degrees per second. Programming is made easy using "teach and Learn" techniques, removing the need for software expertise. ⏎ The XRC series robots have been used succesfully in a number of applications including palletising, assembly, machine tool loading and<< un>>loading, arc welding and packaging. ⏎ The`
2. `L35_N7008_pos` (`pair_idx=5`) | `act_dir=190.000` (`70.9%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` my last miscarriage. ⏎ It was at a really stressful time in my life. ⏎ He was extremely wanted. ⏎ When he left me I was just crushed. Why could my body not hold him? I was healthy! I did everything right! I DID EVERYTHING RIGHT!!! Why? Dr’s said that the<< un>>needed, ridiculous, crazy person stress I`
3. `L35_N7008_pos` (`pair_idx=10`) | `act_dir=165.000` (`61.6%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` has been born. It may non populate long. ⏎ Although Bellow takes us no further than the birth. Marcus Klein [ in The Kenyon Review. Spring 1962 ] has pointed out that At the minute of decease. his gesture is toward being. the verve that defines and<< un>>ites everyone. and his crying is an`
4. `L35_N7008_pos` (`pair_idx=14`) | `act_dir=91.500` (`34.1%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` various bladed tools on the sides of his van, as snapped by Flickr member Andrew Codrington (acodring) in 2008 and Twitter member Laura Dudas in 2015. The van’s sun-faded sides pay homage to 7+ years (and<< un>>counted miles) of slow-motion service`
5. `L35_N7008_pos` (`pair_idx=19`) | `act_dir=119.000` (`44.4%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `With the sea breathing down their necks, New Zealanders are drawn more than most to salty books. One theory has it that all of us retain memories of a long migratory voyage out to these shores. ⏎ Too many passages ended in calamity, with thousands of colonial sailing ships flung onto an<< un>>charted coastline. The dramatic and gruesome wre`

**`p97–99` Positive Snippets (5 sampled):**
1. `L35_N7008_pos` (`pair_idx=0`) | `act_dir=0.373` (`0.1%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Punjab due to caste. ⏎ Kolhapur: Islampur court has sentenced a man. ⏎ ten witnesses and a DNA report of his daughter and her baby. The girl had delivered a child, the court had observed. According to the prosecution. ⏎ Kolhapur: Twenty-six year old Ganesh<< N>>mdeo Kumbhar was found murdered`
2. `L35_N7008_pos` (`pair_idx=2`) | `act_dir=0.371` (`0.1%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` is sweet but not cloyingly so and is great with or without nuts then look no further. ⏎ This is a recipe I have tweaked and played with over time to make the perfect brownie as the base for a hot fudge brownie sundae! Some of my other recipes, while great on<< their>> own, were too sweet when paired with`
3. `L35_N7008_pos` (`pair_idx=4`) | `act_dir=0.535` (`0.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `-Kashmir University of Agriculture Sciences and Technology (SKUAST), told The Kashmir Monitor. ⏎ Noori is among 25 Pashmina goats being reared by the scientists at the farm of Sher-e-Kashmir University of Agricultural Sciences. “There are 25 goats on our farm.<< No>>ori and her four offsprings are`
4. `L35_N7008_pos` (`pair_idx=7`) | `act_dir=0.410` (`0.2%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` to attract bidders, so it is important that you are aware of the true market value of the property, so that you don’t get carried away in bidding. ⏎ Another factor to consider is the demographic in the area surrounding the property you are researching. Use a website like www.checkmyarea.com or<< www>>.upmystreet.co.uk. This`
5. `L35_N7008_pos` (`pair_idx=9`) | `act_dir=0.361` (`0.1%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` has my highest recommendation. The objects, images, texts, and media used to portray the stories of francophone north america are engaging, and the topics are thought provoking. ⏎ It is also in a great location, close to a host other things that will grab the attention of a visitor to this won<<def>>ul city. ⏎ Visited on a Tuesday when`

### 3. Neuron `k=59` (`L08_N2134`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L08_N2134_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.900`, `p97–99 = 0.500` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.584`, `sim_cal_rec_active = +0.709`, `strong20_rec = +0.759` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.000`, `p97–99.9 = 0.161`, `p75–97 = 0.262`, `< p75 = 0.578` |

**Verbatim Ulto85 Pole Label(s):**
- `L08_N2134_pos`: Words expressing avoiding, evading, or getting around something, such as avoid/avoiding/avoidable, steer clear of, dodged/dodging, sidestep, and navigate around. The activation sits on the verb, or on the completing particle of a phrasal verb (clear, of, around, the -ep/-pping piece of sidestep). It often carries over to the next one or two tokens that begin the avoided object: a, what, several, a comma, because. Weaker cases are related motion-around or escape senses, such as 'fled from', 'runs around a', 'drove it around a bit', and 'prevent this'.

**Top-100 Positive Snippets (5 sampled):**
1. `L08_N2134_pos` (`pair_idx=0`) | `act_dir=5.938` (`31.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` salespeople staffing your booth, because they don’t have to work so hard to pull in attendees. Remember, even though your salespeople may be excellent on the phone or in meetings may not be good at getting people out of the aisle and into the booth. Your salespeople will be so grateful to<< avoid>> having to hook all the attendees in. ⏎ `
2. `L08_N2134_pos` (`pair_idx=5`) | `act_dir=6.938` (`36.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` them Japanese and therefore impose an embargo on Naha port. ⏎ In order for Satsuma to continue trade through the port of Naha in Okinawa, Satsuma needed to disguise its actions. Lord Shimazu therefore ordered Okinawa to hide their true relationship with Japan to outsiders, so as to<< avoid>> reprisal from Japan's Shog`
3. `L08_N2134_pos` (`pair_idx=10`) | `act_dir=9.125` (`48.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` a corridor traveled by 200,000 vehicles a day. ⏎ The $173 million project to improve the I-390/I-490/Route 31 interchange near Rochester is designed to allow “southbound through traffic to travel above the busy interchange and<< avoid>> vehicles merging and exiting the highway,” says`
4. `L08_N2134_pos` (`pair_idx=14`) | `act_dir=10.375` (`54.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` A chase ensued, followed by a crash and a physical assault. The dashboard cameras showed officers beating Collins, however, the views from one camera were obscured because Officer Deutsch turned the lens away. The beating left Collins with bruises and he developed blood clots. Collins was charged with three traffic warrants, ev<<ading>> arrest, and possession of marijuana. D`
5. `L08_N2134_pos` (`pair_idx=19`) | `act_dir=5.094` (`26.8%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` meals, or 12 ounces total, of fish or shellfish a week. They should limit high-mercury fish to one serving per week. ⏎ To be safest, however, The Green Guide and the Environmental Working Group recommend limiting moderate-mercury fish to one meal a month, and bypass<<ing>> high-mercury fish completely. In`

**`p97–99` Positive Snippets (5 sampled):**
1. `L08_N2134_pos` (`pair_idx=0`) | `act_dir=0.246` (`1.3%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` selling donuts, ⏎ MAYBE DON'T GET YOUR WEDDING CAKE THERE. ⏎ (King cakes should be fine, though.) ⏎ And finally, Elydia B. writes: ⏎ "My wedding was like a fairy tale. The cake was supposed to be from a very prestigious bakery in Tusc<<any>>..." ⏎ Wait, Tuscany? Like`
2. `L08_N2134_pos` (`pair_idx=2`) | `act_dir=0.216` (`1.1%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` sixty mile ride that I didn't even know existed. The horseshoe curves, switchbacks, and elevation changes were the most extreme I had ridden in all of the previous 7,000 Alaskapade miles. Fortunately, these roads were all paved and in great shape with no<< gravel>> sections or pot holes. When I saw`
3. `L08_N2134_pos` (`pair_idx=4`) | `act_dir=0.216` (`1.1%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` has said it would act after Moscow. ⏎ "Too much testosterone and not enough good sense," is how Rep. Mark Kirk, R-Ill., a naval intelligence reservist who flew on surveillance missions over Serbia, sums up the problem. ⏎ Washington and Moscow realized long ago something had to be done to keep<< encounters>> from getting out of hand. Now the`
4. `L08_N2134_pos` (`pair_idx=7`) | `act_dir=0.224` (`1.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `=China X-Large:Length:23, Our wide selection is elegible for free shipping and free returns. You should avoid shift knobs with set screws or plastic parts as they tend to fail or wobble. Buy Genuine Hyundai 89200-3J000-V<<9>>D Seat Cushion Assembly. 2`
5. `L08_N2134_pos` (`pair_idx=9`) | `act_dir=0.221` (`1.2%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` (pace yourselves!), and in Europe organized outings range from sampling the snails of France to seafood and olives in Greece. For people traveling in countries in which they don’t speak or read the language, this can be a good introduction to menus — plus proper mealtime etiquette in that locale. ⏎ Add<< some>> fun research to your trip planning by reading`

### 4. Neuron `k=225` (`L32_N146`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L32_N0146_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 1.000`, `p97–99 = 1.000` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.429`, `sim_cal_rec_active = +0.852`, `strong20_rec = +0.318` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.050`, `p97–99.9 = 0.854`, `p75–97 = 0.011`, `< p75 = 0.084` |

**Verbatim Ulto85 Pole Label(s):**
- `L32_N0146_pos`: The comma token that comes directly after the leading digit(s) of a number and is followed by more digits, as in 4,500 / $12,000 / 3,000 MW. It also fires on a European decimal comma (0,67% / € 5,08 / 3,5 µm). The activation sits on the comma itself. It is strongest when only 1–2 digits come before the comma (4, 5, 2, 3, 15, 12). When a number has several groups, later commas are weaker (5,201,3 → +9 then +3). The digit right after the comma sometimes gets about +1. Ordinary list or clause commas get weak activation (+1 to +4), most often in enumerations of nouns or names (Oldsmobile, / durable goods, / one, if not both / March 23, 201x). A stray number-like comma such as 'great, great grandson' can reach +8.

**Top-100 Positive Snippets (5 sampled):**
1. `L32_N0146_pos` (`pair_idx=0`) | `act_dir=35.000` (`68.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` Forbes for the service hat it provides as one of the top four banks as ranked in six states. It is one of two to achieve this status, along with Capital One. Huntington Banks serves customers in the states of Indiana, Michigan and Wisconsin from which it is highly rated, along with almost 1<<,>>000 branches distributed between a total`
2. `L32_N0146_pos` (`pair_idx=5`) | `act_dir=51.000` (`100.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `els (dB). It is equal to 20 x the Log10 of the ratio of the Root Mean Square (RMS) of sound pressure to the reference of sound pressure (the reference sound pressure in air is 2 x 10-5 N/m2, or 0<<,>>00002 Pa). Or`
3. `L32_N0146_pos` (`pair_idx=10`) | `act_dir=32.250` (`63.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `9. ⏎ When we left two years ago we never did a bike trip with camping, our camping experience was limited to some guided trekking in Irian Jaja and a once and never again camping trip in France. Our longest bike trip was 3 weeks on the bike, never longer than 1<<,>>000 km and with the luxury`
4. `L32_N0146_pos` (`pair_idx=14`) | `act_dir=33.500` (`65.7%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` according to media reports, even though expats make up 20 per cent of the total voting population. ⏎ In the provinces of Naama, Tamenraset and Adrar, where the government allocates mobile polling stations for nomads, voter turnout was announced on Wednesday night at 34<<,>>000, 25,`
5. `L32_N0146_pos` (`pair_idx=19`) | `act_dir=39.250` (`77.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` and ever since their fates have been bound together for better or for worse. ⏎ The Niña is a replica of the ship on which Columbus sailed across the Atlantic on his three voyages of discovery to the new world beginning in 1492. Columbus sailed the tiny ship over 25<<,>>000 miles. That ship was`

**`p97–99` Positive Snippets (5 sampled):**
1. `L32_N0146_pos` (`pair_idx=0`) | `act_dir=7.781` (`15.3%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` ⏎ The only time keeping device which an applicant is permitted to use at the Bar examination is a non-digital (analog) wristwatch with no audible alarm. The Board prohibits digital watches, stop watches, clocks, and all digital devices. Do not bring personal digital assistants, cell phones, cameras<<,>> imaging devices, or any other electronic device`
2. `L32_N0146_pos` (`pair_idx=2`) | `act_dir=6.312` (`12.4%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` illustration. ⏎ Features & Details ⏎ - Category Science Fiction & Fantasy ⏎ Large Format Landscape, 13×11 in, 33×28 cm ⏎ - Publish Date Feb 14, 2010 ⏎ - Tags steve thorpe, nemesis 109<<,>> science fiction and fantasy illustration, beyond,`
3. `L32_N0146_pos` (`pair_idx=4`) | `act_dir=2.984` (`5.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `-time Polymerase Chain Reaction (RT-PCR) and serological assays. The genome sequencing of SARS-CoV-2 is also helping uncover the epidemiology of this coronavirus. Thus, here, genomic epidemiology is finding its application in surveillance - unraveling the mystery of the origin of the virus<<,>> frequency and types of mutations in the viral`
4. `L32_N0146_pos` (`pair_idx=7`) | `act_dir=5.625` (`11.0%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` Silver Stars, who have already qualified for a playoff berth. ⏎ “It means a lot to me [to represent St Vincent],” Young tells Caribbean Journal. “I think there’s a lot of girls and boys who are thinking about playing sports, and for us to be doing this and to be an example<<,>> and to show them this is possible,`
5. `L32_N0146_pos` (`pair_idx=9`) | `act_dir=5.781` (`11.3%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` time you choose to choose a name loan all of the way till the extremely end when you effectively pay back your final part of interest. These are generally highly-trained professionals and then help you produce a far more decision that is informed getting that loan against your vehicle. ⏎ Criteria Of Eligibility ⏎ Love said before<<,>> credit score is nowhere near our world of`

## Group B: Perfect Top-100 Detection (`1.000`) & Negative Causal Recovery

### 5. Neuron `k=23` (`L00_N8146` — `PCR`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L00_N8146_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 1.000`, `p97–99 = 0.600` |
| Ulto85 Causal Scores | `sim_cal_rec = -0.570`, `sim_cal_rec_active = +0.919`, `strong20_rec = +0.144` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.777`, `p97–99.9 = 0.126`, `p75–97 = 0.023`, `< p75 = 0.074` |

**Verbatim Ulto85 Pole Label(s):**
- `L00_N8146_pos`: Single-token all-caps initialisms/acronyms, especially those beginning with P or C (PCR, PCB, PCA, PC/PCs, PVC, CPR, CDC, CQC, QC, GC, CD) plus DNA/mRNA. Activation sits on the acronym token itself, mostly when it is used as a noun or modifier in running text. PCR in a scientific methods context is strongest. A trailing piece such as 's' or 'R' after the main token gets a small spillover.

**Top-100 Positive Snippets (5 sampled):**
1. `L00_N8146_pos` (`pair_idx=0`) | `act_dir=1.164` (`35.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` of using a wired internet connection over WiFi. Wired connections have no barriers and can never be interrupted as long as your PC is tethered to your router with a physical cable. ⏎ Unfortunately, mobile devices and tablets cannot take advantage of this feature. But that shouldn’t be an issue in a workplace environment where<< PCs>> are the main workstations. ⏎ Most routers`
2. `L00_N8146_pos` (`pair_idx=5`) | `act_dir=0.625` (`19.1%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` attacks worldwide causing losing their personal information, credit card numbers, bank account passwords, and so on. ⏎ So, how to prevent from being traceable and protect your activities? ⏎ Well, the idea is simple and easy to follow. Use VPN Software, but the question is how to choose the best vpn for<< PC>>, Windows, Android and Mac devices?`
3. `L00_N8146_pos` (`pair_idx=10`) | `act_dir=0.668` (`20.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Revisions ⏎ - Related Practices & Information ⏎ Controllers of Personal Information ⏎ Discounted Cleaning Supplies, at 137 - 139 West Street, Crewe, Cheshire, CW1 3HH (Company Number 07663214) are data controllers under<< GDPR>> of personal information collected and processed through our`
4. `L00_N8146_pos` (`pair_idx=14`) | `act_dir=0.652` (`20.0%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `ightyText 5.3.1 (32-bit) LATEST ⏎ Windows 7 / Windows 8 / Windows 10 ⏎ User Rating:Click to vote ⏎ Author / Product: ⏎ MightyText (32-bit) 2020 full offline installer setup for<< PC>> ⏎ Features and Highlights ⏎ Get notified on`
5. `L00_N8146_pos` (`pair_idx=19`) | `act_dir=0.605` (`18.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Francisco Police Department announced that they believe they have found the people responsible for killing Canadian backpacker Audrey Carey in Golden Gate Park last weekend — and that the same suspects are implicated in another high-profile Bay Area slaying on Monday. ⏎ If you haven't already read about Carey, who was found dead near G<<GP>>'s bison paddock on Saturday morning`

**`p97–99` Positive Snippets (5 sampled):**
1. `L00_N8146_pos` (`pair_idx=0`) | `act_dir=0.048` (`1.5%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` level, SS appliances, tankless water heater, flat roof (2 years old), pitched roof (8 years old), windows, siding, electric and plumbing all 7 years old. Wait- there's still room for value add! An unfinished basement with a half bath, walk-out access to your<< incredible>> yard that includes lush vegetable garden with blo`
2. `L00_N8146_pos` (`pair_idx=2`) | `act_dir=0.044` (`1.3%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` Price ⏎ Call For Price 404-752-6715 ⏎ Star 748TA Specifications ⏎ Please Note: Picture is for reference. ⏎ C o n s t r u c t i o n : ⏎ Ultra-Max griddles feature 1” thick highly polished<< steel>> griddle plate, 5” tapered`
3. `L00_N8146_pos` (`pair_idx=4`) | `act_dir=0.037` (`1.1%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` not lose composure in front of my covered face. My jovial spirits did not let them sense any paranormal activity around me. ⏎ The camera was one of his prized possessions that conveyed his immortal passion for images, so I did not let it go. More than a tribute to the artist, the camera<< helped>> me learn the ropes of photography. On`
4. `L00_N8146_pos` (`pair_idx=7`) | `act_dir=0.037` (`1.1%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` buttocks in second place (25%), and elastic hips on the third (11%). ⏎ The survey showed that the British as ideal of beauty and sexuality are increasingly choosing athletic body. Popular models of "size zero" and skinny girls to the past, which is not good. More than half<< of>> women (51%) would like to`
5. `L00_N8146_pos` (`pair_idx=9`) | `act_dir=0.043` (`1.3%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` made of vegetable tanned leather. It is suitable for children and smaller people with a torso length of approx. 48 cm and a chest circumference of 66 cm up to 82 cm. The armor is nicely priced and is a good base for a leather LARP outfit of a<< hunter>> or warrior. The armor has hand stitched`

### 6. Neuron `k=17` (`L00_N7325`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L00_N7325_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.950`, `p97–99 = 0.700` |
| Ulto85 Causal Scores | `sim_cal_rec = -7.110`, `sim_cal_rec_active = +0.222`, `strong20_rec = +0.402` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.064`, `p97–99.9 = 0.161`, `p75–97 = 0.024`, `< p75 = 0.751` |

**Verbatim Ulto85 Pole Label(s):**
- `L00_N7325_neg`: Strong negative on the subword token 'el', 'els' or 'elson' when it is a word-internal continuation of a name or word fragment. Examples: Nick|el (-10), Nick|els (-9), Gock|el (-9), Eb|els|berg (-8), Midd|el|burg (-8), Mick|elson (-8), Cap|el (-7), Papp|el|allee (-7), Sig|el (-7), Eiff|el (-7), Heg|el (-7), Matt|el (-5), Hamm|el (-4), Trav|els (-3), Picker|el (-3), Ad|elson (-3), Buss|el|ton (-2). A few other mid-word pieces score mildly negative, around -1 to -4: Trip|lic, Fug|itive, Gend|ai.

**Top-100 Positive Snippets (5 sampled):**
1. `L00_N7325_neg` (`pair_idx=0`) | `act_dir=0.641` (`27.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Under. ⏎ The “Sydney Moderns” who dominated the Australian scene between World Wars I and II were at once engaging with revolutionary artistic concepts and reflecting the mercurial modern age. This volume comprises some 400 works by Ralph Balson, Frank and Margel Hinder, Roland Wak<<elin>>, and others in the Australian vanguard`
2. `L00_N7325_neg` (`pair_idx=5`) | `act_dir=0.543` (`23.6%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` stories are entwined rather than sequential, and anyway, Zafon stuffs long chunks of catch-up exposition into his dialogue with an unabashed glee that is, like almost every other storytelling trick he tosses into this paella of a yarn, hard to resist. ⏎ Young Daniel Semp<<ere>>, our narrator, takes up the saga`
3. `L00_N7325_neg` (`pair_idx=10`) | `act_dir=0.543` (`23.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` botanical terpenes to make your happy hour that much happier. ⏎ - 3 TBS grapefruit juice ⏎ - 2 TBS fresh lime juice ⏎ - 1.5 – 2.5 TBS (depending on your sweet tooth) maple syrup ⏎ - Daytrip Grapefruit CBD Spark<<ling>> Water ⏎ - Sliced limes and`
4. `L00_N7325_neg` (`pair_idx=14`) | `act_dir=0.594` (`25.9%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` situation, and ⏎ - our responses - physiological, behavioral and emotional ⏎ Different Types of Stress ⏎ It’s important to distinguish between Type I stress and Type II. Type I refers to discrete events that have a readily identifiable solution. Most of the examples above are Type I, e.g., the burned bag<<el>>, the inkless printer. When you`
5. `L00_N7325_neg` (`pair_idx=19`) | `act_dir=0.699` (`30.4%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` similar paths when the outrageous becomes something tangible. ⏎ Overall, find your comfy corner of the universe and enjoy tales from Burning Bulb veterans like Rich Bottles Jr., Nelson W. Pyles, Jon Towers, David J. Fairhead, Michelle Bowser, Solon Tsangaras and Paul Warg<<elin>>. Then you will also get into the`

**`p97–99` Positive Snippets (5 sampled):**
1. `L00_N7325_neg` (`pair_idx=0`) | `act_dir=0.047` (`2.1%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` up the spikes after this season, and the surefire Hall of Famer is being sure to give back before he calls it a career. ⏎ Rivera already took the time to meet with and sign autographs for a large group of Cleveland Indians team staff back in April. On Wednesday, during his final<< regular>>-season trip to Oakland, the closer arrived`
2. `L00_N7325_neg` (`pair_idx=2`) | `act_dir=0.051` (`2.2%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `013 meal. I believe this kind of hot clone labor protrudes. ⏎ It meets the needs and costs of other consumer clothes. This review is more comfortable, sets replika the beautiful and variation of the traitor. This location uses two plates, and jomashop the site subpar<<aph>>es are two mirrors representing the moon.`
3. `L00_N7325_neg` (`pair_idx=4`) | `act_dir=0.033` (`1.4%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` stuff and we strongly believe in forming your own opinion. There's no better way to do that than to do something yourself. People may overhype some lame plots, some weak origin stories, but you know that you have to just read the thing for yourself to have your own opinion. As of<< right>> now, just the latest buzz – what`
4. `L00_N7325_neg` (`pair_idx=7`) | `act_dir=0.042` (`1.8%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` quality consulting services range from routine monitoring, modeling, permitting, and air pollution compliance projects to non-traditional projects with a focus on unique products that require special handling. We efficiently manage more traditional projects by applying time-tested methods and delivering high-quality products. Non-traditional projects will have one or<< more>> unique characteristics, such as new or unfamiliar`
5. `L00_N7325_neg` (`pair_idx=9`) | `act_dir=0.044` (`1.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` mining companies with sound management will prosper, and a mixed basket of rare earth, uranium, and copper seemed to be popular secondary holdings. ⏎ No one believed we had finished the current Bull Market with ongoing geo-political events propping the metals up at times they were long overdue for natural corrections, and the<< impending>> financial crisis looming over the United States and`

### 7. Neuron `k=31` (`L04_N1558`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L04_N1558_pos`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.950`, `p97–99 = 0.600` |
| Ulto85 Causal Scores | `sim_cal_rec = -1.104`, `sim_cal_rec_active = -0.147`, `strong20_rec = +0.320` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.234`, `p97–99.9 = 0.153`, `p75–97 = 0.079`, `< p75 = 0.534` |

**Verbatim Ulto85 Pole Label(s):**
- `L04_N1558_pos`: Non-initial subword tokens inside rare foreign or exotic proper names and loanwords, usually tokens beginning with or ending a syllable on 'i': -ika, -ima, -iro, -ide, -iny, -ito, -ite, -izo, -ini, -ina, -imb, -itia, -ik, -iman, -ich, -idi. Examples: chit«ika», Brah«ima», Shuich«iro», Nik«ide», Ar«ima», Yak«ima», Akih«ito», chor«izo», Rim«ini». The activation sits on the fragment that continues the word, not on the first piece. Japanese, African, Italian, Indian and Sanskrit names are typical.

**Top-100 Positive Snippets (5 sampled):**
1. `L04_N1558_pos` (`pair_idx=0`) | `act_dir=1.211` (`25.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` have Zakiya on my Facebook and she announced the she will be on the Golf Channel show the “Big Break Atlantis” in May. I was so excited to hear this and sent her a good luck and that I will be sure to be watching this show in May. I have known Zaky<<ia>> since 2006 and now`
2. `L04_N1558_pos` (`pair_idx=5`) | `act_dir=2.812` (`59.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `" is a professional MIDI Backing Track. Press the blue PLAY button to hear "Little L". View and print "Little L lyrics" and view all "Jamiroquai" MIDI and MP3 backing tracks via links under the DETAILS section. ⏎ Little L | Live Band Backing Track | Jam<<iro>>quai ⏎ "Little L" backing`
3. `L04_N1558_pos` (`pair_idx=10`) | `act_dir=2.859` (`60.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` universal approach to Funkineven’s tilted West London swing creates a chemistry and understanding reminiscent of the greats that came before them. 808’s and MPC’s run amok whilst Fatima injects the soul and sex appeal with her sultry vocal. Their previous collaborative efforts includeFunk<<ine>>ven’s ‘Kleer’ (`
4. `L04_N1558_pos` (`pair_idx=14`) | `act_dir=3.891` (`81.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` snacks. They even have festival special videos where they demonstrate how to make delicacies specific to each occasion. Continue reading » ⏎ Happy New year ! May 2010 bring happiness, health and prosperity to you all. Yesterday we had the most relaxed New Years celebration. We went to Nilig<<iris>> Cake exhibition in St Josephs High School`
5. `L04_N1558_pos` (`pair_idx=19`) | `act_dir=1.648` (`34.7%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Firemen’s Parade, which steps off at 6 p.m. June 14. ⏎ “We have local departments that we work with daily and ones as far away as New York,” said Mike Pratt, former fire chief and chairman of the carnival, about the parade. ⏎ First Selectman Christopher Biel<<ik>> said the town is always pleased to see`

**`p97–99` Positive Snippets (5 sampled):**
1. `L04_N1558_pos` (`pair_idx=8`) | `act_dir=0.169` (`3.6%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - `. ⏎ BACKGROUND: The aim of this study was to assess the efficacy and safety of combination regimen of capecitabine plus everolimus in patients with refractory gastric cancer who have failed to at least two cytotoxic regimens. METHODS: Patients received capecitabine 650<< mg>>/m(2) twice daily (D`
2. `L04_N1558_pos` (`pair_idx=2`) | `act_dir=0.222` (`4.7%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` the recent Spring Carnival. ⏎ In a bid to eliminate cheats RV has beefed up its integrity department, planning to boost the number of stable raids and horse testing and intensify scrutiny. ⏎ A string of scandals, including allegations of race-fixing and horse doping, the long-running Danny Nikolic saga and an<< admission>> of betting by pin-up jockey Damien`
3. `L04_N1558_pos` (`pair_idx=4`) | `act_dir=0.163` (`3.4%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - `aping and being majestic. You imagine feathers of all colours, the soft lullaby of birdsong in so many different birdtongues, mothers bringing food to their young. It is romantic, you think. Maybe I could sit on a bench under these trees, you consider, and be reminded<< of>> nature, and the splendour of`
4. `L04_N1558_pos` (`pair_idx=7`) | `act_dir=0.213` (`4.5%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` railing with gate: ⏎ See A professional ⏎ Pay a visit to the experts in balustrades, Sydney. They will be able to show you a wide variety of options for your deck railing and even illustrate what it would look like in situ. This will help you to choose the best possible solution for your<< individual>> and unique home. ⏎ The professionals will also`
5. `L04_N1558_pos` (`pair_idx=9`) | `act_dir=0.213` (`4.5%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` euphoric feeling similar to opioids. However, because it doesn’t bind to the same opioid receptors, these effects are not as potent as with an opioid. ⏎ Who Uses Kratom? ⏎ Although it has been used as a traditional medicine for centuries in South Asia, these days when used in the United States<< it>> is generally used for one of three reasons`

### 8. Neuron `k=93` (`L12_N3077`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L12_N3077_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 1.000`, `p99.9+ = 0.950`, `p97–99 = 0.600` |
| Ulto85 Causal Scores | `sim_cal_rec = -1.067`, `sim_cal_rec_active = +0.522`, `strong20_rec = +0.245` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.092`, `p97–99.9 = 0.268`, `p75–97 = 0.184`, `< p75 = 0.456` |

**Verbatim Ulto85 Pole Label(s):**
- `L12_N3077_neg`: Text about taking or bringing something to another place, or about portable and mobile service. The strongest values fall on the preposition 'to' or 'on' and on the words of the movement idiom: 'taking their show on the road' (on -10, the -7, their -5, road -5, show -3), 'take their series to' (to -7), 'take to guild' (-6), 'take to an emergency scene' (take -5, to -4), 'bring your yoga to the mat' (to -5), 'bringing my CDs to the YMCA' (to -5). It also fires on nouns for mobile or portable service and outing settings: 'room service' -5, 'martini trolley' -5, 'mobile truck' -5, 'mobile canner' -3, 'coffee cart' -3, 'service vans' -3, 'picnic' -4/-5, 'delivered locally' -3, and 'with a signature cocktail' (a -3).

**Top-100 Positive Snippets (5 sampled):**
1. `L12_N3077_neg` (`pair_idx=0`) | `act_dir=3.938` (`28.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `, loppers, pruners etc. please bring them along but there is plenty of work to be done that doesn’t require tools. ⏎ Old clothes and boots, gardening gloves and waterproofs if it’s wet. Please don’t wear your best ‘Arcteryx’ gear! Bring a packed<< lunch>> and a drink please! ⏎ Any questions,`
2. `L12_N3077_neg` (`pair_idx=5`) | `act_dir=7.469` (`54.3%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `I've gained 40 lbs", she said, "My butt is the size of Cleveland." She just needed clothes that fit to reclaim her lush beauty. ⏎ Here's where to find clothes with body-skimming, not swamping, lines. ⏎ 1. Take a favorite pair of basic pants or skirt<< to>> a tailor and have them copied. ⏎ L`
3. `L12_N3077_neg` (`pair_idx=10`) | `act_dir=2.578` (`18.8%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `'s Wedding ⏎ "• Bartenders: VIP Bartending and Event Planning The two bartenders that worked at our wedding were great; they poured proper drinks and made sure everyone had a great time without going too far. They did cut people off when they felt the guests had consumed enough. They rented a<< portable>> bar and set it up in the venue`
4. `L12_N3077_neg` (`pair_idx=14`) | `act_dir=5.438` (`39.5%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `, preparing and delivering lunches in the greater Bridgeport and New Haven areas. In addition to Prep's food service provider Vazzy's, several Prep community restaurants have joined the team, including Little Pub, Old Post Tavern, Gaetano's and Country Cow. ⏎ When a Vazzy’s catering<< truck>> packed full of bagged lunches pulled up`
5. `L12_N3077_neg` (`pair_idx=19`) | `act_dir=6.312` (`45.9%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` of the Gospel, one person, one community at a time. ⏎ At Remembrance Community Church, we believe that God is on a mission to rescue a world hurt by sin. ⏎ We believe that God has reconciled us as individuals and as a church and has given us the ⏎ ministry of reconciliation to<< take>> part in carrying out His glorious work of`

**`p97–99` Positive Snippets (5 sampled):**
1. `L12_N3077_neg` (`pair_idx=0`) | `act_dir=0.227` (`1.6%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - `, but it turned out that none of them was indeed the lost artifact. We found our factions combating each other, near the water and in the alley. It was all ranged combat and the upper hand of the battle changed several times. In the end Chris once again managed to remove my Detective Inspector,<< I>> think for the fourth game in a row`
2. `L12_N3077_neg` (`pair_idx=2`) | `act_dir=0.262` (`1.9%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` of solute or liters of solution given the other two variables. ⏎ 3. Calculate the amount of concentrated solution to mix a given amount of dilute solution. ⏎ 4. Solve stoichiometry problems involving solutions of known molar concentrations. ⏎ 5. Identify the limiting reactant (or reagent) in a<< chemical>> reaction and predict the amount of precipitate`
3. `L12_N3077_neg` (`pair_idx=4`) | `act_dir=0.219` (`1.6%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` gyratory crushing and conveyor system by 2016. The belt ⏎ Nov 7, 2017 Welcome back to Gold Rush The Game! Today on Gold Rush we're FINALLY moving everything over to our new gold mining site in River Town! I also buy the conveyor system<< which>> is the final pieces that we need to`
4. `L12_N3077_neg` (`pair_idx=7`) | `act_dir=0.246` (`1.8%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` that implied Microsoft's Kinect motion-control camera for the Xbox 360 might recognize sign language. ⏎ In the patent, an illustration depicts a user spelling words using their hands. ⏎ "The user...is making a gesture with his left hand...to signal the character 'a' in American Sign Language (AS<<L>>)," reads the accompanying text. The patent`
5. `L12_N3077_neg` (`pair_idx=9`) | `act_dir=0.279` (`2.0%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` and before he disappeared in 1995, sightings of the mobster occurred on a daily basis. ⏎ “I’ve seen him in the flesh. Broadway, Dorchester. Up around Cathy Grieg’s house on Fourth Street,” said Maureen Manning, walking her dog down Broadway. ⏎ Whitey Bul<<ger>> was both admired and feared in this neighborhood`

## Group C: Failing Top-100 Detection (`< 0.800`)

### 9. Neuron `k=32` (`L04_N2686`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `bipolar` (`L04_N2686_pos`, `L04_N2686_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 0.450`, `p99.9+ = 0.600`, `p97–99 = 0.750` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.008`, `sim_cal_rec_active = +0.040`, `strong20_rec = +0.525` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.203`, `p97–99.9 = 0.457`, `p75–97 = 0.279`, `< p75 = 0.061` |

**Verbatim Ulto85 Pole Label(s):**
- `L04_N2686_pos`: Fires on a content word that opens a slot the next words must fill. The activation sits on that word itself, not on the filler that follows. Main cases: - Transitive or linking verbs and prepositions right before their object or complement: 'among' (the best), 'demonstrates', 'became'/'become'/'becoming' (a citizen, president), 'enables' (us to), 'justify', 'relates' (to), 'spent' (last night), 'spend', 'handle', 'involving', 'by' (Apple). - Sentence-subject nouns just before their verb: 'bitcoin is', 'graduate, and', 'independent peer-run'. - The close-paren of an enumeration marker at the start of a list item: '1)', '2)'. These reach about +5 to +6. Weaker hits: nouns that introduce a following 'of'/'to' phrase ('example' of, 'implications', 'plug' of, 'heat' demands), and the article 'a' in some predicate positions.
- `L04_N2686_neg`: Strongly negative (about -10) on a closing parenthesis ')' or '.)' that ends a parenthetical inserted mid-sentence, after which the main clause resumes. Examples: 'Smith (R-TX) said…', '(…) was quickly surrounded', '(…) who must survive'. The token right after often gets a small positive (+1 to +3). Moderately negative (-3 to -4) on the indefinite article 'a' or token-final 'a' in some contexts: 'a poor second', 'a midnight rendezvous', 'a.m.', 'a 5.75'. Also on some possessive 's. Sentence-final periods and newlines sometimes reach about -1.

**Top-100 Positive Snippets (5 sampled):**
1. `L04_N2686_pos` (`pair_idx=0`) | `act_dir=0.273` (`27.7%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` Help file offers some steerage, though it may stand to be more detailed and better written. Bonus Room AboutBonus Room About. The Mac App Retailer has a wide 1checker array of Productiveness apps in your Mac. And select ” 1Checker Online. 1Checker is your very<< best>> resolution for proofreading. Download the Add`
2. `L04_N2686_pos` (`pair_idx=10`) | `act_dir=0.266` (`26.9%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - `whether in public, educational, college or distinctive libraries—with the fundamental instruments of undertaking administration so one can achieve self assurance and an expectation of luck. ⏎ What does it suggest to have significant metrics in modern complicated larger schooling panorama? With a foreword through Heather Piwowar and Jason Priem, this hugely<< attractive>> and activity-laden booklet serves to introduce`
3. `L04_N2686_pos` (`pair_idx=19`) | `act_dir=0.228` (`23.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` leading them to spend more once they're inside. In turn, this can improve client retention, as they are more probable to return if their experience is a positive one. Should you be looking to kickstart a more seamless system for entry, these are the effective measures you can take. ⏎ 1. Design<<ate>> separate lines for each type of guest. ⏎ `
4. `L04_N2686_neg` (`pair_idx=0`) | `act_dir=0.220` (`22.2%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` is prepared to invest time acquiring to know you so you know they recognize you and your requirements. This is the ideal way to locate out if they are a excellent in shape for you. ⏎ Your wedding photographer need to be educated and valuable. They ought to offer ideas about choices you have for diverse issues,<< supply>> guidance on scheduling and set up, be`
5. `L04_N2686_neg` (`pair_idx=19`) | `act_dir=0.181` (`18.3%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` company’s earnings is defined as the earnings that it gains with selling goods or services. A service can not be exclusively profit-oriented, as well as can not be exclusively based on customer satisfaction. There are 2 main kinds of businesses: for-profit and also not-for-profit. If an individual is marketing<< furniture>> as well as marketing it on a regular`

**`p97–99` Positive Snippets (5 sampled):**
1. `L04_N2686_pos` (`pair_idx=0`) | `act_dir=0.051` (`5.2%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` influenza and other harmful pathogens. ⏎ "We are excited and flattered to have this opportunity in sharing our continued technology developments with such an important organization. DHHS is certainly qualified to offer constructive advice on ANI's primary objective of proactively protecting people from harmful disease," said Thomas K. Allen, President and<< CEO>>, ANI. "In addition,`
2. `L04_N2686_pos` (`pair_idx=4`) | `act_dir=0.039` (`4.0%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` walk in to any of our five stores, or when the technician repairs your current phone, you will know you have made the right choice for all of your Portland cell phone needs. Need a used cell phone or smart-phone for AT&T, Verizon, Sprint, Cricket or T-Mobile? We got it<<!>> Visit your local Portland cell phone store location`
3. `L04_N2686_pos` (`pair_idx=9`) | `act_dir=0.059` (`6.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` of the most advanced orchestration tools that currently exists in the software world. It provides out-of-the-box automation for environment maintenance and simplifies deployment and upgrade processes. It has different implementation types (on-premise, cloud-managed, hybrid, and more), multiple open-source supporting tools, and supports<< a>> wide range of configuration options. But the`
4. `L04_N2686_neg` (`pair_idx=0`) | `act_dir=0.063` (`6.4%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `ac, Dominique Ropion, Daniela (Roche) Andrier, Antoine Maisondieu, Guy Robert, Karine Dubreuil, Maurice Roucel, Firmenich, Ilias Ermenidis, Givaudan, Jacques Huclier and Lorenzo Villoresi<<. ⏎ >>Currently, Gucci has 52`
5. `L04_N2686_neg` (`pair_idx=9`) | `act_dir=0.042` (`4.3%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` 28. ⏎ Vatican City produces a portion of its electricity by solar panel, but much is imported from Italy and is therefore supplied at 220V/50Hz. US travelers will need a voltage converter in order to use appliances from home which are not designed for dual voltage.<< Plug>> sockets in Vatican City generally accept round pr`

### 10. Neuron `k=82` (`L08_N8930`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `bipolar` (`L08_N8930_pos`, `L08_N8930_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 0.775`, `p99.9+ = 0.600`, `p97–99 = 0.550` |
| Ulto85 Causal Scores | `sim_cal_rec = +0.029`, `sim_cal_rec_active = +0.552`, `strong20_rec = +0.203` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.318`, `p97–99.9 = 0.368`, `p75–97 = 0.292`, `< p75 = 0.021` |

**Verbatim Ulto85 Pole Label(s):**
- `L08_N8930_neg`: Tropical fruit and plants: 'banana', 'bananas', 'bann|ana', 'palm' (as in palm trees or Palm Desert) and 'coconut'. The noun token itself carries the activation, about −3 to −4. The following 'trees' gets about −2. 'Apple', 'orange' and other fruits do not fire.
- `L08_N8930_pos`: PC hardware and CPU-architecture context. The strongest trigger is the digits inside x86 architecture names. Examples: the '8' and '6' in 'i386' and 'i686', the '6' and '4' in 'x64', 'x86', 'AMD64' and '64bit'. The company name 'Intel' (and 'intel', 'Pentium') also fires reliably. Weaker hits come from other low-level PC and OS jargon: 'page fault', 'CSUM', register numbers, Compaq model numbers, and Windows provisioning terms like 'ESP'. A few odd name-suffix pieces also fire weakly to moderately, e.g. 'Calzag|he', 'Pachel|bel', 'kras|inski', 'X|imin'.

**Top-100 Positive Snippets (5 sampled):**
1. `L08_N8930_neg` (`pair_idx=0`) | `act_dir=1.242` (`11.4%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - `Drink this: Sneak leafy greens into smoothies. Try adding baby spinach to your next healthy shake. You'll get all the goodness with none of the grassy taste! ⏎ - NUTS and SEEDS: All excellent sources of magnesium, walnuts, flax, hemp and pumpkin<< seeds>> contain the omega-3 fatty acid AL`
2. `L08_N8930_neg` (`pair_idx=10`) | `act_dir=1.031` (`9.5%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - `2018 book, Beneath A Surface, that Microsoft would launch a new modular Surface Studio in 2020. ⏎ Sams suggests this modular Surface Studio would allow owners to swap the computer unit, which could potentially include the processor, GPU, RAM and storage. Since monitors don’t<< need>> to be upgraded or updated as often as`
3. `L08_N8930_neg` (`pair_idx=19`) | `act_dir=4.188` (`38.5%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` certain foods that cause problems for digestion. If you have frequent problems with bowel movements, you should know which diets make the intestine particularly sluggish. You may Download Youtube video to mp4 if you have some files regarding foods or recipes that need to be converted. ⏎ Do bananas have a puffing effect? ⏎ Ban<<anas>> have a general reputation for promoting constipation`
4. `L08_N8930_pos` (`pair_idx=0`) | `act_dir=0.902` (`8.3%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` Romney ran Bain Capital for three years longer than he says he did. ⏎ The newspaper reported Thursday that Securities and Exchange Commission documents filed after 1999, the year Romney said he left Bain, showed he was still indicated as the firm's "sole stockholder, chairman of the board, chief<< executive>> officer and president," and may have stayed`
5. `L08_N8930_pos` (`pair_idx=19`) | `act_dir=1.242` (`11.4%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` nutrients necessary for the proper functioning of an adult dog’s body. A particularly important advantage of the food with salmon is the high content of n-3 fatty acids, which have a positive effect on the nervous system and the condition of the skin and coat. The fulfilled demand for other nutrients, minerals and<< vitamins>> means that the use of any additional complementary`

**`p97–99` Positive Snippets (5 sampled):**
1. `L08_N8930_neg` (`pair_idx=0`) | `act_dir=0.185` (`1.7%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` Minnesota, these flags have been carefully cleaned, documented, and stitched to specially prepared mounts to allow visitors to see each one fully unfurled. The flags had been displayed at the capitol since its dedication in 1905. ⏎ The first four flags to go on display are the First Minnesota<< Volunteer>> Infantry Regiment, the Third Minnesota Volunteer Infantry`
2. `L08_N8930_neg` (`pair_idx=4`) | `act_dir=0.192` (`1.8%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` among both children and adults. Dental professionals use fillings to remove the decayed parts of a tooth and replace them with durable and artificial material. ⏎ Fillings restore healthy mouths by helping to prevent future cavities while keeping the structure of the infected tooth intact. Unfortunately, over 90% of Americans<< experience>> cavities, with many of those going`
3. `L08_N8930_neg` (`pair_idx=9`) | `act_dir=0.197` (`1.8%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` shut down. ⏎ The sense of indignant outrage appears to have blurred political divides too. Popular Left-wing social activist Pasam Padma said there was no guarantee that “nothing untoward and obscene” would happen at the Playboy Club. “How could permission be given to such a notorious club when the government<< appointed>> a woman tennis star as its brand ambassador`
4. `L08_N8930_pos` (`pair_idx=3`) | `act_dir=0.270` (`2.5%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` village, Belessa, Ethiopia is completed! It was furnished with school desks and blackboards and handed over to the local community education office. ⏎ Thank you to the local American Jewish Joint Distribution Committee (JDC) in the endeavor to build this school with the help of the local community and to Path<<ways>> to Children, a non-profit based in`
5. `L08_N8930_pos` (`pair_idx=9`) | `act_dir=0.262` (`2.4%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` brush. ⏎ The Nail Decal I used are from Born Pretty Store that was sent to me for review. It’s a big sheet of Nail Decals. You can do variety of nail designs just with this one sheet! It consists of 12 patterns for both set of nails. One set consists<< of>> 5 designs and comes in different sizes`

### 11. Neuron `k=265` (`L35_N3526`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `bipolar` (`L35_N3526_pos`, `L35_N3526_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 0.550`, `p99.9+ = 0.525`, `p97–99 = 0.450` |
| Ulto85 Causal Scores | `sim_cal_rec = -0.723`, `sim_cal_rec_active = -0.005`, `strong20_rec = +0.076` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.519`, `p97–99.9 = 0.262`, `p75–97 = 0.195`, `< p75 = 0.024` |

**Verbatim Ulto85 Pole Label(s):**
- `L35_N3526_neg`: Other tokens in the same kind of boilerplate, especially the following. 1) Modal or negated auxiliaries in disclaimer and aphorism sentences: " may" in "any use which may be made" (-10), " must" (-9), " cannot" (-6). 2) Head nouns or verbs that close a clause and set up the next predicate: " site" in "Wikia is a free-to-use site that" (-7 to -9), " contact" in "...information, contact" (-9), " update" in "Cannot delete or update a parent row" (-8). 3) Subject+verb starts: " have" in "We have a modified experience" (-6 to -7), " from"/" advertising" (-3 to -4), " you" (-3 to -7), and a comma before "we provide" (-9). Milder negatives of -1 to -5 show up on licensing, cookie, signup and company-blurb text.
- `L35_N3526_pos`: Tokens in stock website/legal/meta boilerplate that lead into or name the specific content of the formula, typically just before or on the payload phrase. Examples: " using" in "for viewers using ad blockers" (+8 to +9), with " ad" +4 to +5 and " blockers" +6 right after; " to" in "using extensions to block ads" (+7); "descending"/" order" in forum sort instructions (+7); "ographic" in "bibliographic or download information" (+6); " intelligent" in a comment-moderation policy (+6); " embed" (+4); " style" in "with style sheets (CSS) enabled" (+4).

**Top-100 Positive Snippets (5 sampled):**
1. `L35_N3526_neg` (`pair_idx=0`) | `act_dir=21.625` (`31.8%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` 'apt-get autoremove' to remove them. ⏎ 0 upgraded, 0 newly installed, 1 reinstalled, 0 to remove and 0 not upgraded. ⏎ Need to get 0 B/28.0 kB of archives. ⏎ After this operation, 0 B of additional disk space<< will>> be used. ⏎ (Reading database ... `
2. `L35_N3526_neg` (`pair_idx=10`) | `act_dir=10.875` (`16.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` is the cluster of little islands in the Vembanad Lake in Kerala, is an enchanting backwater destination in God’s Own Country. Kumarakom houseboat rates are same to that of Alleppey houseboat charges. Houseboat day cruise rates are less than that of overnight stay. ⏎ <<You>> can reach Kumarakom from major cities`
3. `L35_N3526_neg` (`pair_idx=19`) | `act_dir=10.875` (`16.0%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - `. Sound enters the meter through a microphone, and then the meter evaluates the sound values by converting them into electrical information and the measurement values in decibels are displayed. An SPL Meter is generally handheld, mobile and lightweight, and is manufactured in line with international standards such as IEC 61<<6>>72. Pulsar Instruments manufactures`
4. `L35_N3526_pos` (`pair_idx=0`) | `act_dir=6.844` (`10.1%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - `ST ⏎ - Leviticus 18:2 Speak unto the children of Israel, and say unto them, I am the LORD your God. ⏎ - Leviticus 20:23 And ye shall not walk in the manners of the nation, which I cast out before you: for<< they>> committed all these things, and therefore I`
5. `L35_N3526_pos` (`pair_idx=19`) | `act_dir=8.625` (`12.7%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - `004. He retired as the most-capped South African forward (a mark that has since been eclipsed by John Smit and Victor Matfield) and is currently the fifth most-capped Springbok of all-time. ⏎ He played the majority of his provincial rugby with the Free State C<<heet>>ahs although he spent a season with`

**`p97–99` Positive Snippets (5 sampled):**
1. `L35_N3526_neg` (`pair_idx=0`) | `act_dir=0.551` (`0.8%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` huge varieties of people who have a effective, trouble-free Lasik surgery tend not to report a in-depth and long story of their experience, unlike those that have actually experienced trouble later. ⏎ However, it readies to know what the possible adverse effects of the Lasik treatment are, as well as the<< possibility>> that one of these negative effects will occur`
2. `L35_N3526_neg` (`pair_idx=4`) | `act_dir=0.566` (`0.8%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` one? Yes," Kornegay said. "I have been going to make a field application on a crop-protection flight, and all of a sudden something that wasn't there two days ago is there now. Because these things can go up fast." ⏎ Known as meteorological evaluation towers, or<< MET>>s, the dangers posed to crop d`
3. `L35_N3526_neg` (`pair_idx=9`) | `act_dir=0.668` (`1.0%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` camp. ⏎ Votes: 417,695 | Gross: $57.60M ⏎ 6. District 9 (2009) ⏎ R | 112 min | Action, Sci-Fi, Thriller ⏎ An extraterrestrial race forced to live in sl<<um>>-like conditions on Earth suddenly finds a kind`
4. `L35_N3526_pos` (`pair_idx=0`) | `act_dir=0.451` (`0.7%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` meet like-minded people and experience new places through road-trips. Each chapter of The RoadTripsCo has its own city lead who coordinates with the central team to ensure that all the drives happen smoothly. They assist the drive captains to enable the community that sets off on a road-trip to not<< just>> have a good time but a safe one`
5. `L35_N3526_pos` (`pair_idx=9`) | `act_dir=0.414` (`0.6%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - ` anchor trolley system. ⏎ These are some of the reasons why I have found kayaking so enjoyable: ⏎ 1. Kayaking is easy to do. I can carry and launch my kayak in the water without difficulty. No elaborate preparations are needed. The learning curve for effective paddling and maneuvering on smooth<< water>> is short. The trickiest parts are`

### 12. Neuron `k=119` (`L16_N2492`)

| Field | Value |
|---|---|
| Polarity / Active Poles | `unipolar` (`L16_N2492_neg`) |
| Detection (`gpt-6.1-terra@none`) | `top100 = 0.500`, `p99.9+ = 0.250`, `p97–99 = 0.400` |
| Ulto85 Causal Scores | `sim_cal_rec = -0.015`, `sim_cal_rec_active = -0.012`, `strong20_rec = +0.000` |
| 4-Band Exact `fp32` KL Share | `p99.9+ = 0.000`, `p97–99.9 = 0.133`, `p75–97 = 0.250`, `< p75 = 0.617` |

**Verbatim Ulto85 Pole Label(s):**
- `L16_N2492_neg`: Fires on tokens of a technical or domain term, usually a compound noun, that the passage is introducing, defining or explaining. The strongest values fall on the core subword of the term's head, especially when it is being defined. Examples: 'What is absorption costing? / Absorption costing goes by…' gives -7 on 'absorption' and -10 on 'orption', with -1 to -4 on 'costing'. Others: 'Hal|ogen' bulbs (-3), hip 'dys|pl|asia' (-2 to -3 on 'pl'), 'pass through entities' (-3/-2), 'S Corporations' (-2), 'Diastolic pressure' (-2), 'fixed rate' (-2), 'amino acids' (-2), 'array index' (-2). Weaker cases include an ellipsis mid-explanation ('you can…use them', -3), a label colon ('UPC Code:', -3) and 'Reviewed by' (-2). Magnitude grows when the term is being defined (questions like 'What is X?' or 'X is…'), when it is repeated, and when it is rare or jargon-like.

**Top-100 Positive Snippets (5 sampled):**
1. `L16_N2492_neg` (`pair_idx=0`) | `act_dir=0.523` (`22.0%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` Cline: Crazy for Loving You, a biographical exhibit opening Friday (Aug. 24). ⏎ Opening weekend festivities will include a panel discussion on Saturday featuring Cline’s husband, Charlie Dick; their daughter, Julie Fudge; Country Music Hall of Fame member Harold Bradley; and singers George Hamilton<< IV>> and Jan Howard. In addition, a`
2. `L16_N2492_neg` (`pair_idx=5`) | `act_dir=0.695` (`29.3%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - `5| ⏎ |File Size:||28.97 Mb| ⏎ |Operating Systems:||Windows NT/2000/XP/2003/2003/7/8/10 MacOS 10/X| ⏎ |Price:||Free* [*Free Reg<<sit>>ration Required]| ⏎ Have used Cobra before`
3. `L16_N2492_neg` (`pair_idx=10`) | `act_dir=0.527` (`22.2%` of max) | judge: `CORRECT` (pred `A`, pos `A`)
   - ` specific offset. Because Members of Congress are hesitant to vote for funding programs without offsets, they must search for ones with which to fund the NHTF. One proposed offset has been offered by Senator Jack Reed (D-RI) and Representative Elijah Cummings (D-MD). ⏎ The Preserving Homes and<< Communities>> Act of 2011,`
4. `L16_N2492_neg` (`pair_idx=14`) | `act_dir=0.488` (`20.6%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - `, however, retains the potential to disrupt society and tradition as society and tradition can interfere with the ideal functioning of market forces. Andrews reasoning pertains serious consequences for Australian conservatism. ⏎ Voters are not static they respond to perceived opportunities and perceived threats. Housing is becoming more unaffordable, secure employment is<< scar>>cer and the cost of living has increased`
5. `L16_N2492_neg` (`pair_idx=19`) | `act_dir=0.520` (`21.9%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` (FFV)Flex-fuel vehicles accept, and are powered by, regular gasoline or an ethanol blend, such as E85 — a fuel blend consisting of 85 percent ethanol and 15 percent gasoline requiring less demand on natural resources than traditional fossil fuels. ⏎ Fossil FuelsF<<ossil>> fuels (such as oil and natural gas`

**`p97–99` Positive Snippets (5 sampled):**
1. `L16_N2492_neg` (`pair_idx=0`) | `act_dir=0.239` (`10.1%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` of the falls. ⏎ - Outside the proper city limits of Superior, you’ll find beautiful town of Lake Nebagamon. Besides the shops and restaurants and other attractions, you’ll find a wonderful public beach and swimming. ⏎ - Lester River in Lester Park is the swimming hole of choice for many of Duluth<<’s>> residents. ⏎ - Sucker River is also`
2. `L16_N2492_neg` (`pair_idx=2`) | `act_dir=0.250` (`10.5%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` Extract the game and run “Tormentum” application file. Please turn off your antivirus for sometime as many antivirus program declare a clean application file as malware. We have tested and run it, all is clean but my Norton Internet Security detected it as malware. I have disabled it and then extracted and<< play>> it. ⏎ After running the file, you`
3. `L16_N2492_neg` (`pair_idx=4`) | `act_dir=0.277` (`11.7%` of max) | judge: `CORRECT` (pred `B`, pos `B`)
   - ` cf Wikipedia. These code lists are sometimes called UNCL (United Nations Code List). UNECE has standardized code lists for many different things: units of measure, payment means, modes of transport, packaging, taxes, etc…. ⏎ Table of contents ⏎ The configuration takes place in the menu Settings ><< Technical>> > Parameters > UNECE Code Lists`
4. `L16_N2492_neg` (`pair_idx=7`) | `act_dir=0.287` (`12.1%` of max) | judge: `WRONG` (pred `A`, pos `B`)
   - `. Look at when it came, and what it is. You think I needed to pay to get it today? Don’t worry I went back, and checked the shipping if you really need proof – but love is love. ⏎ Magic is magic – bottom line. Don’t demand to meet him – Just<< know>>. ⏎ Life will only spell it out if`
5. `L16_N2492_neg` (`pair_idx=9`) | `act_dir=0.271` (`11.4%` of max) | judge: `WRONG` (pred `B`, pos `A`)
   - ` Sometimes there is nowhere or nobody to turn to when the lights go out. So take this handbook (so to speak) with you into the night and be prepared rather than scared of the unknown. You see, we are the Fiends of the Flesh, and we know what frightens you or in<< some>> cases, entices you to open that`
