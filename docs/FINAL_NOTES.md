# Final Notes

## 1. How to run it

The project requires Python `>=3.10,<3.14` (verified on Python `3.10.13` and `3.13.15`). From a fresh clone, choose either `uv` or standard `pip`.

### Setup with uv

```bash
git clone https://github.com/flykrth/AI-Engineer-Task.git
cd AI-Engineer-Task
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

### Setup with standard pip

```bash
git clone https://github.com/flykrth/AI-Engineer-Task.git
cd AI-Engineer-Task
python3 -m venv .venv  # using Python 3.10 to 3.13
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### Environment variables

Edit `.env` and set your credentials for the chosen provider:

- Default (`gemini`): set `GEMINI_API_KEY` (uses `LLM_PROVIDER=gemini`, `GEMINI_CHAT_MODEL=gemini-3.5-flash-lite`, `GEMINI_EMBED_MODEL=gemini-embedding-001`).
- Optional OpenAI fallback: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` (defaults to `gpt-4o-mini` and `text-embedding-3-small`).
- Optional local Ollama fallback: set `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_CHAT_MODEL=llama3.1:8b`, and `OLLAMA_EMBED_MODEL=nomic-embed-text`.
- Optional rate-limit pacing: set `LLM_REQUEST_DELAY_SECONDS` (default `0.0`; set e.g. `1.5` or `2.0` if hitting strict free-tier requests-per-minute limits).

### Running the tool

```bash
# 1. Interactive Rich CLI (prints State of the project, then opens Q&A prompt)
python run.py --chat data/team_chat.txt

# 2. State of the project only (terminal tables, or --json for machine-readable output)
python run.py --chat data/team_chat.txt --state-only
python run.py --chat data/team_chat.txt --state-only --json

# 3. One-shot Q&A in English or Hinglish (text or structured --json)
python run.py --chat data/team_chat.txt --question "Who currently owns payments and is it finished?"
python run.py --chat data/team_chat.txt --question "Kya localhost pe demo chalega aur chat feature kitna hua hai?" --json

# 4. Run on any unseen WhatsApp export (use --rebuild to bypass .cache/)
python run.py --chat tests/fixtures/unseen_ios_24h_chat.txt --state-only
python run.py --chat tests/fixtures/unseen_ios_24h_chat.txt --question "How is badge printing handled now and how did that change?" --json

# 5. FastAPI single-page web dashboard (default http://127.0.0.1:8000)
python run.py --chat data/team_chat.txt --web
python run.py --chat data/team_chat.txt --web --host 127.0.0.1 --port 8010

# 6. Automated test suite and evaluation runner
pytest -v
python eval/run_eval.py --suite eval/team_chat_eval.json
python eval/run_eval.py --suite eval/unseen_eval.json --rebuild
```

## 2. How it works (one paragraph)

First, the tool cleans hidden formatting marks out of an exported WhatsApp text file, figures out whether the dates are day-first or month-first, groups multi-line messages under their original timestamp and sender, and gives every message a permanent reference tag. Next, it reads through the chat in chronological order from start to finish to maintain a running project scoreboard that tracks current decisions, task owners, deadlines, and open issues along with their full revision history whenever plans flip, tasks change hands, or dates move. When you ask a question in English or Hinglish, it searches the chat using both exact word matches and meaning-based similarity, pulls in neighboring messages around each match so short replies and photo placeholders stay tied to their context, and sends both the running scoreboard and those raw snippets to the language model to draft an answer. Finally, a strict rule-based checker verifies every message reference in the draft against the real chat file, rejects guesses about deleted messages or future promises that were never confirmed as finished, and either shows the verified answer with exact citations or says "the chat doesn't say".

## 3. My test set and results

All 15 questions below were worked out by hand from `data/team_chat.txt` in `eval/team_chat_eval.json` and run end-to-end with `rebuild=True` using `gemini-3.5-flash-lite` and `gemini-embedding-001`. Note that the automated `eval/run_eval.py` report includes naive literal-substring hint checks only as a pre-audit diagnostic; the table below uses strict manual grading against actual outputs.

| # | Question | Correct answer (by hand) | System's answer | Correct? | Notes |
|---|---|---|---|---|---|
| 1 | Who currently owns payments, what payment method is being used, and is the feature finished? | Kabir currently owns payments using Razorpay test mode. It is not finished yet: checkout is working, but the payments webhook is still remaining. | Kabir currently owns payments, using Razorpay test mode. The chat doesn't say whether payments finished, as the webhook was still pending. | Yes | Accurate owner, method, and pending webhook status (`M062, M066, M067, M086`). |
| 2 | What was the original decision about payments and how did it change over time? | Initially Meera owned payments and planned to use Razorpay (`05/10/26`). On `20/10/26` the team dropped Razorpay due to missing business KYC and switched to UPI QR manual confirm. On `02/11/26` Kabir found Razorpay test mode works without KYC, so they scrapped UPI QR, reverted to Razorpay test mode, and Kabir took over payments while Meera focused on report and PPT. | Initially, the team planned to use Razorpay, but after realizing it required business KYC, they decided to switch to a UPI QR code with manual confirmation. Later, they reverted this decision when they found out Razorpay test mode works without KYC for demo purposes, so the final decision was to use Razorpay test mode and scrap the QR code. | Partial (No) | Real miss under strict grading: traced the full Razorpay -> UPI QR -> Razorpay test mode reversal accurately (`M047, M050, M051, M062, M063, M065, M066`), but omitted the Meera to Kabir ownership swap. |
| 3 | When is the project report due, and what rule determines that deadline? | The project report is due `23 Nov`. It is governed by the rule that the report is due 3 days before demo day (which shifted from `4 Dec` to `27 Nov` and was finalized as `26 Nov`). | The project report is due on November 23, and the deadline is determined by the rule that it is due three days before the demo day. | Yes | Fully correct (`M028, M072, M073, M079, M080, M081`). Automated literal-string hint flagged `"November 23"` vs `"23 Nov"` and `"three days before"` vs `"3 days before"`. |
| 4 | When is the final demo day, how did the date shift, and can the team demo on localhost? | Final demo day is `26 Nov` (Thursday). It shifted from `4 Dec` to `27 Nov` (Friday) on `05/11/26`, then was corrected to `26 Nov` on `09/11/26` due to a typo. Localhost is not allowed; a live deployed link is required. | The final demo day is scheduled for 26 Nov (Thursday). The date originally shifted from 4 Dec to 27 Nov (Friday) due to a college reschedule, and was finally corrected to 26 Nov because of a notice typo. The team cannot demo on localhost as a live deployed link is required. | Yes | Complete and accurate across all three clauses (`M028, M069, M079, M080, M082`). |
| 5 | Who worked on auth, how was it handed off, and is it completed? | Rohan originally took auth (`05/10/26`) and finished 60% on branch `auth-wip`. On `24/10/26` (overloaded with semester project and internship) he handed it off to Kabir in exchange for DB schema and tests. On `10/11/26` Kabir confirmed auth is done and merged. | Auth was originally assigned to Rohan, who completed about 60% of it on the branch auth-wip before handing it off to Kabir due to being overloaded with his semester project and internship. Kabir took over the task and ultimately completed and merged it. | Yes | Accurate handoff chain, branch name, reason, and merged status (`M017, M031, M052, M054, M085`). |
| 6 | Who owns the DB schema and what is its current status? | Rohan owns the DB schema after the `24/10/26` swap. After sharing schema v1, Rohan confirmed on `06/11/26` that schema v2 is ready with the `orders` table added. | Rohan owns the DB schema and its status is in progress (schema v2 ready with 'orders' table added, but the chat doesn't say whether DB schema finalization is fully completed). | Yes | Accurate owner, `v2` + `orders` table, and conservative partial flag (`M054, M055, M057, M075, M091`). |
| 7 | Who is responsible for tests, and did they finish them? | Rohan is responsible for tests (`24/10/26` swap). After missed ETAs (`kal raat tak`, `kal pakka`, `by this Sunday`), tests remained pending and the chat doesn't say whether they finished. | Rohan is responsible for tests. He promised to finish them by Sunday, but the chat doesn't say whether tests finished. | Yes | Correctly avoids treating broken ETAs as done (`partial=True`, `M054, M055, M076, M078, M084`). |
| 8 | Is there anything nobody is working on? | Yes, `search` has no owner. Kabir dropped it on `19/10/26`, Priya declined on `19/10/26` and `10/11/26`, and Aarav put deciding who takes search on the Saturday `14 Nov` meeting agenda. | The search feature is currently unassigned and nobody is working on it, as Kabir dropped it and asked someone else to take it, and it was deferred for discussion. The chat doesn't say whether anyone eventually took over the search feature. | Yes | Accurate global negative lookup (`M044, M087, M088, M089, M090, M091`). |
| 9 | What work has Priya finished and what is she currently working on? | Priya finished the listings UI and filters (`26/10/26`). She is currently working on the seller dashboard (started `26/10/26`, still on it `10/11/26`), and the chat doesn't say whether the seller dashboard is finished. | Priya has finished the listings UI and filters, and she is currently working on the seller dashboard. | Yes (minor omission) | Answers both asked clauses accurately without claiming seller dashboard is done (`M017, M023, M024, M025, M058, M089`), but sets `partial=False` and omits the explicit unconfirmed completion caveat for seller dashboard. |
| 10 | Which domain registrar did the team choose, how much did the domain cost, and who bought it? | Aarav bought `campuscart.in` from Namecheap for Rs 650 (`16/10/26`), choosing it over GoDaddy (Rs 800/year). | The team chose Namecheap as the registrar, where Aarav bought the domain (campuscart.in) for Rs 650. | Yes | Resolves `"wahi se leta hoon"` and `"bought it"` cleanly (`M038, M040, M041`). |
| 11 | What hosting platform was chosen for deployment, and which alternative was rejected and why? | Render was chosen for deployment (`26/10/26`). AWS was rejected because Meera warned its free tier can still produce unexpected bills. | Render was chosen as the hosting platform. AWS was rejected because its free tier can result in unexpected bills. | Yes | Accurate decision and rejection rationale (`M059, M060, M061`). |
| 12 | Did Aarav finish deploying on Render? | The chat doesn't say whether Aarav finished deploying on Render. On `09/11/26` he gave an ETA of `15 Nov`, and on `12/11/26` deployment status was still on the `14 Nov` meeting agenda. | The chat doesn't say whether Aarav finished deploying on Render. Aarav stated he was deploying on Render and would have it ready by 15 Nov, but the chat ended before that date. | Yes | Correctly abstains on completion while citing `15 Nov` ETA (`M083`). Automated check hit a false-positive substring match on `"aarav finished deploying"` inside `"doesn't say whether Aarav finished deploying"`. |
| 13 | Which logo option did the team finalize? | The team did not finalize a logo option (the chat doesn't say). Meera preferred the second (blue) option, while Aarav's `"obviously"` reply was ambiguous across interleaved messages. | Meera preferred the second option (blue logo), but the chat doesn't say whether the logo was officially finalized by the team. | Yes | Handles interleaved ambiguity without false confirmation (`M098, M099, M101, M102`). |
| 14 | What did Aarav's deleted message on 16 Oct say, what was decided at the Saturday 14 Nov meeting, and what is the project's GitHub repo URL? | `the chat doesn't say` | `the chat doesn't say` | Yes | Clean abstention (`supported=False`, `evidence=none`) across deleted, post-chat, and absent facts. |
| 15 | Kya localhost pe demo chalega aur chat feature kitna hua hai? | Nahi, localhost par demo nahi chalega: live deployed link chahiye (`09/11/26`). Chat feature Kabir kar raha hai aur `10/11/26` tak 70% hua hai, complete nahi hua hai. | Nahi, localhost pe demo nahi chalega kyunki sir ne live deployed link bola hai. Chat feature 70% ho gaya hai, lekin the chat doesn't say whether chat finished. | Yes (minor omission) | Answers both asked Hinglish clauses accurately with partial-completion guard (`M082, M085`), but omits Kabir's name as the owner of the 70% chat task. |

Overall: **14 / 15 correct** under strict question-answering scoring (scoring Q2 as a partial miss because it omitted the Meera to Kabir ownership swap, and noting minor omissions on Q9 and Q15; **12 / 15** if requiring every secondary detail). Verified citations: **15 / 15** answers passed all deterministic citation invariants with **0** fabricated or rejected `msg_id`s.

### Unseen stress-test results (generalization without overfitting)

To verify that nothing was hardcoded to `data/team_chat.txt`, the same pipeline was evaluated (`rebuild=True`) on two unseen synthetic chat exports (`eval/unseen_eval.json`, 10 hand-worked questions):

1. `tests/fixtures/unseen_ios_24h_chat.txt` (102 messages across 114 lines in iOS 24h bracket format `[DD/MM/YYYY, HH:MM:SS]`, FestPass kiosk project):
   - Compiled state had 0 invariant errors, capturing the triple reversal on `badge_printing_sdk` (`escpos-bt` -> `usb-spooler` -> `WiFi CUPS print server`), the confirmed sticker decision, 6 tasks (`volunteer_roster_sync` unassigned, `check_in_analytics_dashboard` unconfirmed), and 2 deadlines (`21/09/2026` opening ceremony and `17 Sep 2026` dry run).
   - On questions `U1` to `U5`, all 5 answers were factually accurate with 5 / 5 structural invariants passing and 0 rejected `msg_id`s, including clean abstention on `U5` (deleted WiFi password and post-chat planning call). Two secondary-detail omissions matched the pattern seen in Q15: `U2` omitted the `"4 days before"` rule wording, and `U3` omitted previous owner Ananya.

2. `tests/fixtures/unseen_noisy_large_chat.txt` (320 messages across 576 lines spanning 5 weeks of noisy banter, memes, and off-topic chatter, RoverX robotics build):
   - Compiled state had 0 invariant errors, correctly filtering out noise while capturing the `motor_driver_selection` triple reversal (`L298N` -> `BTS7960 43A H-bridge` -> `VESC 6.0 CAN controllers`), the `telemetry_radio_link` / `sensor_mast_cad` swap, 9 tasks (`soil_moisture_probe` unassigned, `autonomous_waypoint_nav` at 65%), and 5 deadlines (`04/09/26` qualifiers, `30 Aug` safety inspection video).
   - On questions `U6` to `U10`, all 5 answers were factually accurate with 5 / 5 structural invariants passing and 0 rejected `msg_id`s, including clean abstention on `U10` (deleted competition portal PIN and post-chat lab review), with one secondary-detail omission on `U8` (omitted the `"5 days before"` rule phrasing).

## 4. What it still gets wrong, and why

1. **Cross-window task key splitting (`report_ppt` vs `report` + `ppt`):**
   At `M017` (`05/10/26`, `"Meera - payments + report/PPT"`), the extractor split at `+` and kept `report/PPT` as a single task key `report_ppt`. Later at `M068` (`02/11/26`, `"I'll take report + PPT fully"`), `_EXTRACTION_SYSTEM_PROMPT` rule 6 required splitting `"A + B"` into separate atomic tasks, producing `report` and `ppt`. Because `resolve_canonical_key` in `src/compiler.py` only merges exact normalized token sets or single-token prefixes, `report` and `ppt` did not link back to `report_ppt`, leaving three rows (`report_ppt`, `report`, `ppt`) for Meera in `ProjectState.tasks`.

2. **Disconnected `Decisions` and `Tasks` histories in Q&A context (root cause of Q2's miss):**
   In `src/schemas.py` and `src/qa.py`, `DecisionRecord` (`topic_key="payments"`) tracks technology changes without an owner field, while `TaskRecord` (`task_key="payments"`) tracks ownership changes (`Meera` -> `Kabir`). `format_state_for_qa` prints `Decisions:` and `Tasks:` in separate blocks. When Q2 asked what the original decision about payments was and how it changed, retrieval emphasized the decision-change cluster (`M047` to `M066`) and the Q&A prompt answered from `Decisions -> payments`, missing `M017`/`M020` (Meera) and `M067` (Kabir taking ownership) under `Tasks -> payments`.

3. **Occasional omission of secondary clauses or rules in concise answers:**
   In Q9 (omitting the explicit unconfirmed completion caveat for seller dashboard), Q15 (omitting Kabir's name on the 70% chat task), U2/U8 (omitting the `"4 days before"` / `"5 days before"` rule phrasing), and U3 (omitting previous owner Ananya), `DraftAnswer` answered every explicit question target correctly but dropped linked metadata from `ProjectState` or earlier handoff entries when summarizing concisely.

4. **Heuristic phrase regexes in `src/verifier.py` and sequential window latency on large chats:**
   Guard 3 (`_detect_completion_overclaim` in `src/verifier.py`) depends on curated English and Hinglish regex lists (`_COMPLETION_PHRASE_RE`, `_UNCERTAINTY_OR_NEGATION_RE`, `_PARTIAL_PERCENT_RE`) and clause splitting (`_CLAUSE_SPLIT_RE`). Unseen Hinglish completion expressions outside those patterns would bypass the deterministic completion guard. Separately, `compile_project_state_with_meta` (`src/compiler.py`) processes windows sequentially so each window sees the updated `_CanonicalRegistry`. On a 576-line, 320-message chat (`17` windows at `window_size=25`, `overlap=6`), compilation requires `17` sequential LLM calls before embedding and Q&A, taking multiple minutes when `LLM_REQUEST_DELAY_SECONDS` pacing is enabled for free-tier RPM limits.

## 5. What I'd do with 2 more days

1. **Priority 1: Cross-entity state linking and post-compilation key consolidation.**
   Add a deterministic post-reduction reconciliation pass in `src/compiler.py` that links composite keys (`report_ppt`) to split sub-keys (`report`, `ppt`) sharing the same owner or message anchors, and cross-links `DecisionRecord` and `TaskRecord` entries that share keys or overlapping `evidence_msg_ids` inside `format_state_for_qa`. That directly eliminates duplicate task rows like `report_ppt` and ensures decision-timeline answers (like Q2) always surface coupled ownership handoffs.

2. **Priority 2: Time-gap session clustering plus Hinglish lexical normalization and query expansion.**
   In `src/retriever.py`, group chunks along natural conversation gaps (instead of fixed 6-message strides) so multi-turn debates stay whole without dragging unrelated earlier banter into the window, add a Hinglish normalization table (`"kal"`, `"bacha"`, `"baaki"`, `"karunga"`, `"nahi"`) to `tokenize_text`, and expand queries with active participant and task aliases from `ProjectState` before BM25 ranking.

3. **Priority 3: Rate-aware parallel window extraction and rubric-based semantic evaluation.**
   Replace purely sequential window extraction with bounded concurrent batches under a shared token-bucket rate limiter, followed by an embedding-similarity key-merge pass across batches so 500+ line chats compile in a fraction of the time without dropping key consistency. In `eval/run_eval.py`, replace literal substring hint checks with semantic or clause-level checks so valid phrasing (`"November 23"` vs `"23 Nov"`) and negated strings (`"doesn't say whether Aarav finished deploying"`) do not create noisy diagnostic warnings.

## 6. What I learned that I didn't know on day one

1. **Why naive chunked vector RAG fails on temporal reversals and global negative queries:**
   Before reading `data/team_chat.txt`, standard chunked retrieval looked sufficient. Seeing `"agreed. final decision: no Razorpay, UPI QR manual"` (`20/10/26`) and `"Demo day 4 Dec so report 1 Dec"` (`12/10/26`) sit verbatim in earlier chunks made clear why similarity search ranks outdated commitments above short later reversals (`"QR wala scrap"`, `"26 Nov final"`), and why negative global questions (`"Is there anything nobody is working on?"`) require an explicit event-sourced `ProjectState` built chronologically.

2. **Why letting an LLM re-type timestamps or line IDs causes drift, and deterministic ID lookup fixes it:**
   Asking a model to copy timestamps, line numbers, and quotes directly invites subtle mismatches (`M086` drifting to `M091`, or misquoted line spans). Restricting `DraftAnswer` and delta schemas to cite only short `msg_id` strings (`"M086"`) and resolving `line_start`, `line_end`, `timestamp_raw`, `sender`, and `quote` from `ParsedChat` in Python guarantees authentic citations and turns verification into deterministic validation.

3. **Real WhatsApp export quirks (invisible Unicode characters and `DD/MM` vs `MM/DD` ambiguity):**
   Real exports embed narrow no-break spaces (`\u202f`) before `AM`/`PM`, left-to-right marks (`\u200e`), unheaded continuation lines (`lines 18 to 22`), and ambiguous numeric dates like `05/10/26` that cannot be disambiguated on one line alone. Building a file-wide date-order scorer (`day > 12` plus chronological inversion checks) alongside Unicode whitespace normalization prevented silent timestamp bugs.

4. **Live API quirks require early probing and defensive rate-limit handling:**
   Official SDK defaults, request timeouts, and free-tier RPM/TPM limits vary across environments. Setting explicit HTTP timeouts on the Gemini client (`HttpOptions`), parsing `Retry-After` headers and error `retryDelay` hints, and adding configurable `LLM_REQUEST_DELAY_SECONDS` kept live evaluation runs predictable and repeatable.

5. **Safe DOM rendering matters even for benign chat placeholders:**
   WhatsApp literally writes `<Media omitted>` and `<This message was edited>` with angle brackets. Inserting those strings via `innerHTML` makes the browser swallow them as unrecognized HTML tags. Constructing every element in `src/web/static/app.js` with `document.createElement` and `.textContent` keeps literal placeholders visible and prevents HTML injection.

## 7. What I understand deeply vs. what I accepted from AI

### Deeply owned and explainable end-to-end

- **Problem decomposition and pipeline design (`docs/DECOMPOSITION.md`):** Separating raw chat parsing, chronological event-sourced state compilation, hybrid retrieval with neighbor expansion, and deterministic verification before building UI layers.
- **Multi-format WhatsApp parser (`src/parser.py`):** Invisible Unicode translation table, Android/iOS header patterns, multi-line span merging, `<Media omitted>` / deleted / edited classification, and file-wide `DD/MM` vs `MM/DD` monotonicity scoring.
- **Temporal state schema and deterministic reducer (`src/schemas.py`, `src/compiler.py`):** Delta types (`DecisionDelta`, `TaskDelta`, `DeadlineDelta`, `OpenIssueDelta`), first-person pronoun resolution (`"Main"` -> sender), canonical key alias matching, overlapping-window deduplication, and chronological reduction into `ProjectState`.
- **Hybrid retriever (`src/retriever.py`):** Compound-token BM25 tokenization, dense vector batching and `.npz` caching, Reciprocal Rank Fusion (`RRF_K = 60`), `+/- 3` context span merging, and automatic BM25 fallback when embeddings fail.
- **Grounded Q&A and deterministic verifier (`src/qa.py`, `src/verifier.py`):** Prompting with `ProjectState` + `history[]` + expanded raw blocks, resolving authentic `EvidenceRef` records, blocking empty/deleted-only citations and completion overclaims on incomplete tasks, and enforcing `"the chat doesn't say"` abstention after at most one corrective retry.
- **Evaluation methodology (`eval/team_chat_eval.json`, `eval/unseen_eval.json`, `eval/run_eval.py`):** Working out all 15 `team_chat.txt` ground-truth answers and 10 unseen answers by hand, running invariant checks, and auditing outputs without hiding partial misses.

### Accepted from AI assistance (reviewed and tested)

- **Frontend CSS layout and native `<dialog>` accessibility boilerplate (`src/web/static/styles.css`, `src/web/static/index.html`, `src/web/static/app.js`):** Responsive `@container (width < 560px)` card-table rules, `closedby="any"` native modal dialog with `pointerdown`/`pointerup` outside-click fallback in `setupDialogFallbacks`, and ARIA focus styling, verified via `tests/test_web.py` and live FastAPI checks.
- **Structured JSON recovery helpers (`src/llm.py`):** Balanced-brace scanning (`_extract_balanced_json`), code-fence stripping, and trailing-comma cleanup (`_build_json_candidates`), verified via `tests/test_llm.py`.
- **Verifier regex phrase lists and synthetic fixture banter (`src/verifier.py`, `tests/fixtures/`):** Initial English/Hinglish completion and negation regex vocabularies (`_COMPLETION_PHRASE_RE`, `_UNCERTAINTY_OR_NEGATION_RE`) and background conversational filler in `unseen_ios_24h_chat.txt` and `unseen_noisy_large_chat.txt`. I audited the synthetic chats line by line and wrote/verified every test expectation and ground-truth citation myself.

## 8. Demo video

Link to your 3-minute screen recording: pending (to be added once recorded)
