# Decomposition (fill this in BEFORE writing code)

## 1. The problem in my own words

I need to build a tool that takes an exported WhatsApp group chat text file and figures out what the team actually decided, who currently owns each piece of work, when deadlines are, and what is still unfinished or unassigned. Instead of treating the chat like a static document where every message is equally true, the tool has to read the messages in order so later updates override earlier plans. Users should be able to run one command on any chat file in this format to see a clean summary of the current project state and ask questions in English or Hinglish. Every answer must point back to the exact messages (timestamp, sender, or line) that support it, and say "the chat doesn't say" whenever the chat never settled or answered the question.

## 2. What I noticed in the data

Read `data/team_chat.txt` completely. Below are 7 specific challenges in the chat that would break a naive system:

| # | What I noticed (quote + timestamp) | Why it makes answering hard |
|---|---|---|
| 1 | **Triple reversal on payments (tech and ownership):** `05/10/26, 8:02 PM` (Aarav): `"Meera - payments + report/PPT"` and `05/10/26, 8:10 PM` (Meera): `"payments Razorpay se karungi"`. Then `20/10/26, 8:10 PM` (Meera): `"Razorpay ko business KYC chahiye, humare paas nahi hai"` and `20/10/26, 8:16 PM` (Aarav): `"agreed. final decision: no Razorpay, UPI QR manual"`. Then `02/11/26, 8:45 PM` (Kabir): `"Razorpay test mode bina KYC ke chalta hai!"`, `02/11/26, 8:49 PM` (Aarav): `"haan. Razorpay test mode final. QR wala scrap."`, `02/11/26, 8:50 PM` (Kabir): `"payments ab mere paas, Meera tum report pe focus karo"`, and `10/11/26, 9:13 PM` (Kabir): `"payments webhook bacha hai"`. | Even the second decision literally includes the phrase `"final decision"`, so keyword or similarity matching surfaces conflicting answers (UPI QR vs Razorpay), names the wrong owner (Meera instead of Kabir), or misses that the webhook is still unfinished. |
| 2 | **Cascading deadline changes and derived dates:** `12/10/26, 7:30 PM` (Meera): `"Reminder: report due 3 days before demo day. Demo day 4 Dec so report 1 Dec"`. Then `05/11/26, 6:10 PM` (Aarav): `"college ne demo day prepone kar diya. ab 27 Nov (Friday)"` and `6:13 PM` (Meera): `"toh report ka deadline bhi change. 24 Nov?"`. Then `09/11/26, 5:30 PM` (Priya): `"Correction. Sir se baat hui. Demo day 26 Nov (Thursday), 27 nahi. Notice mein typo tha"`, `5:31 PM` (Aarav): `"ohh. ok so 26 Nov final"`, `5:32 PM` (Meera): `"toh report 23 Nov"`, and `5:33 PM` (Priya): `"sir ne bola demo mein live deployed link chahiye, localhost allowed nahi"`. | The report due date is defined by a dependency rule (`3 days before demo day`) and shifts twice (`1 Dec` to `24 Nov?` to `23 Nov`). Retrieving only the explicit `"report due"` message from `12/10/26` or stopping at the `27 Nov` typo gives the wrong date. |
| 3 | **Multi-line messages without repeated timestamps and Hinglish first-person pronoun:** `05/10/26, 8:02 PM` (Aarav): `"Task split karte hain:"` followed by 5 lines with no timestamps: `"Rohan - auth (login/signup)"`, `"Priya - listings UI"`, `"Meera - payments + report/PPT"`, `"Kabir - search + chat between buyer/seller"`, and `"Main - deployment"`. Later on `19/10/26, 6:16 PM`, Kabir writes `"chat main kar lunga"`. | A naive line-by-line parser drops lines 18 to 22 because they have no timestamp prefix, losing the initial task assignments. Even after parsing, `"Main"` is Hinglish for "I" (Aarav on `05/10/26`, Kabir on `19/10/26`), not a person's name, so it only resolves using the message sender. |
| 4 | **Partial task handoffs and unassigned work:** `24/10/26, 11:20 PM` (Rohan): `"auth Kabir le sakta hai? auth ka 60% ho chuka hai, branch auth-wip dekh lena"`, `11:25 PM` (Kabir): `"arre theek hai, auth bhi le leta hoon. but Rohan you do the DB schema and the tests then"`, `11:26 PM` (Rohan): `"deal. schema + tests mere"`, and `10/11/26, 9:12 PM` (Kabir): `"auth done, merged. chat 70%"`. Meanwhile, `search` was dropped on `19/10/26, 6:16 PM` (Kabir): `"search koi aur le lo please"`, still open on `10/11/26, 9:14 PM` (Priya): `"search ka kya? abhi bhi koi nahi le raha"`, and pushed to the Saturday meet agenda on `12/11/26, 8:10 PM` (Aarav): `"ok agenda: search kaun karega + deployment status"`. | Auth is not a clean single-owner task: Rohan wrote 60% on `auth-wip`, Kabir finished and merged it, and Rohan swapped into DB schema and tests. For `search`, a system cannot just check if a task was ever assigned (it was assigned to Kabir on `05/10/26`); it has to track that it was dropped and never re-claimed. |
| 5 | **Sarcasm, broken ETAs, and unconfirmed completions before the log ends:** `12/10/26, 7:35 PM` (Rohan): `"auth kal tak khatam. promise"`, `12/10/26, 7:36 PM` (Priya): `"Sure Rohan, \"kal tak\""` and `"you said that last week too"`. Later, `06/11/26, 11:06 PM` (Rohan): `"tests kal raat tak kar dunga"`, `07/11/26, 11:50 PM` (Rohan): `"sorry yaar, tests nahi ho paye. kal pakka"`, `09/11/26, 5:40 PM` (Rohan): `"tests ab bhi pending hain, I'll do by this Sunday"`. Also `09/11/26, 5:34 PM` (Aarav): `"Render pe deploy kar raha hoon, 15 Nov tak ready"`. | Confident promises (`"promise"`, `"kal pakka"`, `"by this Sunday"`, `"15 Nov tak ready"`) do not mean the work got done, and Priya's `"Sure Rohan, \"kal tak\""` is sarcasm, not approval. Since the chat ends on `12/11/26`, deployment, tests, chat (`70%`), webhook, seller dashboard, and report/PPT are all unconfirmed rather than completed. |
| 6 | **Media placeholders, edited markers, and deleted messages:** `09/10/26, 10:41 PM` (Priya): `"<Media omitted>"` explained at `10:42 PM` (`"ye screenshot hai, filters abhi nahi hain"`). `16/10/26, 9:51 PM` (Aarav): `"<Media omitted>"` explained at `9:50 PM` (`"bought it. receipt bhej raha hoon"`). `24/10/26, 11:27 PM` (Rohan): `"<Media omitted>"` explained at `11:27 PM` (`"ye schema ka diagram, v1"`). Also `16/10/26, 9:07 PM` (Meera): `"Rs 800 theek hai. 5 log hain, split kar lenge <This message was edited>"` and `16/10/26, 9:55 PM` (Aarav): `"This message was deleted"`. | A `<Media omitted>` line carries zero information on its own and only makes sense alongside adjacent lines. The edited marker must be stripped cleanly as metadata, while a deleted message is permanently missing content and must trigger abstention if asked about. |
| 7 | **Informal/ambiguous consensus, interleaved replies, and Hinglish anaphora:** `12/11/26, 8:11 PM` (Priya): `"logo ke liye 2 options bheje hain, dekho"` + `"<Media omitted>"`, `8:12 PM` (Rohan): `"Aarav you're also coming right?"`, `8:12 PM` (Meera): `"second wala better lag raha, blue wala"`, `8:13 PM` (Aarav): `"obviously"` (trailing laughing reaction omitted). Earlier on `16/10/26`, `9:05 PM` (Aarav): `"GoDaddy pe campuscart.in Rs 800/year dikha raha"`, `9:20 PM` (Kabir): `"Namecheap pe check karo, I think cheaper"`, `9:41 PM` (Aarav): `"haan Namecheap pe Rs 650. wahi se leta hoon"`, `9:50 PM` (Aarav): `"bought it. receipt bhej raha hoon"`. | Aarav's `12/11/26, 8:13 PM` reply (`"obviously"`) is genuinely ambiguous because of the interleaved `8:12 PM` messages: it could answer Rohan's question about attending the Saturday library meet, or playfully respond to Meera's blue logo comment, so either way the logo choice is not formally confirmed. Meanwhile, `"wahi se leta hoon"` ("I'll buy from there") and `"bought it"` require resolving back across earlier lines to Namecheap and `campuscart.in`. |

## 3. The simplest thing that could work, and how it would fail

The simplest baseline is naive chunked vector RAG (or keyword search + LLM): split the chat into fixed chunks of 10 to 15 lines, embed each chunk (or index keywords), retrieve the top-k chunks most similar to the user's question, and prompt an LLM to answer using only those chunks.

Three exact questions on `data/team_chat.txt` that this baseline would fail:

1. **"How are payments being handled right now, who owns them, and is the feature done?"**
   Top-k similarity pulls the most explicit payment discussions: `05/10/26, 8:10 PM` (Meera saying `"payments Razorpay se karungi"`) and `20/10/26, 8:16 PM` (Aarav writing `"agreed. final decision: no Razorpay, UPI QR manual"`), and may miss the short `10/11/26, 9:13 PM` line (`"payments webhook bacha hai"`) sitting in a separate chunk. Without chronological state overrides, the LLM gets misled by the phrase `"final decision"` on `20/10/26` or Kabir's `"checkout chal raha hai"` on `02/11/26`, reporting the wrong payment method (UPI QR), the wrong owner (Meera), or claiming payments is completely done.

2. **"When is the project report due?"**
   The `12/10/26, 7:30 PM` message (`"Reminder: report due 3 days before demo day. Demo day 4 Dec so report 1 Dec"`) contains both `"report due"` and `"demo day"` in full, so it scores highest in vector and keyword search. Later updates are split across short lines on `05/11/26` (`"27 Nov"`, `"24 Nov?"`) and `09/11/26` (`"Demo day 26 Nov (Thursday), 27 nahi"`, `"toh report 23 Nov"`). Naive RAG either returns `1 Dec`, picks the superseded `24 Nov` typo date, or misses the dependency between Priya's `26 Nov` correction and Meera's `23 Nov` line.

3. **"Is there anything nobody is working on?"**
   This is a negative global aggregation query across all 5 weeks, and no single chunk contains a summary list of unclaimed work. Top-k retrieval for `"working on"` or `"task"` retrieves the `05/10/26, 8:02 PM` task list where `search` WAS assigned (`"Kabir - search + chat between buyer/seller"`), or misses the chain linking Kabir dropping it (`19/10/26`), Priya declining it (`10/11/26`), and Aarav putting `"search kaun karega"` on the `12/11/26` meeting agenda. Naive RAG also tends to hallucinate completions on unconfirmed items like `"15 Nov tak ready"` for Render deployment or pick the blue logo from Aarav's `"obviously"`, rather than stating what is open or unconfirmed as of `12/11/26`.

## 4. Break it into sub-problems

```text
  chat.txt
     |
     v
+-----------------------------+
|   1. Multi-format parser    |  (Android/iOS, multi-line merge, line spans)
+-----------------------------+
     |
     +--------------------------------------------+
     | normalized messages                        |
     v                                            v
+-------------------------------------+    +-------------------------------------+
| 2. Temporal state compiler          |    | 3. Hybrid retriever                 |
|    (chronological sliding windows   |    |    (BM25 + API embeddings + RRF     |
|     -> typed deltas -> reducer)     |    |     + +/- N line context expansion) |
+-------------------------------------+    +-------------------------------------+
     |                                            |
     | ProjectState (current state                | expanded evidence windows
     | + full revision history)                   |
     +----------------------+---------------------+
                            |
                            v
     +--------------------------------------------+
     | 4. Grounded Q&A + deterministic verifier   |
     |    (citation check + abstention guard:     |
     |     "the chat doesn't say")                |
     +--------------------------------------------+
                            |
                            v
     +--------------------------------------------+
     | 5. CLI (Rich / --json) and optional --web  |
     +--------------------------------------------+

  Shared layer: [ 6. Provider adapter (.env: Gemini default | OpenAI | Ollama) ]
                serves LLM calls in (2) & (4) and API embeddings in (3).
```

### 4.1 Multi-format parser
- **Input:** Path to an exported WhatsApp chat `.txt` file.
- **Output:** Chronological list of normalized message objects (`msg_id`, `line_start`, `line_end`, `timestamp_raw`, `timestamp_iso`, `sender`, `text`, `kind` in `chat | system | media_omitted | deleted`, `is_edited`), plus file-level date format metadata.
- **How I'll know it's working:** Run parser checks on `data/team_chat.txt` and synthetic Android/iOS/24h/`MM/DD/YY` fixtures; confirm all 107 lines of `data/team_chat.txt` are accounted for, lines 18 to 22 merge into Aarav's `05/10/26, 8:02 PM` message, `<Media omitted>` / `<This message was edited>` / `This message was deleted` are tagged cleanly, and zero non-empty lines are dropped.

### 4.2 Temporal state compiler and reducer
- **Input:** Normalized messages processed sequentially in overlapping chronological windows, along with the current list of canonical topic/task keys from prior windows.
- **Output:** An append-only stream of typed state deltas (`decision_proposed`, `decision_confirmed`, `decision_reversed`, `task_assigned`, `task_reassigned`, `task_unassigned`, `task_progress`, `deadline_set`, `deadline_changed`, `dependency_rule`, `open_question`) merged by a deterministic Python reducer into a persistent `ProjectState` JSON structure (current decisions, task owners/statuses, unassigned/open items, deadlines, and full revision history with exact message citations).
- **How I'll know it's working:** Run `--state-only` on `data/team_chat.txt` and inspect the output JSON: payments shows Kabir owning Razorpay test mode with webhook remaining (and revision history Meera -> UPI QR -> Kabir), report due date shows `23 Nov` derived from `26 Nov` minus 3 days, `search` is listed as unassigned/open, and every entry links back to valid message IDs and timestamps.

### 4.3 Hybrid retriever with context expansion
- **Input:** User question, normalized messages, and compiled `ProjectState`.
- **Output:** Top-ranked evidence windows built by fusing BM25 scores and API embedding similarity via Reciprocal Rank Fusion (RRF), then expanding each hit by +/- N adjacent messages so `<Media omitted>` captions, Hinglish anaphora (`"wahi se leta hoon"`), and interleaved replies stay together.
- **How I'll know it's working:** Query `"receipt"`, `"schema"`, `"domain"`, and `"logo"` against `data/team_chat.txt` in retrieval-only checks and verify the expanded windows contain both the placeholder or short line and the surrounding lines that explain it (e.g., `16/10/26, 9:41 PM` Namecheap + `9:50 PM` receipt + `9:51 PM` `<Media omitted>`).

### 4.4 Grounded Q&A and deterministic citation/abstention verifier
- **Input:** User question, compiled `ProjectState`, and expanded evidence windows.
- **Output:** Answer in English or Hinglish with exact source citations (`timestamp`, `sender`, `line_start-line_end`), validated by deterministic verification code. If cited timestamps/senders/message IDs do not match actual parsed messages, if claims lack valid citations, or if the evidence shows the topic was never covered, deleted, or unconfirmed before the chat ended, the verifier blocks guessing and returns `"the chat doesn't say"` (or states only the verified partial status up to the last message).
- **How I'll know it's working:** Run tests with (a) real answerable questions on `data/team_chat.txt`, (b) mock LLM responses containing fabricated timestamps or invalid line ranges (verifier must reject them), and (c) unanswerable prompts like `"What did Aarav delete on 16 Oct?"` or `"Did Aarav's Render deployment go live?"` (verifier must return `"the chat doesn't say"` rather than guessing).

### 4.5 CLI and optional web interface
- **Input:** CLI arguments (`python run.py --chat path/to/chat.txt`, `--question "..."`, `--state-only`, `--json`, `--web`) and interactive user questions.
- **Output:** Rich terminal views of project state and cited answers, clean `--json` output for scripting and testing, or a FastAPI single-page UI under `--web` displaying current decisions, task board, revision timeline, and Q&A with evidence references.
- **How I'll know it's working:** Run `python run.py --chat data/team_chat.txt --state-only`, `--question`, `--json`, and `--web` end-to-end on both `data/team_chat.txt` and an unseen synthetic chat file without errors.

### 4.6 Provider-agnostic inference adapter
- **Input:** Structured JSON prompts or Q&A prompts, embedding batches, and `.env` configuration (`LLM_PROVIDER`, API key, model names, or Ollama base URL).
- **Output:** Validated structured outputs and embedding vectors from Gemini API (default), OpenAI, or Ollama through one common interface, keeping dependencies lightweight (no PyTorch or heavy compiled ML stacks on Python 3.13/3.14).
- **How I'll know it's working:** Run adapter unit tests (mocked responses + schema error handling) and a live smoke check against Gemini from `.env` to confirm structured JSON extraction and embeddings work cleanly.

## 5. What I don't know yet

1. **Unseen WhatsApp date/locale formats and invisible Unicode characters:**
   Exports vary across Android (`DD/MM/YY, 9:14 PM - Name:`), iOS (`[DD/MM/YY, 21:14:00] Name:`), 12h vs 24h clocks, 4-digit years, and `MM/DD/YY` vs `DD/MM/YY` ordering, plus invisible markers like `\u202f` (narrow no-break space before AM/PM) or `\u200e` (LTR mark). A line like `05/10/26` is ambiguous alone unless checked against the rest of the file (e.g., seeing `13/10/26` or monotonic jumps across the timeline).
   - *How I'll find out:* Review WhatsApp export format specs, build a file-wide date-order heuristic (`day > 12` and chronological monotonicity checks) with Unicode whitespace normalization, and test against synthetic Android, iOS, 24h, and `MM/DD/YY` exports.

2. **Merging synonymous entity and task names across extraction windows:**
   Window 1 might label a task `"auth (login/signup)"`, window 2 `"authentication"`, and window 3 `"DB schema"` vs `"schema v2"`. If keys do not match across windows, the reducer will fail to link handoffs and reversals to the same item.
   - *How I'll find out:* Pass active canonical task/decision keys from earlier windows into each window's extraction prompt, add normalized alias matching in the reducer, and inspect `--state-only` output on `data/team_chat.txt` to confirm tasks like `payments`, `auth`, `search`, `DB schema`, and `report/PPT` merge cleanly.

3. **Hinglish lexical mismatch in BM25 retrieval:**
   BM25 relies on exact tokens, so an English query like `"Who finished authentication?"` or `"What is still pending?"` misses Hinglish lines like `"auth done"` vs `"bacha hai"` vs `"abhi bhi koi nahi le raha"` or `"wahi se leta hoon"`.
   - *How I'll find out:* Run retrieval ablations comparing BM25 alone, API embeddings alone, and hybrid RRF on a small set of English and Hinglish queries over `data/team_chat.txt`, and measure how much passing the compiled `ProjectState` directly to Q&A closes lexical gaps.

4. **API rate limits and structured JSON consistency across providers:**
   Free-tier Gemini limits requests per minute, and smaller Ollama models sometimes break strict JSON schemas or wrap responses in markdown fences. Extracting one message at a time would quickly hit rate limits.
   - *How I'll find out:* Check Gemini (`google-genai`), OpenAI, and Ollama structured-output docs, tune chronological window sizes (e.g., 20 to 30 messages with 5-message overlap so a 100-line chat takes ~4 to 6 extraction calls), and add retry/backoff plus strict schema parsing with repair checks.

5. **Setting deterministic verifier thresholds without false abstentions:**
   If the citation verifier requires exact English word overlaps against Hinglish chat lines, valid bilingual answers could be wrongly rejected as `"the chat doesn't say"`. If it is too loose, the LLM might guess past `12/11/26`.
   - *How I'll find out:* Base deterministic checks on structural anchors (valid message IDs, exact timestamps, exact sender names, line span bounds, and explicit state flags like `deleted` or `unconfirmed`) rather than literal string matching across languages, then validate on edge-case test questions.

## 6. Build order

1. **Slice 1: Multi-format parser + minimal end-to-end skeleton (`run.py` + provider adapter)**
   Parse any WhatsApp `.txt` file into normalized messages with line numbers and send them through a minimal `run.py` call to answer one question with basic citations.
   *Why first:* Proves the end-to-end loop, `.env` credentials, and Python environment work immediately, while locking down accurate message IDs and line numbers before adding state or retrieval logic.

2. **Slice 2: Temporal state compiler and reducer (`--state-only`)**
   Implement windowed extraction into typed deltas and the deterministic reducer that produces `ProjectState` with current state and revision history.
   *Why second:* Temporal reversals, derived deadlines, and dropped tasks are the hardest part of this problem. Building and inspecting `--state-only` early ensures the state model is accurate before mixing it with retrieval.

3. **Slice 3: Hybrid retriever with context expansion + deterministic verifier**
   Add BM25 + API embedding hybrid search (RRF + +/- N line expansion), connect `ProjectState` + expanded evidence to grounded Q&A, and enforce deterministic citation checking and `"the chat doesn't say"` abstention.
   *Why third:* Gives Q&A both global state awareness and fine-grained local evidence (including `<Media omitted>` neighbors), while adding strict trustworthiness guarantees before building user-facing views.

4. **Slice 4: CLI polish (`Rich`, `--json`) and optional `--web` FastAPI UI**
   Add readable Rich terminal formatting, `--json` output, and the optional single-page FastAPI interface under `--web`.
   *Why fourth:* Keeping presentation separate until the core data and verification outputs stabilize avoids rewriting UI templates while state fields change.

5. **Slice 5: Hand-scored evaluation (10+ questions) and unseen stress tests**
   Write and manually grade at least 10 tough questions on `data/team_chat.txt`, plus run synthetic unseen chats (different topics, `MM/DD/YY`, iOS exports) to test generalization and document failures honestly in `docs/FINAL_NOTES.md`.
   *Why fifth:* Unit tests run alongside slices 1 to 4, while final end-to-end scoring happens once all components are integrated so failure analysis reflects the real system.

## 7. Where I'll use AI, and where I won't

- **What I'll design and own myself:**
  - **The `ProjectState` schema and delta taxonomy:** Deciding which delta types exist (`decision_reversed`, `task_reassigned`, `task_unassigned`, `deadline_changed`, `dependency_rule`) and what counts as `unconfirmed` vs `done`, based directly on the patterns in `data/team_chat.txt`.
  - **Deterministic reduction rules:** Writing the Python state-merging logic that preserves chronological order, builds revision history, and propagates deadline changes without letting an LLM overwrite state unpredictably.
  - **Citation and abstention verification logic:** Implementing deterministic checks that verify every cited timestamp, sender, and line number actually exists and force `"the chat doesn't say"` on deleted/unconfirmed/missing claims, rather than trusting the LLM to police itself.
  - **Manual test set and honest grading:** Writing the 10+ evaluation questions and working out ground-truth answers by hand from `data/team_chat.txt`, so the evaluation stays independent of the models generating the answers.

- **Where I'll use AI assistance (and audit afterward):**
  - **WhatsApp parsing regex boilerplate:** Drafting date/time regex patterns for Android, iOS, and Unicode edge cases, then auditing them with unit tests against synthetic chat fixtures and `data/team_chat.txt`.
  - **CLI and web UI scaffolding:** Generating Rich display boilerplate and the HTML/CSS/FastAPI single-page layout for `--web`, then reviewing it to ensure it strictly renders verified state and citation fields.
  - **Synthetic unseen chat generation:** Creating extra realistic test chats (different teams, formats, and reversals) to check generalization, while manually verifying each synthetic file's expected answers.
  - **Prompt refinement:** Iterating on delta extraction prompts when inspecting `--state-only` errors, keeping changes grounded in measurable pass/fail checks.
