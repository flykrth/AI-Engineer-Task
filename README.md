# Take-Home Task: Make Sense of the Team Chat


---

## The problem

A college project team has a WhatsApp group. For five weeks they argued, changed their minds, gave each other tasks, missed deadlines, made decisions, reversed them, and joked around.

Nobody on the team can say what the **current** state of their project is. Not who owns what, not what was decided, not even when the deadline is.

**Your job: build a system that reads the chat and answers questions about it.**

The chat is in `data/team_chat.txt`. It's a real-format WhatsApp export, in a mix of English and Hinglish. **Read it yourself before you design anything.** All of it. It's short.

Someone should be able to ask things like:

- "What's the current status of X?"
- "Who is responsible for Y, and did they finish?"
- "What did we decide about Z, and has that changed?"
- "Is there anything nobody is working on?"

and get a correct, trustworthy answer.

### What your system must do

1. **Run on any chat file in this format, not just ours.** We will run it on a different chat during your interview. Hardcoding facts from this chat will fail. Give us one command, for example `python run.py --chat path/to/chat.txt`. A web app, notebook or CLI is all fine. We care that it runs.
2. **Answer questions in English or Hinglish, and show its evidence.** Each answer should point back to the message(s) it is based on (timestamp, sender, or line).
3. **Say "the chat doesn't say" when it doesn't.** A confident wrong answer is worse than no answer.
4. **Produce a "state of the project" view.** It should list current decisions, who owns what, and what is still open or unassigned. The format is up to you.

That's the whole spec. We have deliberately not told you how to do it. Figuring that out is the task.



## How to work

### Step 0: Decompose before you code

Copy `docs/DECOMPOSITION.md` and fill it in **before writing any code**. Commit it on its own, first, so the git history shows the order.

This is the most important document in your submission. Spend real time on it.

### Step 1: Build in small slices

Get the smallest end-to-end version working first (chat in, one answer out), then improve it. Commit often. 

### Step 2: Test yourself honestly

Write **at least 10 questions** about this chat, with the correct answers worked out by you, by hand. Run your system on them and score it. Don't hide the failures. We trust a candidate who finds their own bugs more than one who claims everything works.

### Step 3: Wrap up

- Fill in `docs/FINAL_NOTES.md`.
- Upload to a Repo

---

## Rules and practical notes


- **No secrets in the repo.** Keep API keys in a `.env` file (already git-ignored) and tell us what we need to set.
- **Do your own work.** Don't share this task or your solution publicly before deadline / submission date .

---

## Submission checklist

- [x] Public GitHub repo (fork or copy of this one) with a commit history that shows your process
- [x] `docs/DECOMPOSITION.md`, committed before any code
- [x] A working command to run the system on any chat file
- [x] Your own test questions and results, in `docs/FINAL_NOTES.md`
- [x] `docs/FINAL_NOTES.md` complete, with the video link
- [x] The repo link, submitted via the form before the deadline


Good luck. We're looking forward to seeing how you think.
