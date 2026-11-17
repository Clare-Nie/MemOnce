#!/usr/bin/env python
"""MemOnce post-training benchmark runner.

One entry point for all released benchmarks. Every benchmark follows the
same protocol: the conversation history is written to memory once (shared
histories are written once per conversation), each question is answered
from memory only, and responses are scored with the benchmark-native
metric (LLM judge where the benchmark defines one, rules otherwise).

Usage:
  python bench.py --bench BENCH --data-dir DIR \
      --base-model M --checkpoint C [--max-questions N] \
      [--judge-url http://host:port/v1 --judge-model NAME] \
      [--out results.json --gens-out generations.jsonl]

Benchmarks and expected files under --data-dir:
  HotpotQA            --bank-file bank.jsonl + --questions-file questions.jsonl,
                  or built from HotpotQA with --budget (needs `datasets`)
  locomo         locomo10.json
  longmemeval    longmemeval_s.json
  personamem128  shared_contexts_128k.jsonl + questions_128k.csv
  personamem1m   shared_contexts_1M.jsonl + questions_1M.csv
                 (random 500, seed 20261016, grouped by conversation)
  membench       {FirstAgent,ThirdAgent}/*.json (the MemData layout)

locomo / longmemeval / HotpotQA require an LLM judge (--judge-url); personamem
and membench are scored by option-letter rules.
"""
import argparse
import csv
import glob
import json
import os
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import MemOnceChat, load_memonce  # noqa: E402

FREE_BINARY = ("A question was asked about a long document. The reference "
               "answer(s): {gold}. The model's response was: {gen}\n\n"
               "Is the model's response substantively correct (matches the "
               "reference answer, minor wording differences are fine)? "
               "Answer yes or no only.")

LOCO_JUDGE = """Your task is to label an answer to a question as `Yes` or `No`. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a `gold` (ground truth) answer,
    (3) a generated answer
which you will score as Yes/No.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as Yes.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as Yes. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it Yes if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Is the generated answer correct? Answer yes or no only."""


def lme_anscheck(task, question, answer, response, abstention=False):
    base = ("I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the "
            "intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of "
            "the information required by the answer, answer no. ")
    if task == 'temporal-reasoning':
        base += ("In addition, do not penalize off-by-one errors for the "
                 "number of days. If the question asks for the number of "
                 "days/weeks/months, etc., and the model makes off-by-one "
                 "errors (e.g., predicting 19 days when the answer is 18), "
                 "the model's response is still correct. ")
    if abstention:
        base += ('When the reference answer is "I don\'t know", "Unknown", '
                 '"I have no information", you should answer yes if the '
                 'model response expresses uncertainty or lack of knowledge '
                 '(e.g. "I\'m sorry, I don\'t have that information") or '
                 'the content of the reference answer. ')
    base += ("\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
             "\n\nIs the model response correct? Answer yes or no only.")
    return base.format(question, answer, response)


class LLMJudge:
    def __init__(self, url, model):
        if not model:
            raise ValueError("--judge-url requires --judge-model")
        self.url, self.model = url.rstrip("/") + "/chat/completions", model

    def __call__(self, prompt):
        import requests
        r = requests.post(self.url, json={
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16, "temperature": 0}, timeout=120)
        txt = r.json()["choices"][0]["message"]["content"]
        return 1.0 if (txt or "").strip().lower().startswith("yes") else 0.0


def _gold_letter(gold):
    """Normalise gold labels: datasets store them as 'b', '(b)', 'B' or '(B)'."""
    m = re.search(r"[abcd]", str(gold or "").lower())
    return m.group(0) if m else None


def option_letter(gen, gold):
    g = _gold_letter(gold)
    m = re.findall(r"\(([abcd])\)", (gen or "").lower()) or \
        re.findall(r"\b([abcd])\b", (gen or "").lower())
    return float(m[0] == g) if m and g else 0.0


def abcd_letter(gen, gold):
    g = _gold_letter(gold)
    m = re.findall(r"(?:^|\b)([ABCD])(?:\b|\.|\))", (gen or "").strip()) or \
        re.findall(r"\b([ABCD])\b", gen or "")
    return float(m[0].lower() == g) if m and g else 0.0


def _norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", s)


def f1_score(gen, gold):
    gt, kt = _norm(gen).split(), _norm(gold).split()
    if not kt:
        return 0.0
    overlap = sum(min(gt.count(t), kt.count(t)) for t in set(gt) & set(kt))
    if overlap == 0:
        return 0.0
    p, r = overlap / len(gt), overlap / len(kt)
    return 2 * p * r / (p + r)


# ------------------------------------------------------------- data loaders
def load_locomo(data_dir, limit=None):
    data = json.load(open(Path(data_dir) / "locomo10.json"))
    items = []
    for idx, item in enumerate(data):
        sa = item["conversation"]["speaker_a"]
        sb = item["conversation"]["speaker_b"]
        base = [{"role": "user", "content":
                 f"You are role-playing as {sb} in a conversation with the user is playing is  {sa}."}]
        for s in range(len(item["session_summary"])):
            session = item["conversation"][f"session_{s + 1}"]
            t = item["conversation"][f"session_{s + 1}_date_time"]
            for d in session:
                text = d["text"]
                if d.get("blip_caption"):
                    text = f"{text} (image description: {d['blip_caption']})"
                if d["speaker"] == sa:
                    base.append({"role": "user", "content": f"{t} {sa}\n{text}"})
                elif d["speaker"] == sb:
                    base.append({"role": "assistant", "content": f"{sb}\n{text}"})
        for i, qa in enumerate(item["qa"]):
            if str(qa["category"]) == "5":
                continue
            answer = qa["answer"] if "answer" in qa else qa["adversarial_answer"]
            qp = (f"Any content referring to 'User' in the prompt refers to {sa}'s content, and any content referring to 'AI'or 'assiant' refers to {sb}'s content."
                  f"Your task is to answer questions about {sa} or {sb} in an extremely concise manner.\nQuestion: {qa['question']}")
            items.append({"qid": f"locomo_{idx}_{i}", "conv": idx,
                          "messages": base + [{"role": "user", "content": qp}],
                          "question": qa["question"], "answer": answer,
                          "question_type": qa["category"], "judge": "locomo"})
            if limit and len(items) >= limit:
                return items
    return items


def load_longmemeval(data_dir, limit=None):
    data = json.load(open(Path(data_dir) / "longmemeval_s.json"))
    items = []
    for item in data[:limit] if limit else data:
        msgs = []
        for session, ts in zip(item["haystack_sessions"], item["haystack_dates"]):
            for msg in session:
                c = msg["content"]
                if msg["role"] == "user":
                    c = f"{ts}\n{c}"
                msgs.append({"role": msg["role"], "content": c})
        qp = ("Please answer the following question based on the history of conversations:\n"
              f"Question time: {item['question_date']} and question: {item['question']}")
        msgs.append({"role": "user", "content": qp})
        items.append({"qid": item["question_id"], "conv": item["question_id"],
                      "messages": msgs, "question": item["question"],
                      "answer": item["answer"],
                      "question_type": item["question_type"],
                      "abstain": "_abs" in item["question_id"],
                      "judge": "lme"})
    return items


def load_personamem(data_dir, variant, limit=None):
    data_dir = Path(data_dir)
    if not (data_dir / f"shared_contexts_{variant}.jsonl").exists():
        data_dir = data_dir / "personaMem"
    ctxs = {}
    with open(data_dir / f"shared_contexts_{variant}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            ctxs[list(s.keys())[0]] = list(s.values())[0]
    with open(data_dir / f"questions_{variant}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: int(x.get("end_index_in_shared_context", 0)))
    by_ctx = {}
    for r in rows:
        by_ctx.setdefault(r["shared_context_id"], []).append(r)
    items = []
    for key, qs in by_ctx.items():
        conv = ctxs.get(key)
        if conv is None:
            continue
        base = [{"role": "user", "content": str(conv[0]["content"])}]
        for m in conv[1:]:
            if m["role"] == "system":
                continue
            c = m["content"].replace(
                "User:" if m["role"] == "user" else "Assistant:", "").strip()
            base.append({"role": m["role"], "content": c})
        for i, qa in enumerate(qs):
            qp = (qa["user_question_or_message"] + "\nOptions:\n" + qa["all_options"]
                  + "\nFind the most appropriate model response and give your final answer (a), (b), (c), or (d).")
            items.append({"qid": f"pmem_{key}_{i}", "conv": key,
                          "messages": base + [{"role": "user", "content": qp}],
                          "question": qa["user_question_or_message"],
                          "answer": qa["correct_answer"],
                          "question_type": qa["question_type"],
                          "judge": "option_letter"})
    if variant == "1M" and (limit is None or limit > 500):
        import random
        rng = random.Random(20261016)
        items = rng.sample(items, 500)
        items.sort(key=lambda r: (r["conv"], int(r["qid"].rsplit("_", 1)[-1])))
    return items[:limit] if limit else items


def load_membench(data_dir, limit=None):
    items = []
    files = sorted(glob.glob(str(Path(data_dir) / "*" / "*.json"))) or         sorted(glob.glob(str(Path(data_dir) / "*.json")))
    for fp in files:
        task = Path(fp).stem
        d = json.load(open(fp))
        for cat, trajs in d.items():
            for tr in trajs:
                msgs = []
                for sess in tr["message_list"]:
                    ml = sess if isinstance(sess, list) else [sess]
                    for m in ml:
                        if "user_message" in m:
                            msgs.append({"role": "user", "content": f"{m['user_message']} (place: {m.get('place')}; time{m.get('time')})"})
                            if m.get("assistant_message"):
                                msgs.append({"role": "assistant", "content": m["assistant_message"]})
                        elif "user" in m:
                            msgs.append({"role": "user", "content": f"{m['user']} (place: {m.get('place')}; time{m.get('time')})"})
                            if m.get("assistant"):
                                msgs.append({"role": "assistant", "content": m["assistant"]})
                        elif "message" in m:
                            msgs.append({"role": "user", "content": "记录以下内容。"})
                            msgs.append({"role": "assistant", "content": f"{m['message']} (place: {m.get('place')}; time{m.get('time')})"})
                qa = tr["QA"]
                ch = qa["choices"]
                q = (f"Please answer the following question based on past memories of the user's messages.\n"
                     f"Question: (current time is {qa.get('time')}) {qa['question']}\n"
                     f"Choices:\nA. {ch['A']}\nB. {ch['B']}\nC. {ch['C']}\nD. {ch['D']}\n"
                     f"Please output the correct option for the question, only one corresponding letter, without any other messages.")
                items.append({"qid": f"{task}/{cat}/{tr['tid']}", "conv": f"{task}/{cat}/{tr['tid']}",
                              "messages": msgs, "question": q,
                              "answer": qa["ground_truth"], "judge": "abcd"})
                if limit and len(items) >= limit:
                    return items
    return items


def load_hotpotqa(args, tok):
    if args.bank_file:
        bank = [json.loads(l) for l in open(args.bank_file)]
        qs = [json.loads(l) for l in open(args.questions_file)]
    else:
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                          streaming=True, trust_remote_code=True)
        bank, qs, used, n_pid = [], [], 0, 0
        for ex in ds:
            q, ans = ex.get("question"), ex.get("answer")
            ctx = ex.get("context") or {}
            kept = []
            for title, sents in zip(ctx.get("title", []) or [],
                                    ctx.get("sentences", []) or []):
                text = " ".join(sents)
                if not q or not ans or len(text.split()) < 20:
                    continue
                toks = len(tok(text, add_special_tokens=False)["input_ids"])
                if used + toks > args.budget:
                    continue
                pid = f"p{n_pid:08d}"
                bank.append({"pid": pid, "title": title, "text": text})
                kept.append(pid)
                used += toks
                n_pid += 1
                if used >= args.budget:
                    break
            if kept and q and ans:
                qs.append({"question": q, "answer": ans, "evidence_pids": kept})
            if used >= args.budget:
                break
        step = max(1, len(qs) // 500)
        qs = qs[::step][:500]
        for i, item in enumerate(qs):
            item["qid"] = f"hotpotqa_{i:05d}"
    msgs = [{"role": "user", "content": f"[{d['pid']}] {d.get('title', '')}\n{d['text']}"}
            for d in bank]
    items = []
    for i, q in enumerate(qs[: args.max_questions] if args.max_questions else qs):
        qid = q.get("qid", f"hotpotqa_{i:05d}")
        items.append({"qid": qid, "conv": "hotpotqa_bank",
                      "messages": msgs + [{"role": "user", "content": q["question"]}],
                      "question": q["question"], "answer": q["answer"],
                      "question_type": "hotpotqa", "judge": "hotpotqa"})
    return items


LOADERS = {"HotpotQA": None, "locomo": load_locomo, "longmemeval": load_longmemeval,
           "personamem128": lambda d, limit=None: load_personamem(d, "128k", limit),
           "personamem1m": lambda d, limit=None: load_personamem(d, "1M", limit),
           "membench": load_membench}


# -------------------------------------------------------------------- score
def score_item(item, gen, judge, hotpotqa_f1):
    j = item["judge"]
    if j == "locomo":
        return judge(LOCO_JUDGE.format(question=item["question"],
                                       gold_answer=str(item["answer"]),
                                       generated_answer=gen))
    if j == "lme":
        return judge(lme_anscheck(item.get("question_type", "qa"),
                                  item["question"], item["answer"], gen,
                                  abstention=item.get("abstain", False)))
    if j == "hotpotqa":
        if hotpotqa_f1 or judge is None:
            return f1_score(gen, item["answer"])
        return judge(FREE_BINARY.format(gold=item["answer"], gen=gen))
    if j == "option_letter":
        return option_letter(gen, item["answer"])
    if j == "abcd":
        return abcd_letter(gen, item["answer"])
    raise ValueError(j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=sorted(LOADERS))
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--base-model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank-file", type=Path, default=None, help="HotpotQA only")
    p.add_argument("--questions-file", type=Path, default=None, help="HotpotQA only")
    p.add_argument("--budget", type=int, default=1350000, help="HotpotQA only")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--judge-url", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--hotpotqa-f1", action="store_true",
                   help="score HotpotQA with F1 instead of the LLM judge")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gens-out", type=Path, default=None)
    a = p.parse_args()

    model, tok = load_memonce(a.base_model, a.checkpoint)
    if a.bench == "HotpotQA":
        items = load_hotpotqa(a, tok)
    else:
        if a.data_dir is None:
            p.error(f"--bench {a.bench} requires --data-dir")
        items = LOADERS[a.bench](a.data_dir, a.max_questions)

    need_llm = any(it["judge"] in ("locomo", "lme") or
                   (it["judge"] == "hotpotqa" and not a.hotpotqa_f1) for it in items)
    judge = LLMJudge(a.judge_url, a.judge_model) if (a.judge_url and need_llm) else None
    if need_llm and judge is None:
        p.error("this benchmark requires --judge-url and --judge-model "
                "(or --hotpotqa-f1 for HotpotQA)")

    by_conv = {}
    for it in items:
        by_conv.setdefault(it["conv"], []).append(it)
    print(f"[bench] {a.bench}: {len(items)} questions over "
          f"{len(by_conv)} conversation(s)", flush=True)

    gens, t0 = [], time.time()
    done = 0
    for conv, qs in by_conv.items():
        chat = MemOnceChat(model, tok)
        hist = qs[0]["messages"][:-1]
        if hist:
            chat.write(hist)
        for it in qs:
            ans, info = chat.answer(it["messages"][-1]["content"],
                                    a.max_new_tokens)
            sc = score_item(it, ans, judge, a.hotpotqa_f1)
            gens.append({"qid": it["qid"], "question": it["question"],
                         "answer": it["answer"], "generated": ans,
                         "score": sc})
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(items)}]", flush=True)
    vals = [g["score"] for g in gens if g["score"] is not None]
    result = {"bench": a.bench, "n": len(gens),
              "score": round(100.0 * sum(vals) / max(1, len(vals)), 2),
              "scoring": ("llm:" + a.judge_model) if judge else
                         ("f1" if a.bench == "HotpotQA" else "rule"),
              "seconds": round(time.time() - t0, 1)}
    out = a.out or Path(f"bench_{a.bench}_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    with open(a.gens_out or Path(f"bench_{a.bench}_generations.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
