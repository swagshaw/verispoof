"""VeriSpoof inference: detection, recording condition, generator identification, source verification and
reasoning.

Closed answers are scored by the log-likelihood of each candidate answer sentence (detection:
score = log p(spoof answer) - log p(bona fide answer), > 0 means spoof); reasoning is generated greedily.
Audio: 16 kHz mono; the first 8 s are used, shorter recordings are repeated to fill the window.
"""
import json
import os
import re

import numpy as np
import torch

from verispoof.model import AUDIO, Network

SR = 16000

AUTH_QUESTION = "Is this recording genuine human speech or spoofed speech? Options: bona fide; spoof."
AUTH_ANSWERS = ("This recording is bona fide: it is genuine human speech.",
                "This recording is a spoof: it is synthetic or converted speech, not genuine human speech.")
PAIR = {  # question, (yes sentence, no sentence)
    "generator": ("Do Audio A and Audio B come from the same generator?",
                  ("Yes, both recordings were produced by the same speech generation system.",
                   "No, the two recordings were produced by different speech generation systems.")),
    "acoustic_model": ("Were Audio A and Audio B generated with the same acoustic model?",
                       ("Yes, both recordings were generated with the same acoustic model.",
                        "No, the two recordings use different acoustic models.")),
    "vocoder": ("Do Audio A and Audio B share the same vocoder?",
                ("Yes, both recordings were synthesized with the same vocoder.",
                 "No, the two recordings use different vocoders.")),
}
CONDITION = {  # question, options, answer sentence per option
    "noise": ("What is the background noise level of this recording?",
              ["clean", "mild", "moderate", "severe", "extreme"],
              lambda o: "The recording is clean, with no audible noise." if o == "clean"
              else f"The recording has {o} background noise."),
    "reverb": ("How much room reverberation does this recording have?",
               ["minimal", "slight", "moderate", "heavy", "extreme"],
               lambda o: "The recording has no noticeable reverberation." if o == "minimal"
               else f"The recording has {o} reverberation."),
    "codec": ("Was this recording compressed by a codec, and if so which kind?",
              ["none", "MP3", "AAC", "Opus", "narrowband telephone", "wideband telephone", "neural codec",
               "MP3 + neural codec"],
              lambda o: "The recording shows no codec compression." if o == "none"
              else f"The recording shows {o} coding." if o.endswith("telephone")
              else "The recording shows MP3 and neural-codec compression in cascade." if o == "MP3 + neural codec"
              else f"The recording shows {o} compression."),
}
GENERATOR_QUESTION = "Which generator was used to create this speech?"
EXPLAIN_QUESTION = "Analyse this recording step by step and decide whether it is genuine human speech or spoofed."
COMPARE_QUESTION = ("Compare the source profiles of the two recordings and decide whether they were produced by the "
                    "same generation system.")
REASONING_FIELDS = ("OBSERVATIONS", "CHANNEL_STATUS", "LOCALIZATION", "SOURCE_PROFILE", "FINGERPRINT",
                    "PROFILE_COMPATIBILITY", "DECISION")


def load_audio(x, sr=None):
    """A path, or a 1-D array with its sample rate -> mono float32 at 16 kHz."""
    if isinstance(x, (str, os.PathLike)):
        import soundfile as sf
        try:
            x, sr = sf.read(str(x), dtype="float32", always_2d=True)
            x = x[:, 0]
        except sf.LibsndfileError:  # formats libsndfile cannot read
            import librosa
            x, sr = librosa.load(str(x), sr=None, mono=False)
            x = x if x.ndim == 1 else x[0]
    else:
        x = np.asarray(x, dtype=np.float32)
        if sr is None:
            raise ValueError("give the sample rate of an array input")
        if x.ndim > 1:
            x = x[0] if x.shape[0] < x.shape[1] else x[:, 0]
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return x.astype(np.float32)


def fit_window(x, S):
    """First S samples; shorter recordings are tiled; an empty one is digital silence."""
    if len(x) >= S:
        return x[:S]
    if not len(x):
        return np.zeros(S, np.float32)
    return np.tile(x, S // len(x) + 1)[:S]


def parse_reasoning(text):
    """Reasoning text -> its lines by field name, the decision and any localized spans (s)."""
    f = {}
    for line in text.split("\n"):
        for k in REASONING_FIELDS:
            if line.startswith(k + ":"):
                f[k] = line[len(k) + 1:].strip()
    d = f.get("DECISION", "")
    short = text.strip()  # a pair answered with the short verification sentence instead of reasoning lines
    if "come from the same generation system" in d or short.startswith("Yes, both recordings were produced by the same"):
        dec = "same"
    elif "come from different generation systems" in d or short.startswith("No, the two recordings were produced by"):
        dec = "different"
    else:
        dec = "bonafide" if d.startswith("The recording is bona fide") else \
            "spoof" if d.startswith("The recording is a spoof") else None
    spans = [[float(a), float(b)] for a, b in re.findall(r"(\d+\.\d+)-(\d+\.\d+) s", f.get("LOCALIZATION", ""))]
    return {"lines": f, "decision": dec, "spans": spans}


class VeriSpoof:
    def __init__(self, net, device="cuda"):
        self.net, self.device = net, device

    @classmethod
    def from_pretrained(cls, path, device="cuda", revision=None):
        """path: a local directory or a Hugging Face repository id."""
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision)
        cfg = json.load(open(f"{path}/config.json"))
        return cls(Network(path, cfg, device=device).to(device).eval(), device)

    # ---------------------------------------------------------------- inputs
    def _windows(self, audios, sr=None):
        w = [fit_window(load_audio(a, sr), self.net.window) for a in audios]
        return torch.from_numpy(np.stack(w))

    @staticmethod
    def _batch(items, n):
        return [items[i:i + n] for i in range(0, len(items), n)]

    def _left_pad(self, P):
        Lp = max(map(len, P))
        ids = torch.full((len(P), Lp), self.net.pad_id, dtype=torch.long)
        att = torch.zeros((len(P), Lp), dtype=torch.long)
        for i, p in enumerate(P):
            ids[i, Lp - len(p):], att[i, Lp - len(p):] = torch.tensor(p), 1
        return ids.to(self.device), att.to(self.device)

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def _score(self, prompts, wav, answers):
        """prompts: B token-id lists; wav: (N_audio, S) in prompt order; answers: per prompt, a list of candidate
        sentences -> list of arrays of total log-likelihoods."""
        lm = self.net.causal_lm()
        ids, att = self._left_pad(prompts)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        emb = self.net.embed(ids, wav.to(self.device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.net.amp):
            out = lm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
            first = lm.lm_head(out.last_hidden_state[:, -1]).float().log_softmax(-1)
        rows = [(b, self.net.text_ids(c + "<|im_end|>")) for b, cs in enumerate(answers) for c in cs]
        Lc = max(len(c) for _, c in rows)
        cand = torch.full((len(rows), Lc), self.net.pad_id, dtype=torch.long, device=self.device)
        cm = torch.zeros((len(rows), Lc), dtype=torch.long, device=self.device)
        for r, (_, c) in enumerate(rows):
            cand[r, :len(c)], cm[r, :len(c)] = torch.tensor(c), 1
        owner = torch.tensor([b for b, _ in rows], device=self.device)
        lp = first[owner].gather(1, cand[:, :1]).squeeze(1)
        if Lc > 1:
            cache = out.past_key_values
            cache.reorder_cache(owner)
            att2 = torch.cat([att[owner], cm], 1)
            pos2 = att.sum(1)[owner, None] + torch.arange(Lc, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.net.amp):
                h = lm.model(input_ids=cand, attention_mask=att2, position_ids=pos2, past_key_values=cache,
                             use_cache=False).last_hidden_state
                logp = lm.lm_head(h[:, :-1]).float().log_softmax(-1)
            lp = lp + (logp.gather(2, cand[:, 1:, None]).squeeze(2) * cm[:, 1:]).sum(1)
        lp = lp.cpu().numpy()
        out_, o = [], 0
        for cs in answers:
            out_.append(lp[o:o + len(cs)])
            o += len(cs)
        return out_

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def _generate(self, prompts, wav, max_new_tokens):
        lm = self.net.causal_lm()
        ids, att = self._left_pad(prompts)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        emb = self.net.embed(ids, wav.to(self.device))
        eos, pad = self.net.end_id, self.net.pad_id
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.net.amp):
            out = lm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
            nxt = lm.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
        h_prompt = out.last_hidden_state
        cache, cur = out.past_key_values, att.sum(1)
        gen, done = [nxt], nxt == eos
        for _ in range(max_new_tokens - 1):
            if done.all():
                break
            att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.net.amp):
                out = lm.model(input_ids=nxt[:, None], attention_mask=att, position_ids=cur[:, None],
                               past_key_values=cache, use_cache=True)
                nxt = lm.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            cache, cur = out.past_key_values, cur + 1
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            gen.append(nxt)
            done |= nxt == eos
        G = torch.stack(gen, 1).cpu().tolist()
        texts = [self.net.text_tok.decode(g[:g.index(eos)] if eos in g else g) for g in G]
        return texts, h_prompt, ids

    # ---------------------------------------------------------------- public API
    def detect(self, audios, sr=None, batch_size=16):
        """One recording or a list -> {'label', 'score'} (or a list of them). score = log p(spoof answer) -
        log p(bona fide answer); the paper's EERs are computed on this score, 'label' uses threshold 0."""
        single = not isinstance(audios, (list, tuple))
        audios = [audios] if single else list(audios)
        p = self.net.prompt_ids(f"{AUDIO}\n{AUTH_QUESTION}")
        out = []
        for chunk in self._batch(audios, batch_size):
            for s in self._score([p] * len(chunk), self._windows(chunk, sr), [AUTH_ANSWERS] * len(chunk)):
                score = float(s[1] - s[0])
                out.append({"label": "spoof" if score > 0 else "bonafide", "score": score})
        return out[0] if single else out

    def choose(self, audio, question, options, answers, sr=None):
        """Closed question about one recording: the option whose answer sentence is most likely."""
        prompt = self.net.prompt_ids(f"{AUDIO}\n{question} Options: {'; '.join(options)}.")
        s = self._score([prompt], self._windows([audio], sr), [answers])[0]
        return {"answer": options[int(np.argmax(s))], "loglik": dict(zip(options, map(float, s)))}

    def generator(self, audio, candidates, sr=None):
        """Which of the candidate generation systems produced the recording (closed set you supply)."""
        return self.choose(audio, GENERATOR_QUESTION, list(candidates),
                           [f"This recording was generated by {c}." for c in candidates], sr)

    def condition(self, audio, attr="codec", sr=None):
        """Recording condition: attr in noise, reverb, codec."""
        q, opts, sent = CONDITION[attr]
        return self.choose(audio, q, opts, [sent(o) for o in opts], sr)

    def same_source(self, audio_a, audio_b, level="generator", sr=None):
        """Do two spoofed recordings share the generator (or acoustic model, or vocoder)? score = log p(yes) -
        log p(no), > 0 means the same."""
        q, (yes, no) = PAIR[level]
        prompt = self.net.prompt_ids(f"Audio A: {AUDIO}\nAudio B: {AUDIO}\n{q}")
        s = self._score([prompt], self._windows([audio_a, audio_b], sr), [[yes, no]])[0]
        score = float(s[0] - s[1])
        return {"same": score > 0, "score": score}

    @torch.no_grad()
    def explain(self, audio, sr=None, max_new_tokens=320):
        """Grounded detection reasoning (greedy) plus the per-token head: probability that each 0.16 s token of the
        heard window is synthesized."""
        p = self.net.prompt_ids(f"{AUDIO}\n{EXPLAIN_QUESTION}")
        texts, h, ids = self._generate([p], self._windows([audio], sr), max_new_tokens)
        z = self.net.token_logits(h, ids)[0].sigmoid().cpu().numpy()
        r = parse_reasoning(texts[0])
        r.update(text=texts[0], token_prob=z.tolist(), token_seconds=self.net.seconds_per_token())
        return r

    def compare(self, audio_a, audio_b, sr=None, max_new_tokens=320):
        """Source-verification reasoning for two recordings (greedy)."""
        p = self.net.prompt_ids(f"Audio A: {AUDIO}\nAudio B: {AUDIO}\n{COMPARE_QUESTION}")
        texts, _, _ = self._generate([p], self._windows([audio_a, audio_b], sr), max_new_tokens)
        r = parse_reasoning(texts[0])
        r["text"] = texts[0]
        return r
