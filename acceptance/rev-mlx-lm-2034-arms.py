"""Mutation arms for the review of ml-explore/mlx-lm#2034 (run from src/ on the runner)."""
import hashlib, os, re, subprocess, sys, time

SRC_FILE = "mlx_lm/server.py"
BASE_SHA = "736ab0d8c1d0edc79c3d798044fe6904e2fa23be"
env = dict(os.environ)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def blob(path):
    return sh(["git", "hash-object", path]).stdout.strip()


HEAD_TEXT = open(SRC_FILE).read()
HEAD_BLOB = blob(SRC_FILE)
assert HEAD_BLOB == sh(["git", "rev-parse", f"HEAD:{SRC_FILE}"]).stdout.strip(), "working copy is not the head blob"

ARMS = []


def arm(aid, why, old, new, count=1):
    ARMS.append((aid, why, old, new, count))


# ---- call sites (the issue's sentence: a prompt over the context is accepted and processed) ----
arm("A01-handler-no-400", "handle_completion without the except PromptTooLongError branch",
    '''        except PromptTooLongError as e:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        except Exception as e:''', '        except Exception as e:')
arm("A02-handler-status-500", "handler answers 500 instead of 400",
    'except PromptTooLongError as e:\n            self._set_completion_headers(400)',
    'except PromptTooLongError as e:\n            self._set_completion_headers(500)')
arm("A03-batch-call-removed", "the batch branch never checks the prompt length",
    '                        self._check_prompt_length(prompt)\n                    except Exception as e:',
    '                    except Exception as e:')
arm("A04-single-call-removed", "the single path never checks the prompt length",
    '            self._check_prompt_length(prompt)\n            stop_sm = TextStateMachine.from_stop_words(args.stop_words)\n            text_sm = TextStateMachine.from_tokenizer(tokenizer)\n\n            # Start',
    '            stop_sm = TextStateMachine.from_stop_words(args.stop_words)\n            text_sm = TextStateMachine.from_tokenizer(tokenizer)\n\n            # Start')
arm("A05-single-call-after-ctx", "single path checks only after rqueue.put(ctx) (refusal after the answer began)",
    '            self._check_prompt_length(prompt)\n            stop_sm = TextStateMachine.from_stop_words(args.stop_words)\n            text_sm = TextStateMachine.from_tokenizer(tokenizer)\n\n            # Start',
    '            stop_sm = TextStateMachine.from_stop_words(args.stop_words)\n            text_sm = TextStateMachine.from_tokenizer(tokenizer)\n\n            # Start')
# A05 needs a second edit: handled in code below (insert after rqueue.put(ctx) in _serve_single)
# ---- the check ----
arm("A06-check-never-raises", "the check never refuses",
    'if limit is not None and len(prompt) > limit:', 'if False:')
arm("A07-off-by-one", ">= instead of >", 'len(prompt) > limit:', 'len(prompt) >= limit:')
arm("A08-unknown-limit-refuses", "an unknown limit refuses",
    'if limit is not None and len(prompt) > limit:', 'if limit is None or len(prompt) > limit:')
arm("A09-message-no-limit", "message without the limit",
    'f"model context length of {limit} tokens."', '"model context length."')
arm("A10-message-no-context-word", "message without the word context",
    'f"model context length of {limit} tokens."', 'f"model max length of {limit} tokens."')
arm("A11-message-no-count", "message without the prompt token count",
    'f"Prompt has {len(prompt)} tokens, which is longer than the "', '"Prompt is longer than the "')
arm("A12-error-base-exception", "PromptTooLongError no longer a ValueError",
    'class PromptTooLongError(ValueError):', 'class PromptTooLongError(Exception):')
# ---- the provider ----
arm("A13-init-no-attr", "ModelProvider.__init__ declares no context_length",
    '        self.context_length: Optional[int] = None\n', '')
arm("A14-reset-keeps", "reset keeps the limit",
    '        self.is_batchable = False\n        self.context_length = None\n', '        self.is_batchable = False\n')
arm("A15-load-drops", "_load drops the limit",
    'self.context_length = _declared_context_length(config)', 'self.context_length = None')
arm("A16-load-tokenizer-source", "_load takes the limit from the tokenizer",
    'self.context_length = _declared_context_length(config)',
    'self.context_length = getattr(tokenizer, "model_max_length", None)')
# ---- the reader ----
arm("A17-rope-gate-removed", "reader ignores rope scaling",
    'if any(_scales_rope(section) for section in sections):', 'if False:')
arm("A18-rope-toplevel-only", "rope scaling read only at the top level",
    'if any(_scales_rope(section) for section in sections):', 'if _scales_rope(config):')
arm("A19-scales-default-counts", "a default-type rope counts as scaled",
    '        if (params.get("type") or params.get("rope_type", "default")) != "default":\n            return True',
    '        return True')
arm("A20-scales-ignores-rope_type", "rope_type key ignored",
    '(params.get("type") or params.get("rope_type", "default"))', 'params.get("type", "default")')
arm("A21-scales-ignores-type", "legacy type key ignored",
    '(params.get("type") or params.get("rope_type", "default"))', 'params.get("rope_type", "default")')
arm("A22-scales-ignores-rope_parameters", "rope_parameters key ignored",
    '("rope_scaling", "rope_parameters")', '("rope_scaling",)')
arm("A23-scales-ignores-rope_scaling", "rope_scaling key ignored",
    '("rope_scaling", "rope_parameters")', '("rope_parameters",)')
arm("A24-scales-nested-not-scaled", "per-layer-type nested rope params not treated as scaled",
    '''        if not isinstance(params, dict) or any(
            isinstance(v, dict) for v in params.values()
        ):
            return True''', '''        if not isinstance(params, dict):
            return True''')
arm("A25-scales-nondict-not-scaled", "non-dict truthy rope params not treated as scaled",
    '''        if not isinstance(params, dict) or any(
            isinstance(v, dict) for v in params.values()
        ):
            return True''', '''        if any(isinstance(v, dict) for v in params.values()):
            return True''')
arm("A26-scales-falsy-not-skipped", "falsy rope params (None/{}) are not skipped",
    '        if not params:\n            continue\n', '')
arm("A27-text-config-ignored", "text_config never consulted",
    '        sections.append(config["text_config"])', '        pass')
arm("A28-text-config-wins", "text_config wins over the top level",
    '        sections.append(config["text_config"])', '        sections.insert(0, config["text_config"])')
arm("A29-bool-accepted", "bool accepted as a limit",
    'isinstance(limit, int) and not isinstance(limit, bool) and limit > 0', 'isinstance(limit, int) and limit > 0')
arm("A30-zero-accepted", "zero accepted as a limit",
    'isinstance(limit, int) and not isinstance(limit, bool) and limit > 0', 'isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0')
arm("A31-float-accepted", "float accepted as a limit",
    'isinstance(limit, int) and not isinstance(limit, bool) and limit > 0', 'isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0')
arm("A32-negative-accepted", "negative accepted as a limit",
    'isinstance(limit, int) and not isinstance(limit, bool) and limit > 0', 'isinstance(limit, int) and not isinstance(limit, bool)')

arm("A33-limit-key-wrong", "reader looks up the wrong key",
    'limit = section.get("max_position_embeddings")', 'limit = section.get("max_position_embedding")')
arm("A34-handler-catches-all-400", "every generation error answers 400",
    'except PromptTooLongError as e:\n            self._set_completion_headers(400)',
    'except Exception as e:\n            self._set_completion_headers(400)')
arm("A35-max-tokens-counted", "batch call counts prompt plus max_tokens",
    '                        self._check_prompt_length(prompt)\n                    except Exception as e:',
    '                        self._check_prompt_length(list(prompt) + [0] * args.max_tokens)\n                    except Exception as e:')
arm("A36-limit-minus-two", "limit lowered by two",
    'len(prompt) > limit:', 'len(prompt) > limit - 2:')


def apply(text, old, new, count):
    assert text.count(old) == count, f"pattern count {text.count(old)} != {count}: {old[:60]!r}"
    return text.replace(old, new)


def run_tests(tag, extra=()):
    log = f"../arm-{tag}.log"
    t = time.time()
    p = subprocess.run(
        ["uv", "run", "python", "-m", "unittest", "discover", "-v", "-s", "tests", "-p", "test_server.py", *extra],
        capture_output=True, text=True, env=env, timeout=1500)
    open(log, "w").write(p.stdout + p.stderr)
    out = p.stdout + p.stderr
    ran = re.findall(r"^Ran (\d+) tests", out, re.M)
    red = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", out, re.M)))
    return p.returncode, (ran[-1] if ran else "none"), red, round(time.time() - t)


def restore():
    open(SRC_FILE, "w").write(HEAD_TEXT)
    assert blob(SRC_FILE) == HEAD_BLOB, "restore failed"


only = re.compile(os.environ.get("ARMS_ONLY", ".*"))
print(f"ARMS-HEAD sha={sh(['git','rev-parse','HEAD']).stdout.strip()} server_blob={HEAD_BLOB}", flush=True)
nred = nrun = 0
# base-server arm: the head tests against the base server.py
if only.search("B00-base-server"):
    base_text = sh(["git", "show", f"{BASE_SHA}:{SRC_FILE}"]).stdout
    open(SRC_FILE, "w").write(base_text)
    nfiles = len([l for l in sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.splitlines() if l.strip()])
    print(f"ARM id=B00-base-server why=head tests on the base server.py before={HEAD_BLOB} after={blob(SRC_FILE)} base_blob_expected={sh(['git','rev-parse',f'{BASE_SHA}:{SRC_FILE}']).stdout.strip()} changed_files={nfiles}", flush=True)
    rc, ran, red, secs = run_tests("B00")
    print(f"ARMRESULT id=B00-base-server rc={rc} ran={ran} red={len(red)} secs={secs}", flush=True)
    for n, c in red:
        print(f"  RED B00-base-server {c}.{n}", flush=True)
    restore()
for aid, why, old, new, count in ARMS:
    if not only.search(aid):
        continue
    text = apply(HEAD_TEXT, old, new, count)
    if aid == "A05-single-call-after-ctx":
        text = apply(text, "            rqueue.put(ctx)\n\n            # Seed if requested", "            rqueue.put(ctx)\n            self._check_prompt_length(prompt)\n\n            # Seed if requested", 1)
    open(SRC_FILE, "w").write(text)
    nfiles = len([l for l in sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.splitlines() if l.strip()])
    print(f"ARM id={aid} why={why} before={HEAD_BLOB} after={blob(SRC_FILE)} changed_files={nfiles}", flush=True)
    assert blob(SRC_FILE) != HEAD_BLOB
    rc, ran, red, secs = run_tests(aid[:3])
    nrun += 1
    nred += 1 if red else 0
    print(f"ARMRESULT id={aid} rc={rc} ran={ran} red={len(red)} secs={secs}", flush=True)
    for n, c in red:
        print(f"  RED {aid} {c}.{n}", flush=True)
    restore()
    print(f"RESTORED id={aid} blob={blob(SRC_FILE)} status_porcelain={len(sh(['git','status','--porcelain','--untracked-files=no']).stdout.splitlines())}", flush=True)
print(f"ARMS-SUMMARY run={nrun} red={nred}", flush=True)
