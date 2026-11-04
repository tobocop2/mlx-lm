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
HEAD_TEXTS = {SRC_FILE: HEAD_TEXT, "mlx_lm/models/rope_utils.py": open("mlx_lm/models/rope_utils.py").read()}
HEAD_BLOBS = {p: blob(p) for p in HEAD_TEXTS}
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
arm("A12-class-dropped", "PromptTooLongError class removed",
    'class PromptTooLongError(Exception):\n    pass\n\n\n', '')
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
    '        if get_rope_type(params) != "default":\n            return True',
    '        return True')
arm("A20-scales-ignores-rope_type", "rope_type key ignored",
    'get_rope_type(params) != "default"', 'params.get("type", "default") != "default"')
arm("A21-scales-ignores-type", "legacy type key ignored (server site)",
    'get_rope_type(params) != "default"', 'params.get("rope_type", "default") != "default"')
arm("A22-scales-ignores-rope_parameters", "rope_parameters key ignored",
    '("rope_scaling", "rope_parameters")', '("rope_scaling",)')
arm("A23-scales-ignores-rope_scaling", "rope_scaling key ignored",
    '("rope_scaling", "rope_parameters")', '("rope_parameters",)')
arm("A24-scales-nested-not-scaled", "per-layer-type nested rope params not treated as scaled",
    '''        if not isinstance(params, dict) or any(
            isinstance(v, dict) for v in params.values()
        ):''', '''        if not isinstance(params, dict):''')
arm("A25-scales-nondict-not-scaled", "non-dict truthy rope params not treated as scaled",
    '''        if not isinstance(params, dict) or any(
            isinstance(v, dict) for v in params.values()
        ):''', '''        if any(isinstance(v, dict) for v in params.values()):''')
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

ROPE = "mlx_lm/models/rope_utils.py"
META = {}
def arm2(aid, why, old, new, path=SRC_FILE, tests=None):
    ARMS.append((aid, why, old, new, 1))
    META[aid] = (path, tests)
arm2("A37-log-removed", "the unknown-rope-format warning removed",
    '''            logging.warning(
                f"Unknown {key} format. The prompt length is not checked "
                "against the model context length."
            )
''', "")
arm2("A38-log-everywhere", "a warning logged for every rope section that is present",
    '        params = section.get(key)\n', '        params = section.get(key)\n        if params:\n            logging.warning(f"Unknown {key} format.")\n')
arm2("A39-alias-route-removed", "the /chat/completions alias route removed",
    '            "/chat/completions": self.handle_chat_completions,\n', '')
MSC = ("tests/test_models.py", ["-k", "test_apply_yarn_mscale", "-k", "test_rope"])
arm2("A40-mscale-ignores-type", "apply_yarn_mscale ignores the legacy type key",
    '    rope_type = get_rope_type(scaling_config)\n    mscale_all_dim', '    rope_type = scaling_config.get("rope_type", "default")\n    mscale_all_dim', ROPE, MSC)
arm2("A41-helper-ignores-type", "get_rope_type ignores the legacy type key",
    'return scaling_config.get("type") or scaling_config.get("rope_type", "default")', 'return scaling_config.get("rope_type", "default")', ROPE, None)
arm2("A42-helper-no-none-guard", "get_rope_type without the None guard",
    '    if scaling_config is None:\n        return "default"\n    return scaling_config.get("type")', '    return scaling_config.get("type")', ROPE, MSC)
arm2("A43-mscale-formula", "apply_yarn_mscale returns scale * s",
    '    return scale * s * s', '    return scale * s', ROPE, MSC)
arm2("A44-init-rope-ignores-type", "initialize_rope ignores the legacy type key (base-equal behaviour broken)",
    'return scaling_config.get("type") or scaling_config.get("rope_type", "default")', 'return scaling_config.get("rope_type", "default")', ROPE, ("tests/test_models.py", ["-k", "test_rope"]))


def apply(text, old, new, count):
    assert text.count(old) == count, f"pattern count {text.count(old)} != {count}: {old[:60]!r}"
    return text.replace(old, new)


def run_tests(tag, extra=(), target="test_server.py"):
    log = f"../arm-{tag}.log"
    t = time.time()
    p = subprocess.run(
        ["uv", "run", "python", "-m", "unittest", "discover", "-v", "-s", "tests", "-p", target, *extra],
        capture_output=True, text=True, env=env, timeout=1500)
    open(log, "w").write(p.stdout + p.stderr)
    out = p.stdout + p.stderr
    ran = re.findall(r"^Ran (\d+) tests", out, re.M)
    red = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+) \((\S+)\)", out, re.M)))
    return p.returncode, (ran[-1] if ran else "none"), red, round(time.time() - t)


def restore():
    for p, t in HEAD_TEXTS.items():
        open(p, "w").write(t)
        assert blob(p) == HEAD_BLOBS[p], "restore failed " + p


only = re.compile(os.environ.get("ARMS_ONLY", ".*"))
print(f"ARMS-HEAD rope_blob={HEAD_BLOBS['mlx_lm/models/rope_utils.py']} sha={sh(['git','rev-parse','HEAD']).stdout.strip()} server_blob={HEAD_BLOB}", flush=True)
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
    path, tests = META.get(aid, (SRC_FILE, None))
    text = apply(HEAD_TEXTS[path], old, new, count)
    if aid == "A05-single-call-after-ctx":
        text = apply(text, "            rqueue.put(ctx)\n\n            # Seed if requested", "            rqueue.put(ctx)\n            self._check_prompt_length(prompt)\n\n            # Seed if requested", 1)
    open(path, "w").write(text)
    nfiles = len([l for l in sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.splitlines() if l.strip()])
    print(f"ARM id={aid} why={why} file={path} before={HEAD_BLOBS[path]} after={blob(path)} changed_files={nfiles}", flush=True)
    assert blob(path) != HEAD_BLOBS[path]
    if tests:
        rc, ran, red, secs = run_tests(aid[:3], tests[1], os.path.basename(tests[0]))
    else:
        rc, ran, red, secs = run_tests(aid[:3])
    nrun += 1
    nred += 1 if red else 0
    print(f"ARMRESULT id={aid} rc={rc} ran={ran} red={len(red)} secs={secs}", flush=True)
    for n, c in red:
        print(f"  RED {aid} {c}.{n}", flush=True)
    restore()
    print(f"RESTORED id={aid} blob={blob(path)} status_porcelain={len(sh(['git','status','--porcelain','--untracked-files=no']).stdout.splitlines())}", flush=True)
print(f"ARMS-SUMMARY run={nrun} red={nred}", flush=True)
