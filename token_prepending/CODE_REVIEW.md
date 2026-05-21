# In-Depth Code Review: Token Prepending Paper Code

## Scope

Reviewed the current `token_prepending` implementation against the paper idea:

- prompt-based sentence embedding extraction from decoder-only LLMs;
- Token Prepending (`<PST>`) across an early layer window;
- early-exit embedding from an intermediate hidden state;
- SentEval and Vietnamese STS evaluation paths;
- Kaggle dual T4 execution constraints.

## Architecture Summary

The code is organized around `evaluate.py` as the runner. It loads a tokenizer, injects `<PST>` for TP mode, loads a custom model class from `senllm`, sets TP layer bounds, encodes prompted sentences, then evaluates cosine similarity with Spearman correlation.

The custom model forks in `senllm/modeling_qwen2.py`, `senllm/modeling_llama.py`, and `senllm/modeling_gemma2.py` implement the paper's core mechanism. During layers `[tp_starting_index, tp_exiting_index)`, the hidden state of the last token from the previous layer is copied into the `<PST>` token position before the next decoder layer. This matches the paper's intended causal-attention workaround at a high level.

## Findings

### High

1. `--tensor_parallel` bypasses the TP implementation.

   In `evaluate.py`, the tensor-parallel branch loads `AutoModelForCausalLM` and wraps it with `tensor_parallel`, but it does not load `LlamaForCausalLM`, `Qwen2ForCausalLM`, or `Gemma2ForCausalLM`. That means `--use_which_plan tp --tensor_parallel` will not execute the custom Token Prepending forward path. For Kaggle T4 x2, use the non-`tensor_parallel` path with `device_map="auto"` so the custom model is split by Accelerate across both GPUs.

2. Dependency pins are not Kaggle-safe.

   `requirements.txt` pins `numpy==1.21.6`, which is incompatible with newer Python runtimes commonly used by Kaggle. It also includes old or unused packages for the default evaluation path (`bitsandbytes==0.39.0`, `gradio`, `tensor-parallel`). The Kaggle notebook should install a minimal, compatible runtime instead of blindly installing the full requirements file.

3. English SentEval is not self-contained.

   The code imports SentEval, but English STS/transfer tasks still require `SentEval/data`. The repo does not include that data. A Kaggle run without attached/downloaded SentEval data should default to `vi-sts` or fail clearly for English task sets.

### Medium

4. Hidden-state memory is high on T4 x2.

   `evaluate.py` requests `output_hidden_states=True` and then indexes `hidden_states[output_layer]`. This returns all layer hidden states, not only the target layer. On 2 x T4, a 7B model should use `batch_size=1`, `float16`, and an intermediate/negative `output_layer` such as `-2` unless the model-specific layer choice is known.

5. Hardware cannot be forced by an `.ipynb` alone.

   Kaggle hardware selection lives in notebook settings or kernel metadata. The notebook can set `CUDA_VISIBLE_DEVICES=0,1` and assert two CUDA devices, but the user still must select `Accelerator -> GPU T4 x2` in Kaggle.

6. `Gemma2Model` defaults to TP mode.

   `Gemma2Model.__init__` sets `self.plan = 'tp'`, while Qwen2/Llama default to `vanilla`. `evaluate.py` overwrites this, so the main path is okay. Direct model use outside `evaluate.py` is less predictable.

### Fixed During Review

7. Removed unused hardcoded token assertions.

   The model forwards computed `first_token_indices` from hardcoded tokenizer IDs (`6025`, `1`, `2`) but never used the values. Those asserts could crash valid prompts/tokenizers for no algorithmic reason. The unused lines were removed from:

   - `senllm/modeling_qwen2.py`
   - `senllm/modeling_llama.py`
   - `senllm/modeling_gemma2.py`

## Kaggle Notebook Decisions

- The notebook uses `Qwen/Qwen2.5-7B` by default because the repo already targets this as the Mac/Kaggle-friendly default.
- It uses TP mode with `device_map="auto"` via the existing `model_runtime.py` CUDA path.
- It defaults to Vietnamese STS from Hugging Face (`nemixo/stsbenchmark-sts-vietnamese`) because this avoids missing SentEval data.
- It does not use `--tensor_parallel`, because that path is incompatible with the custom TP model classes.
- It installs a minimal Kaggle runtime instead of `requirements.txt`.

## Verification

Current local unit test command:

```text
python -m unittest discover token_prepending/tests
```

Result in this workspace: failed before running all tests because the active Python 3.13 environment does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

The Vietnamese STS helper tests that do not require `torch` were discovered before the failure. Re-run the full test suite after activating the project environment and installing the runtime dependencies.

The full model evaluation was not run locally because it requires downloading/loading a 7B model and Kaggle-class GPU memory.
