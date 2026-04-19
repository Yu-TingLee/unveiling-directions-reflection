from collections import defaultdict
import difflib
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_hooks import get_layers


def cosine_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0  # Handle zero vectors to avoid division by zero

    return dot_product / (magnitude_vec1 * magnitude_vec2)


def get_error_pos_sub(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1, text2)
    result_1 = []
    result_2 = []

    matcher_result = list(matcher.get_opcodes())
    for tag, i1, i2, j1, j2 in matcher_result:
        if tag == 'replace' or tag == 'delete' or tag == 'insert':
            result_1.extend(range(i1,i2))
            result_2.extend(range(j1,j2))

    return result_1, result_2

def test_difflib():
                    #0  1  2  3  4  5  6  7  8  9
    cot_original = [1, 2, 2, 2, 3, 4, 5, 6, 7, 9]
                    #0  1  2  3  4  5  6  7  8  9  10
    cot_messy = [1, 5, 5, 5, 3, 4, 7, 6, 7, 8, 9]
    results = get_error_pos_sub(cot_original, cot_messy)
    print(results)
    pass

def doc_to_text(json_item, cot_messy, wait_token="Wait"):

    if json_item["input_type"] == "cruxeval_o":
        input_str =  f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
code and assert statement
[/PYTHON]
[THOUGHT]
your step by step thought
[/THOUGHT]
[ANSWER]
assert f(input) == output
[/ANSWER]

[PYTHON]
{json_item["code"]}
assert f({json_item["input"]}) == ??
[/PYTHON]
[THOUGHT]
{cot_messy}

""" + f" {wait_token} "

    elif json_item["input_type"] == "gsm8k":
        return ("Answer the question: \n\n"
                + json_item[ "input"]
                + "\nPlease always end your response with the final numerical answer.\n"
                + " Let’s solve this step by step … "
                + cot_messy[:cot_messy.rfind("\n####")]
                + f" {wait_token} "
                )
    else:
        input_str = ""
        assert 0
    return input_str



def cot_preprocess(cot_raw):
    cot_raw = re.sub(r'[ \t]+', ' ', cot_raw)
    cot_raw = cot_raw.replace(" \n","\n")
    cot_raw = cot_raw.replace(".\n","\n")
    cot_raw = re.sub(r'([+\-*/=]) ', r'\1', cot_raw)
    return cot_raw


def preprocess_for_error_pos(json_item, tokenizer, wait_token_1="Wait", wait_token_2="Answer", verbose=False):
    cot_wait_text    = cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=wait_token_1))
    cot_original_text = cot_preprocess(doc_to_text(json_item, json_item["cot_truth"]))
    cot_answer_text    = cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=wait_token_2))
    cot_wait_id         = tokenizer.encode(cot_wait_text)
    cot_original_id      = tokenizer.encode(cot_original_text)
    cot_answer_id  = tokenizer.encode(cot_answer_text)

    #print(f"cot_original_len:{len(cot_original_id)}")
    #print(f"cot_wait_len:{len(cot_wait_id)}")

    cot_wait_pos_error, cot_original_pos_error = get_error_pos_sub(cot_wait_id, cot_original_id)

    if verbose:
        print([(x[0],tokenizer.decode(x[1])) for x in enumerate(cot_wait_id)])
        print([(x[0], tokenizer.decode(x[1])) for x in enumerate(cot_original_id)])
    return {
        "cot_wait_text" : cot_wait_text,
        "cot_original_text": cot_original_text,
        "cot_answer_text": cot_answer_text,
        "cot_wait_id": cot_wait_id,
        "cot_original_id": cot_original_id,
        "cot_answer_id": cot_answer_id,
        "cot_wait_pos_error": cot_wait_pos_error,
        "cot_original_pos_error": cot_original_pos_error,
    }


def init_model_hook_attention_func(model):
    def gen_hook_func(layer_idx, attention_scores):
        def get_attention_scores(module, input, output):
            attention_scores[layer_idx].append(output[1])
        return get_attention_scores

    attention_scores = defaultdict(list)
    for k, layer in enumerate(get_layers(model)):
        layer.self_attn.config._attn_implementation = "eager"
        layer.self_attn.register_forward_hook(
            gen_hook_func(layer_idx=k, attention_scores=attention_scores)
        )
    return attention_scores


def init_model(args, hook_attention=True):
    hf_kwargs = {"trust_remote_code": True}
    if getattr(args, "cache_dir", None):
        hf_kwargs["cache_dir"] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, **hf_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **hf_kwargs)

    if hook_attention:
        attention_scores = init_model_hook_attention_func(model)
    else:
        attention_scores = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, attention_scores

def sort_selected_head(selected_head_dict):
    return sorted([(x[0][0], x[0][1], x[1]) for x in selected_head_dict.items()], key=lambda x: (x[2], x[0], x[1]), reverse=True)


def get_attention_matrix(layer, sample, head, attentions_all, wait_pos=5, repeat=1):
    results = attentions_all[layer][0][sample][head][-wait_pos:]
    return results


def get_iter_json_item(args, tokenizer, json_items, wait_token_1="Wait", wait_token_2="Answer"):
    for json_item in json_items[:args.limit]:

        input_processed = preprocess_for_error_pos(json_item, tokenizer,
                                                   wait_token_1=wait_token_1,
                                                   wait_token_2=wait_token_2)

        cot_wait_id = input_processed["cot_wait_id"]
        cot_answer_id = input_processed["cot_answer_id"]
        cot_wait_pos_error = input_processed["cot_wait_pos_error"]

        wait_pos = get_waitpos(cot_wait_id, tokenizer)

        for inputs_id, position_diff, cot_answer in zip([cot_wait_id, cot_answer_id], [cot_wait_pos_error, cot_wait_pos_error], [0, 1]):
            yield inputs_id, position_diff, cot_answer, wait_pos, json_item



def get_waitpos(inputs_id, tokenizer):
    wait_pos = 2
    return wait_pos

if __name__ == '__main__':
    test_difflib()
