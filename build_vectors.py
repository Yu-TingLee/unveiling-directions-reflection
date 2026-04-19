import argparse
import copy
from collections import defaultdict
import json
import os
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import doc_to_text, cosine_sim, cot_preprocess
from steering_hooks import get_layers, get_embed_tokens


def init_model_hook_hidden_states(model):
    def gen_hook_func_hidden_states(layer_idx, hidden_states):
        def get_hidden_states(module, input, output):
            hidden_states[layer_idx].append(output[0].detach().cpu().float().numpy())
        return get_hidden_states

    hidden_states = defaultdict(list)
    for k, layer in enumerate(get_layers(model)):
        layer.register_forward_hook(
            gen_hook_func_hidden_states(layer_idx=k, hidden_states=hidden_states)
        )
    return hidden_states


def init_model(args, hook_hidden=True):
    hf_kwargs = {"trust_remote_code": True}
    if getattr(args, "cache_dir", None):
        hf_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **hf_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **hf_kwargs)

    if hook_hidden:
        hidden_states = init_model_hook_hidden_states(model)
    else:
        hidden_states = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, hidden_states

def get_embed(args, model, tokenizer, json_items, wait_token_1s, wait_token_2s):

    wait_pos_1 = 2
    json_item = json_items[0]

    w1_ids = [[tokenizer.encode(cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=w1)))[-wait_pos_1]]
              for w1 in wait_token_1s]


    embed_tokens = get_embed_tokens(model)
    w1_embeds = embed_tokens(torch.Tensor(w1_ids).to(int).to(model.device)).cpu().detach().float().numpy()

    if len(wait_token_2s) > 0:
        w2_ids = [[tokenizer.encode(cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=w2)))[-wait_pos_1]]
                  for w2 in wait_token_2s]

        w2_embeds = embed_tokens(torch.Tensor(w2_ids).to(int).to(model.device)).cpu().detach().float().numpy()

        wvecs = []
        for w1_embed in w1_embeds:
            wvec = 0
            norm = 0
            for w2_embed in w2_embeds:
                norm += 1
                wvec += np.array(w1_embed[0] - w2_embed[0])
            wvecs.append(wvec/norm)
        return np.array(wvecs)
    else:
        wvecs = []
        for w1_embed in w1_embeds:
            wvec = np.array(w1_embed[0])
            wvecs.append(wvec)
        return np.array(wvecs)

def pad_left(w_ids, tokenizer):
    max_len = max([len(x) for x in w_ids])
    for i, w_id in enumerate(w_ids):
        pad_id = tokenizer.pad_token_id
        while len(w_id) < max_len:
            w_id.insert(0, pad_id)
    return w_ids


def get_forward(args, model, tokenizer, json_items, hidden_states, selected_layers, wait_token_1s, wait_token_2s):

    wait_pos_1 = 2


    limit = min(args.limit, len(json_items))

    output = defaultdict(int)
    success_sample = 0
    i = 0
    pbar = tqdm(total=limit, desc="build vectors")
    while True:
        torch.cuda.empty_cache()
        json_item = json_items[i]
        i += 1


        check_len = tokenizer.encode(cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token="None")))

        embed_w1 = defaultdict(list)

        for w1 in wait_token_1s:
            w1_ids = [tokenizer.encode(cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=w1)))]

            w1_ids = pad_left(w1_ids, tokenizer)

            w1_ids = torch.Tensor(w1_ids).to(int).to(model.device)

            model.forward(w1_ids)# .cpu().detach().numpy()

            for layer, ascore in hidden_states.items():
                if layer in selected_layers:
                    embed_w1[layer].append(ascore[0])
                ascore.clear()

        if len(wait_token_2s) > 0:

            embed_w2 = defaultdict(list)
            for w2 in wait_token_2s:

                w2_ids = [tokenizer.encode(cot_preprocess(doc_to_text(json_item, json_item["cot_messy"], wait_token=w2)))]

                w2_ids = pad_left(w2_ids, tokenizer)

                w2_ids = torch.Tensor(w2_ids).to(int).to(model.device)

                model.forward(w2_ids) #.cpu().detach().numpy()


                for layer, ascore in hidden_states.items():
                    if layer in selected_layers:
                        embed_w2[layer].append(ascore[0])
                    ascore.clear()

            for layer in selected_layers:
                wvecs = []
                for ikey in range(len(wait_token_1s)):
                    wvec = 0
                    norm = 0
                    w1_embed = embed_w1[layer][ikey]
                    for w2_embed in embed_w2[layer]:
                        norm += 1
                        wvec += np.array(w1_embed[-wait_pos_1] - w2_embed[-wait_pos_1])
                    wvecs.append(wvec/norm)
                output[layer] += np.array(wvecs)*(1/limit)
        else:
            for layer in selected_layers:
                wvecs = []
                for ikey in range(len(wait_token_1s)):
                    w1_embed = embed_w1[layer][ikey]
                    wvec = np.array(w1_embed[-wait_pos_1])
                    wvecs.append(wvec)
                output[layer] += np.array(wvecs)*(1/limit)
        success_sample += 1
        pbar.update(1)
        if success_sample >= limit:
            pbar.close()
            return output
    return output


def check_vocab(wait_token_1s, w):
    cond1 = re.match('^[A-Z][a-zA-Z]+$', w)
    cond2 = w not in wait_token_1s
    cond3 = 0 not in [(w not in x) for x in wait_token_1s]
    cond4 = 0 not in [(x not in w) for x in wait_token_1s]
    return cond1 and cond2 and cond3 and cond4

def get_embed_sim(args, model, tokenizer, hidden_states, output_dir):

    if args.is_baseline:
        selected_layers = []
    else:
        selected_layers = list(range(len(get_layers(model))))

    wait_token_1s = args.wait_token_1#["Wait", "Alternatively", "Check"]
    wait_token_2s = args.wait_token_2#[]

    batch_size = 1

    json_items = json.load(open(args.input_file,"r"))



    vec_knowns_emb = get_embed(args, model, tokenizer, json_items, wait_token_1s, wait_token_2s)
    vec_known_emb = np.average(vec_knowns_emb, axis=0)

    wait_token_1s_all = wait_token_1s + [x.upper() for x in wait_token_1s]
    all_words = [x for x in list(tokenizer.vocab.keys()) if check_vocab(wait_token_1s_all, x)]

    results = []
    wait_token_1 = []
    for i, word in tqdm(enumerate(all_words), total=len(all_words)):
        wait_token_1.append(word)
        if len(wait_token_1) < batch_size and i != len(all_words)-1:
            continue
        vec_news = get_embed(args, model, tokenizer, json_items, wait_token_1, wait_token_2s)
        for word_i, vec_new_i in zip(wait_token_1, vec_news):
            result = cosine_sim(vec_known_emb, vec_new_i)
            results.append((word_i,float(result)))
        wait_token_1.clear()

    results = sorted(results, key=lambda x:x[1], reverse=True)
    results = results[:int(len(results)/30)]
    selected_1 = copy.deepcopy(results)


    vec_known = {}
    vec_knowns = {}


    vec_known[-1] = vec_known_emb
    vec_knowns[-1] = vec_knowns_emb

    json.dump(dict((x,y.tolist()) for x,y in vec_known.items()), open(os.path.join(output_dir, "seed_avg.json"), "w"), indent=2)
    json.dump(dict((x,y.tolist()) for x,y in vec_knowns.items()), open(os.path.join(output_dir, "seed_per_vec.json"), "w"), indent=2)

    if args.output_new_vec == 0:
        results = defaultdict(list)
        results[-1] = selected_1
        selected_2 = copy.deepcopy(results)
        json.dump(selected_1, open(os.path.join(output_dir, "result_embedding.json"), "w"), indent=2)
        json.dump(selected_2, open(os.path.join(output_dir, "result_layers.json"), "w"), indent=2)
        open(os.path.join(output_dir,"word_-1.txt"),"w").write("\n".join([x[0] for x in selected_1][:10]))
        return


    vec_knowns_temp = get_forward(args, model, tokenizer, json_items, hidden_states, selected_layers, wait_token_1s, wait_token_2s)
    vec_knowns.update(vec_knowns_temp)

    for layer in vec_knowns.keys():
        vec_known[layer] = np.average(vec_knowns[layer], axis=0)

    json.dump(dict((x, y.tolist()) for x, y in vec_known.items()), open(os.path.join(output_dir, "seed_avg.json"), "w"), indent=2)
    json.dump(dict((x, y.tolist()) for x, y in vec_knowns.items()), open(os.path.join(output_dir, "seed_per_vec.json"), "w"), indent=2)

    batch_size = 1

    all_words = [x[0] for x in results]
    results = defaultdict(list)
    wait_token_1 = []
    for i, word in tqdm(enumerate(all_words), total=len(all_words)):
        wait_token_1.append(word)
        if len(wait_token_1) < batch_size and i != len(all_words)-1:
            continue
        vec_news = get_forward(args, model, tokenizer, json_items, hidden_states, selected_layers, wait_token_1, wait_token_2s)
        for layer in vec_news.keys():
            for word_i, vec_new_i in zip(wait_token_1, vec_news[layer]):
                result = cosine_sim(vec_known[layer], vec_new_i)
                results[layer].append((word_i,float(result)))
        wait_token_1.clear()

    results[-1] = selected_1
    for layer in results.keys():
        results[layer] = sorted(results[layer], key=lambda x:x[1], reverse=True)

    selected_2 = copy.deepcopy(results)
    json.dump(selected_1, open(os.path.join(output_dir, "result_embedding.json"), "w"), indent=2)
    json.dump(selected_2, open(os.path.join(output_dir, "result_layers.json"), "w"), indent=2)
    open(os.path.join(output_dir,"word_-1.txt"),"w").write("\n".join([x[0] for x in selected_1][:10]))
    for layer in results.keys():
        open(os.path.join(output_dir,f"word_{layer}.txt"),"w").write("\n".join([x[0] for x in selected_2[layer]][:10]))

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="visualize/gsm8k_adv/step0/gsm8k_adv.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--wait_token_1", type=str, default=["Wait", "Alternatively", "Check"], nargs='+')
    parser.add_argument("--wait_token_2", type=str, default=["Answer", "Result", "Output"], nargs='+')
    parser.add_argument("--output_dir", type=str, default="visualize/gsm8k_adv/step2/steer_200_20")
    parser.add_argument("--output_new_vec", type=int, default=1)
    parser.add_argument("--is_baseline", type=int, default=0)
    args = parser.parse_args()

    seed_everything(0)

    args.wait_token_2 = [x for x in args.wait_token_2 if len(x) > 0]
    print(f"wait_token_1:{args.wait_token_1}")
    print(f"wait_token_2:{args.wait_token_2}")


    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "result_layers.json")):
        print(f"Skipping {output_dir}: result_layers.json already exists.")
        return

    model, tokenizer, hidden_states = init_model(args)

    get_embed_sim(args, model, tokenizer, hidden_states, output_dir)



if __name__ == '__main__':
    run()
