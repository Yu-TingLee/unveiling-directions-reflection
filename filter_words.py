import argparse
import copy
import json
import os
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def check_vocab(check_list, w):
    cond0 = re.match('^[a-zA-Z]+$', w) is not None
    cond1 = w not in check_list
    cond2 = 0 not in [(w not in x) for x in check_list]
    cond3 = 0 not in [(x not in w) for x in check_list]
    return cond0 and cond1 and cond2 and cond3

def normalize_word(ps, wnl, w):
    w = ps.stem(w)
    w = wnl.lemmatize(w, pos="v")
    return w


def run(args):
    word_limit = args.word_limit
    input_file=f"{args.input_dir}/result_layers.json"
    output_dir = os.path.dirname(input_file)
    layer_embeds = json.load(open(input_file,"r"))
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    wait_token_1s = list(args.exclude_words)
    for w in copy.deepcopy(wait_token_1s):
        w2 = normalize_word(ps, wnl, w)
        wait_token_1s.append(w2)

    for layer in layer_embeds.keys():
        check_list = copy.deepcopy(wait_token_1s)

        all_words = [x[0] for x in layer_embeds[layer]]
        selected_word = []
        for w in all_words:
            if len(selected_word) >= word_limit:
                break
            w2 = normalize_word(ps, wnl, w)
            if check_vocab(check_list, w) and check_vocab(check_list, w2):
                selected_word.append(w)
            check_list.append(w)
            check_list.append(w2)


        open(os.path.join(output_dir,f"word_{layer}.txt"),"w").write("\n".join([x for x in selected_word]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="visualize/gsm8k_adv/step2/steer_200_20")
    parser.add_argument("--word_limit", type=int, default=20)
    parser.add_argument("--exclude_words", nargs="+", default=["Wait", "Alternatively", "Check"],
                        help="Seed words to exclude from the selected word list")
    args = parser.parse_args()
    run(args)