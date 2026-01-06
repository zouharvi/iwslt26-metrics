import argparse
import collections
import json
import statistics
import numpy as np
import scipy.stats
import subset2evaluate.evaluate

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", type=str, required=True)
args.add_argument("-m", "--metric", nargs="+", type=str, required=True)
args = args.parse_args()

with open(args.input, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

for line in data:
    line["score_pred"] = {}

metrics = []
for metric in args.metric:
    with open(metric, "r") as f:
        scores = [json.loads(line) for line in f.readlines()]
    metric = metric.split("/")[-1].split(".")[0]
    metrics .append(metric)
    if len(scores) != len(data):
        raise ValueError(f"Number of lines in {metric} does not match input")
    for i, v in enumerate(scores):
        if not isinstance(v, (int, float)):
            raise ValueError(f"Line {i} in {metric} with value {v} is not a number")
    for line, score in zip(data, scores):
        line["score_pred"][metric] = score


# segment-level
def segment_level(data_lang):
    # aggregate by doc_id
    docs = collections.defaultdict(list)
    for line in data_lang:
        docs[line["doc_id"]].append(line)
    
    corrs = [
        scipy.stats.kendalltau(
            [line["score"] for line in doc],
            [line["score_pred"] for line in doc],
            variant="b",
        ).correlation
        for doc in docs.values()
        if len(doc) >= 2
    ]
    return statistics.mean([x for x in corrs if not np.isnan(x)])

# system-level
def system_level(data_lang):
    # piggy-back on top of subset2evaluate's implementation which however needs some collation
    data_coll = collections.defaultdict(list)
    for line in data_lang:
        data_coll[line["doc_id"]].append(line)
    data_coll = [
        {
            "scores": {
                line["tgt_system"]: {
                    "score": line["score"],
                    "score_pred": line["score_pred"],
                }
                for line in doc
            }
        }
        for doc in data_coll.values()
    ]
    # take rows that have all systems
    # systems that are everywhere
    systems = set.union(*[
        set(doc["scores"].keys())
        for doc in data_coll
    ])
    data_coll = [
        doc
        for doc in data_coll
        if set(doc["scores"].keys()) == systems
    ]
    for doc in data_coll:
        doc["scores"] = {
            system: doc["scores"][system]
            for system in systems
        }
    return subset2evaluate.evaluate.eval_subset_spa(
        data_coll,
        data_coll,
        metric=("score", "score_pred")
    )

results_syslevel = collections.defaultdict(dict)
results_seglevel = collections.defaultdict(dict)
for lang in set(line["src_lang"]+line["tgt_lang"] for line in data):
    data_lang = [
        line
        for line in data
        if line["src_lang"]+line["tgt_lang"] == lang
    ]
    for metric in metrics:
        data_lang_metric = [
            {**line, "score_pred": line["score_pred"][metric]}
            for line in data_lang
        ]
        results_syslevel[metric][lang] = system_level(data_lang_metric)
        results_seglevel[metric][lang] = segment_level(data_lang_metric)


results_seglevel = sorted(results_seglevel.items(), key=lambda x: x[0], reverse=True)
results_syslevel = sorted(results_syslevel.items(), key=lambda x: x[0], reverse=True)
print("\n\n")
print("SEGMENT-LEVEL")
for metric, scores in results_seglevel:
    avg_score = statistics.mean([x for x in scores.values()])
    print(f"{metric:>30}: {avg_score:<5.1%}", end=" | ")
    print(" ".join(f"{lang}:{score:<5.1%}" for lang, score in scores.items()))

print("\n\nSYSTEM-LEVEL")
for metric, scores in results_syslevel:
    avg_score = statistics.mean([x for x in scores.values()])
    print(f"{metric:>30}: {avg_score:<5.1%}", end=" | ")
    print(" ".join(f"{lang}:{score:<5.1%}" for lang, score in scores.items()))