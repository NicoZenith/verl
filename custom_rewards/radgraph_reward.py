from radgraph import F1RadGraph
import re

f1radgraph = F1RadGraph(reward_level="partial")

def radgraph_partial_score(pred_text: str, true_text: str) -> float:
    _, rewards, _, _ = f1radgraph(hyps=[pred_text], refs=[true_text])
    return rewards[0]

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Params follow veRL's RewardManager contract.
    `solution_str`  – detokenized LLM output for one sample
    `ground_truth`  – whatever we stored in reward_model['ground_truth']
    """
    # strip reasoning, keep content inside <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
    if m:
        core_answer = m.group(1)
        return radgraph_partial_score(core_answer, ground_truth)
    else:
        return 0.0