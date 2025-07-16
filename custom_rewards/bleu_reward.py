from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

def compute_score(solution_str, ground_truth):
    """
    Params follow veRL's RewardManager contract.
    `solution_str`  – detokenized LLM output for one sample
    `ground_truth`  – whatever we stored in reward_model['ground_truth']
    """
    # strip reasoning, keep content inside <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
    if m:
        core_answer = m.group(1)
        return bleu_score(core_answer, ground_truth)
    else:
        return 0.0
    

def bleu_score(candidate: str, reference: str) -> float:
    """
    Compute sentence-level BLEU between a candidate and a single reference,
    returning a float in [0, 1].
    """
    smooth = SmoothingFunction().method4
    return sentence_bleu(
        [reference.split()],      # list of reference token lists
        candidate.split(),        # candidate token list
        smoothing_function=smooth
    )

