import argparse

from rouge.rouge import Rouge
from meteor.meteor import Meteor
import xlwt
import os


def main(hyp, ref, len):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        hypo = []
        for line in hypothesis:
            line = line.strip().split('\t')[-1]
            hypo.append(line)
        res = {k: [" ".join(v.strip().lower().split()[1:len])] for k, v in enumerate(hypo)}
    with open(ref, 'r') as r:
        references = r.readlines()
        refe=[]
        for line in references:
            line = line.strip().split('\t')[-1]
            refe.append(line)
        gts = {k: [" ".join(v.strip().lower().split()[1:])] for k, v in enumerate(refe)}



    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("ROUGe: "), score_Rouge

    return score_Meteor, score_Rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--predict_file_path", default='../PromptCS/saved_models/test_0.output', type=str)
    parser.add_argument("--ground_truth_file_path", default='../PromptCS/saved_models/test_0.gold', type=str)

    # print arguments
    args = parser.parse_args()

    main(args.predict_file_path, args.ground_truth_file_path, 64)
