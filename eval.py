import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from bart_score import BARTScorer
from assuredness_4_old import Assuredness
import os
import matplotlib.pyplot as plt
import torch
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
pct = 0.005
def save_llr_histograms(predictions):
    # predictions is a list of 4 lists, each with 2n items: first n are human, last n are LLM
    fig, axes = plt.subplots(2, 2, figsize=(13.65, 9.79))
    #fig.subplots_adjust(hspace=1.5)
    axes = axes.flatten()

    for i, prediction in enumerate(predictions):
        ax = axes[i]
        results = {'human-written': [], 'LLM-generated': []}

        n = int(len(prediction) / 2)
        results['human-written'] = prediction[:n]
        results['LLM-generated'] = prediction[n:]

        # Plot histograms
        ax.hist(results['human-written'], alpha=0.5, bins=50, label='Human-written')
        ax.hist(results['LLM-generated'], alpha=0.5, bins=50, label='LLM-generated')
        #ax.set_title(f"Histogram {i+1}")

        # Add legend to the second quadrant (index 0)
        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout(h_pad=4.05)
    plt.savefig(f"4panel_histograms.png")
    plt.close()
def fill_and_mask(text,  pct = pct):
    tokens = text.split(' ')

    n_spans = pct * len(tokens)
    n_spans = int(n_spans)

    repeated_random_numbers = np.random.choice(range(len(tokens)), size=n_spans)

    return repeated_random_numbers.tolist()


def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]

    for idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for idx in indices:
            text[idx] = ""

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, pct=pct):

    indices_list = [fill_and_mask(x, pct) for x in texts]
    perturbed_texts = apply_extracted_fills(texts, indices_list)

    return perturbed_texts


def perturb_texts(texts, pct=pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs
# 读取 JSON 文件
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    original_texts = data.get("original", [])
    sampled_texts = data.get("sampled", [])
    return original_texts, sampled_texts

# 主流程
def evaluate(json_path):

    original_texts, sampled_texts = load_data(json_path)
    texts = original_texts + sampled_texts
    labels = [0] * len(original_texts) + [1] * len(sampled_texts)

    perturbed_original_texts = perturb_texts([x for x in original_texts for _ in range(2)])
    perturbed_sampled_texts = perturb_texts([x for x in sampled_texts for _ in range(2)])

    perturbed_original_texts_list = []
    perturbed_sampled_texts_list = []

    for idx in range(len(original_texts)):
        perturbed_original_texts_list.append(perturbed_original_texts[idx * 2: (idx + 1) * 2])
        perturbed_sampled_texts_list.append(perturbed_sampled_texts[idx * 2: (idx + 1) * 2])

    # 检测器得分
    predictions = []
    for i in range(len(perturbed_original_texts_list)):
        score = detector(original_texts[i],perturbed_original_texts_list[i])
        predictions.append(score)
    print(np.mean(predictions))
    pp=[]
    for i in range(len(perturbed_original_texts_list)):
        score = detector(sampled_texts[i],perturbed_sampled_texts_list[i])
        predictions.append(score)
        pp.append(score)
    print(np.mean(pp))


    # 计算 AUROC
    auroc = roc_auc_score(labels, predictions)

    # 选择最优阈值计算 ACC
    best_acc = 0
    best_threshold = 0
    thresholds = predictions  # 0.00 到 1.00
    for threshold in thresholds:
        preds = [1 if s >= threshold else 0 for s in predictions]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # 输出结果
    print("数据集", json_path)
    print(f"AUROC: {auroc:.4f}")
    print(f"Best ACC: {best_acc:.4f} at threshold {best_threshold:.2f}")

    return predictions
def detector(text, copies):
    source_texts_list = [text] * 2

    # bart-score

    values = bart_scorer.score(copies, source_texts_list, batch_size=2)
    mean_values = np.mean(values)

    assuredness = assuredness_scorer.compute_crit(text)

    return assuredness*np.exp(mean_values)

if __name__ == "__main__":
    # 替换成你的 JSON 文件路径
    DEVICE = "cuda:0"
    bart_scorer = BARTScorer(device=DEVICE, checkpoint='bart-base')
    assuredness_scorer = Assuredness()
    datasets = ['exp_Open_source_model/xsum_gpt2-xl.raw_data.json',
                'exp_Open_source_model/xsum_gpt-neox-20b.raw_data.json',
                'exp_Open_source_model/xsum_gpt-neo-2.7B.raw_data.json',
                'exp_Open_source_model/xsum_gpt-j-6B.raw_data.json'

   
                'exp_API-based_model/xsum_gemini.raw_data.json',
                'exp_API-based_model/xsum_gpt-3.5-turbo.raw_data.json',
                'exp_API-based_model/xsum_gpt-4.raw_data.json',
                'exp_Open_source_model/xsum_opt-2.7b.raw_data.json',



                'exp_API-based_model/pubmed_gemini.raw_data.json',
                'exp_API-based_model/pubmed_gpt-3.5-turbo.raw_data.json',
                'exp_API-based_model/pubmed_gpt-4.raw_data.json',

                'exp_API-based_model/writing_gemini.raw_data.json',
                'exp_API-based_model/writing_gpt-3.5-turbo.raw_data.json',
                'exp_API-based_model/writing_gpt-4.raw_data.json',
                'exp_Open_source_model/writing_gpt2-xl.raw_data.json',
                'exp_Open_source_model/writing_opt-2.7b.raw_data.json',
                'exp_Open_source_model/writing_gpt-neox-20b.raw_data.json',
                'exp_Open_source_model/writing_gpt-neo-2.7B.raw_data.json',
                'exp_Open_source_model/writing_gpt-j-6B.raw_data.json',

                'exp_Open_source_model/squad_opt-2.7b.raw_data.json',
                'exp_Open_source_model/squad_gpt-neox-20b.raw_data.json',
                'exp_Open_source_model/squad_gpt-neo-2.7B.raw_data.json',
                'exp_Open_source_model/squad_gpt-j-6B.raw_data.json',
                'exp_Open_source_model/squad_gpt2-xl.raw_data.json',
                ]
    pres = []
    for dataset in datasets:
        pres.append(evaluate(dataset))
    #save_llr_histograms(pres)

