import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.metrics.distance import edit_distance

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
IMAGE_DIR = r"C:\Users\Pavaki Maithili\Downloads\Flickr8k_Dataset\Flicker8k_Dataset"

if not os.path.isdir(IMAGE_DIR):
    print("WARNING: Image folder not found.")
    print("Current path:", IMAGE_DIR)
    IMAGE_DIR = input("Paste your actual Flickr8k image folder path: ").strip().strip('"')
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError("Image folder not found. Check the path: " + IMAGE_DIR)

print("Image folder found:", IMAGE_DIR)
start_id = tokenizer.word_index.get("<start>")
end_id = tokenizer.word_index.get("<end>")

if start_id is None:
    start_id = tokenizer.word_index.get("<startseq>")

if end_id is None:
    end_id = tokenizer.word_index.get("<endseq>")

index_word = tokenizer.index_word
def sequence_to_caption(sequence):
    if sequence is None:
        return ""
    if isinstance(sequence, tuple):
        sequence = sequence[0]
    if hasattr(sequence, "numpy"):
        sequence = sequence.numpy()
    if isinstance(sequence, bytes):
        sequence = sequence.decode("utf-8", errors="ignore")
    if isinstance(sequence, str):
        return sequence
    sequence = np.asarray(sequence).reshape(-1).tolist()
    words = []
    for token in sequence:
        try:
            token_id = int(token)
        except (TypeError, ValueError):
            word = str(token).strip()
            if word in ("<start>", "<startseq>"):
                continue
            if word in ("<end>", "<endseq>"):
                break
            if word not in ("<pad>", ""):
                words.append(word)
            continue
        if token_id == 0:
            continue
        if end_id is not None and token_id == end_id:
            break
        if start_id is not None and token_id == start_id:
            continue
        word = index_word.get(token_id)
        if word and word not in ("<start>", "<startseq>", "<end>", "<endseq>", "<pad>"):
            words.append(word)
    return " ".join(words)
def clean_caption(caption):
    text = sequence_to_caption(caption).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def normalize_image_name(image_name):
    if hasattr(image_name, "numpy"):
        image_name = image_name.numpy()
    if isinstance(image_name, bytes):
        image_name = image_name.decode("utf-8", errors="ignore")
    return os.path.basename(str(image_name).strip().strip("'\""))
image_references = defaultdict(list)
duplicate_count = 0

for image_name, caption_entry in zip(img_name_val, cap_val):
    image_key = normalize_image_name(image_name)
    caption_tokens = clean_caption(caption_entry)
    caption_text = " ".join(caption_tokens)
    if not image_key or not caption_text:
        continue
    if caption_text in image_references[image_key]:
        duplicate_count += 1
    else:
        image_references[image_key].append(caption_text)

image_references = {
    image: references
    for image, references in image_references.items()
    if references
}

unique_images = list(image_references.keys())

print("\nTotal validation entries:", len(img_name_val))
print("Unique images with references:", len(unique_images))
print("Duplicate captions removed:", duplicate_count)

if not unique_images:
    raise ValueError("No reference captions found. Check img_name_val and cap_val.")
def resolve_image_path(image_name):
    if os.path.isfile(image_name):
        return image_name
    candidate = os.path.join(
        IMAGE_DIR,
        os.path.basename(image_name)
    )
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        "Image file not found: " + image_name
        + "\nExpected location: " + candidate
    )
def get_predicted_caption(image_path):
    result = evaluate(image_path)
    if isinstance(result, tuple):
        result = result[0]
    return sequence_to_caption(result)
def overlap_scores(reference_tokens, predicted_tokens):
    if not reference_tokens or not predicted_tokens:
        return 0.0, 0.0, 0.0
    reference_counts = Counter(reference_tokens)
    predicted_counts = Counter(predicted_tokens)
    common_words = sum(
        (reference_counts & predicted_counts).values())
    precision = common_words / len(predicted_tokens)
    recall = common_words / len(reference_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
def rouge_l_score(reference_tokens, predicted_tokens):
    if not reference_tokens or not predicted_tokens:
        return 0.0
    rows = len(reference_tokens) + 1
    cols = len(predicted_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if reference_tokens[i - 1] == predicted_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
def word_accuracy(reference_tokens, predicted_tokens):
    if not reference_tokens and not predicted_tokens:
        return 1.0
    max_length = max(
        len(reference_tokens),len(predicted_tokens))
    if max_length == 0:
        return 1.0
    distance = edit_distance(reference_tokens,predicted_tokens)
    return max(0.0, 1.0 - distance / max_length)
smooth = SmoothingFunction().method1
all_results = []

for number, image_name in enumerate(unique_images, start=1):
    image_path = resolve_image_path(image_name)
    predicted_caption = get_predicted_caption(image_path)
    predicted_tokens = clean_caption(predicted_caption)
    reference_texts = image_references[image_name]
    reference_tokens_list = [
        clean_caption(reference)
        for reference in reference_texts]
    reference_tokens_list = [
        reference
        for reference in reference_tokens_list
        if reference]
    if not reference_tokens_list:
        continue
    scored_references = []
    for reference_tokens in reference_tokens_list:
        _, _, reference_f1 = overlap_scores(
            reference_tokens,
            predicted_tokens)
        scored_references.append((reference_f1, reference_tokens))
    _, best_reference = max(scored_references,key=lambda item: item[0])
    precision, recall, f1 = overlap_scores(best_reference,predicted_tokens)
    bleu_scores = []
    for n in range(1, 5):
        weights = tuple(1.0 / n if index < n else 0.0
                        for index in range(4))
        score = sentence_bleu(reference_tokens_list,predicted_tokens,weights=weights,smoothing_function=smooth)
        bleu_scores.append(score)
    try:
        meteor = meteor_score(references=reference_tokens_list,hypothesis=predicted_tokens)
    except Exception as error:
        print("METEOR error for", image_name, ":", error)
        meteor = 0.0
    rouge_l = max(rouge_l_score(reference, predicted_tokens)
                  for reference in reference_tokens_list)
    word_acc = max(word_accuracy(reference, predicted_tokens)
                   for reference in reference_tokens_list)
    predicted_text = " ".join(predicted_tokens)
    exact_match = float(any(predicted_text == " ".join(reference)
                            for reference in reference_tokens_list))
    all_results.append({
        "Image": image_name,
        "Generated Caption": predicted_text,
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "METEOR": meteor,
        "ROUGE-L": rouge_l,
        "Word-level Accuracy": word_acc,
        "Exact Match Accuracy": exact_match,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1})
    if number % 100 == 0:
        print("Evaluated", number,"of", len(unique_images), "images")
if not all_results:
    raise ValueError("No images were evaluated. Check evaluate() and dataset variables.")
print("\nEvaluation completed.")
print("Images evaluated:", len(all_results))
print("\nSAMPLE CAPTION CHECK")

for result in all_results[:5]:
    print("\nImage:", result["Image"])
    print("Generated:", repr(result["Generated Caption"]))
    print("References:")
    for reference in image_references[result["Image"]]:
        print(" -", reference)
    print("Precision:", round(result["Precision"], 4))
    print("Recall:", round(result["Recall"], 4))
    print("F1-score:", round(result["F1-score"], 4))
from PIL import Image
import os
import matplotlib.pyplot as plt
for result in all_results[:5]:
    image_name = result["Image"]
    image_path = resolve_image_path(image_name)

    img = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")

    generated_caption = result["Generated Caption"]
    reference_captions = image_references[image_name]

    plt.title(
        "Generated Caption:\n"
        + generated_caption
        + "\n\nReference Caption(s):\n"
        + "\n".join(reference_captions[:3]),
        fontsize=10,
        wrap=True
    )

    plt.tight_layout()
    plt.show()
metric_names = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "Word-level Accuracy",
    "Exact Match Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

mean_metrics = {}

for metric in metric_names:
    mean_metrics[metric] = float(
        np.mean([
            result[metric]
            for result in all_results]))
print("\n" + "=" * 55)
print("FINAL AVERAGE EVALUATION METRICS")
print("=" * 55)

for metric, value in mean_metrics.items():
    print("{:<25}: {:.4f} ({:.2f}%)".format(metric,value,value * 100))
plt.figure(figsize=(13, 6))
plt.bar(list(mean_metrics.keys()),list(mean_metrics.values()))
plt.title("Average Image Captioning Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
bleu_names = [
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4"]

plt.figure(figsize=(8, 5))
plt.bar(bleu_names,[mean_metrics[name] for name in bleu_names])
plt.title("BLEU Score Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
prf_names = [
    "Precision",
    "Recall",
    "F1-score"]

plt.figure(figsize=(7, 5))
plt.bar(prf_names,[mean_metrics[name] for name in prf_names])
plt.title("Precision, Recall and F1-score")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
accuracy_names = [
    "Word-level Accuracy",
    "Exact Match Accuracy"]

plt.figure(figsize=(8, 5))
plt.bar(accuracy_names,[mean_metrics[name] for name in accuracy_names])
plt.title("Word-level vs Exact Match Accuracy")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
results_df = pd.DataFrame(all_results)

output_file = os.path.abspath("flickr8k_caption_evaluation_results.csv")

results_df.to_csv(output_file, index=False)

print("\nResults saved to:")
print(output_file)
print("\nAll evaluation steps completed.")
