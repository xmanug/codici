##bisogna caricare la textgrid, il file .txt con la trascrizione e avere un token di HF


from huggingface_hub import notebook_login
notebook_login()
!pip install -q -U bitsandbytes
!pip install -q -U accelerate



!pip install -q -U bitsandbytes
!pip install -q -U accelerate
!pip install -q textgrid

import os
import re
import torch
import transformers
import textgrid
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from google.colab import files
from huggingface_hub import notebook_login
from google.colab import drive



drive.mount('/content/drive', force_remount=True)

# !!! CAMBIA QUI !!!
original_textgrid_filename = '/content/drive/MyDrive/modo_frog/PS01_prominenze.TextGrid'
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


def fix_textgrid_overlaps(input_file, output_file, tolerance=0.001):
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
            last_xmax = -1.0
            for line in fin:
                stripped_line = line.strip()
                # La logica di correzione si applica riga per riga
                if stripped_line.startswith("xmin ="):
                    current_xmin = float(stripped_line.split('=')[1].strip())
                    if last_xmax > 0 and current_xmin < last_xmax and (last_xmax - current_xmin) < tolerance:
                        indentation = line.split('xmin')[0]
                        line = f"{indentation}xmin = {last_xmax:.10f}\n"
                elif stripped_line.startswith("xmax ="):
                    last_xmax = float(stripped_line.split('=')[1].strip())
                fout.write(line)

        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        return False

def normalize_word(text):
    return ''.join(char for char in text.lower() if char.isalpha())

def process_textgrid_and_compute_surprisal(filename, model_name):
    tg = textgrid.TextGrid.fromFile(filename)
    ort_tier = tg.getFirst('ort')

    intervals_to_process, words_for_surprisal = [], []
    for interval in ort_tier:
        word = interval.mark.strip()
        if word and not word.startswith('<'):
            intervals_to_process.append(interval)
            words_for_surprisal.append(word)
    story = " ".join(words_for_surprisal)

    # Caricamento modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    model.eval()


    print("\nCalcolo della surprisal in corso...")
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = tokenizer.bos_token_id
    ids = tokenizer(story, return_tensors="pt").input_ids[0]
    if ids[0] != bos_id:
        ids = torch.cat([torch.tensor([bos_id]), ids], dim=0)

    calculated_surprisals = []
    seq_len = ids.shape[0]
    prev_end_loc = 0
    stride = 512

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + ctx_size, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = ids[begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:-trg_len] = -100
        if input_ids.shape[0] <= 1:
            continue

        print(f"  Elaborando batch di token da {begin_loc} a {end_loc}...")
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits


            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[1:].contiguous()

            active_loss = shift_labels != -100
            active_logits = shift_logits[0][active_loss]
            active_labels = shift_labels[active_loss]

            probs = softmax(active_logits)
            true_token_probs = probs[torch.arange(active_labels.shape[0]), active_labels]
            surp_tensor = -torch.log2(true_token_probs)

            calculated_surprisals.extend(zip(
                tokenizer.convert_ids_to_tokens(active_labels.cpu()),
                surp_tensor.cpu().tolist()
            ))

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

        del outputs, logits, probs
        torch.cuda.empty_cache()

    final_word_surprisals = []
    word_iterator = iter(words_for_surprisal)
    current_original_word = next(word_iterator, None)
    current_tokens, current_surprisals = [], []

    for token, surprisal in calculated_surprisals:
        current_tokens.append(token)
        current_surprisals.append(surprisal)
        reconstructed_word = tokenizer.decode(tokenizer.convert_tokens_to_ids(current_tokens)).strip()
        if current_original_word and normalize_word(reconstructed_word) == normalize_word(current_original_word):
            final_word_surprisals.append(sum(current_surprisals))
            current_original_word = next(word_iterator, None)
            current_tokens, current_surprisals = [], []

    surprisal_tier = textgrid.IntervalTier(name='surprisal', minTime=ort_tier.minTime, maxTime=ort_tier.maxTime)
    for i, interval in enumerate(intervals_to_process):
        if i < len(final_word_surprisals):
            surprisal_tier.add(interval.minTime, interval.maxTime, f"{final_word_surprisals[i]:.6f}")
    tg.append(surprisal_tier)
    output_filename = os.path.basename(filename).replace(".TextGrid", "_with_surprisal.TextGrid")
    tg.write(output_filename)
    return output_filename


corrected_filename = "corrected_" + os.path.basename(original_textgrid_filename)

if fix_textgrid_overlaps(original_textgrid_filename, corrected_filename):
    final_output_file = process_textgrid_and_compute_surprisal(corrected_filename, model_name)
    if final_output_file:
        files.download(final_output_file)
