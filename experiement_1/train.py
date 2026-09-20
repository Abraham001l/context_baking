import os
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, TaskType
from model import context_baking_model

# =====================================================================
# 1. Evaluation Utilities
# =====================================================================
def get_first_token_id(tokenizer, text):
    """Encodes a target word, accounting for the leading space in BPE."""
    tokens = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
    if not tokens:
        tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
    return tokens[0]

def probe_probabilities(model, tokenizer, prompt, target_true, target_new):
    """Queries the model and returns probabilities for target_true and target_new."""
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    true_id = get_first_token_id(tokenizer, target_true)
    new_id = get_first_token_id(tokenizer, target_new)
    
    with torch.no_grad():
        outputs = model.peft_model(**prompt_inputs)
        
    last_token_logits = outputs.logits[0, -1, :]
    probs = F.softmax(last_token_logits, dim=-1)
    
    return probs[true_id].item(), probs[new_id].item()

def run_evaluation_suite(model, tokenizer, fact_record):
    """Evaluates Efficacy, Generalization, and Locality (Base & Plus)."""
    base_prompt = fact_record['prompt']
    target_true = fact_record['target_true']
    target_new = fact_record['target_new']
    edit_statement = fact_record['edit_statement']
    rephrase_prompts = fact_record.get('rephrase_prompts', [])
    neighborhood_prompts = fact_record.get('neighborhood_prompts', [])
    
    results = {}
    
    # 1. Efficacy (Without Context)
    p_true, p_new = probe_probabilities(model, tokenizer, base_prompt, target_true, target_new)
    results['efficacy_prob_true'] = p_true
    results['efficacy_prob_new'] = p_new
    results['efficacy_success'] = int(p_new > p_true)
    
    # 2. Efficacy (With Context)
    ctx_prompt = f"{edit_statement} {base_prompt}"
    cp_true, cp_new = probe_probabilities(model, tokenizer, ctx_prompt, target_true, target_new)
    results['efficacy_with_context_prob_true'] = cp_true
    results['efficacy_with_context_prob_new'] = cp_new
    results['efficacy_with_context_success'] = int(cp_new > cp_true)
    
    # 3. Generalization (Without Context)
    gen_succ = 0
    for rp in rephrase_prompts:
        pt, pn = probe_probabilities(model, tokenizer, rp, target_true, target_new)
        if pn > pt:
            gen_succ += 1
    results['generalization_score'] = gen_succ / len(rephrase_prompts) if rephrase_prompts else 0.0
    
    # 4. Generalization (With Context)
    gen_ctx_succ = 0
    for rp in rephrase_prompts:
        pt, pn = probe_probabilities(model, tokenizer, f"{edit_statement} {rp}", target_true, target_new)
        if pn > pt:
            gen_ctx_succ += 1
    results['generalization_with_context_score'] = gen_ctx_succ / len(rephrase_prompts) if rephrase_prompts else 0.0
    
    # 5. Locality Base (Unrelated facts: target_true should remain dominant)
    loc_succ = 0
    for np in neighborhood_prompts:
        pt, pn = probe_probabilities(model, tokenizer, np, target_true, target_new)
        if pt > pn:
            loc_succ += 1
    results['locality_base_score'] = loc_succ / len(neighborhood_prompts) if neighborhood_prompts else 0.0
    
    # 6. Locality Plus (Distracted: edit prepended to unrelated facts)
    loc_plus_succ = 0
    for np in neighborhood_prompts:
        pt, pn = probe_probabilities(model, tokenizer, f"{edit_statement} {np}", target_true, target_new)
        if pt > pn:
            loc_plus_succ += 1
    results['locality_plus_score'] = loc_plus_succ / len(neighborhood_prompts) if neighborhood_prompts else 0.0
    
    return results


# =====================================================================
# 2. Rolling Outlier Storage Tracker
# =====================================================================
class outlier_tracker:
    def __init__(self, save_dir="./models/saved_adapters"):
        self.save_dir = save_dir
        self.top_10 = []     # List of (score, path)
        self.bottom_10 = []
        self.middle_10 = []
        os.makedirs(save_dir, exist_ok=True)

    def process_and_save(self, model, fact_record, efficacy_score, full_metrics):
        case_id = fact_record["case_id"]
        temp_dir = os.path.join(self.save_dir, f"temp_{case_id}")
        
        # Save only the ~10MB LoRA adapter weights
        model.peft_model.save_pretrained(temp_dir, selected_adapters=["current_fact"])
        
        # Save complete provenance metadata
        metadata = {
            "case_id": case_id,
            "subject": fact_record.get("subject"),
            "relation_id": fact_record.get("relation_id"),
            "prompt": fact_record.get("prompt"),
            "target_true": fact_record.get("target_true"),
            "target_new": fact_record.get("target_new"),
            "edit_statement": fact_record.get("edit_statement"),
            "metrics": full_metrics
        }
        with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 1. Top 10 Evaluation
        if len(self.top_10) < 10 or efficacy_score > min(self.top_10, key=lambda x: x[0])[0]:
            self._insert_and_prune(self.top_10, temp_dir, case_id, efficacy_score, "top", reverse=True)
        # 2. Bottom 10 Evaluation
        elif len(self.bottom_10) < 10 or efficacy_score < max(self.bottom_10, key=lambda x: x[0])[0]:
            self._insert_and_prune(self.bottom_10, temp_dir, case_id, efficacy_score, "bottom", reverse=False)
        # 3. Middle 10 Evaluation (closest to 0.5)
        else:
            dist = abs(efficacy_score - 0.5)
            max_dist = max([abs(s - 0.5) for s, _ in self.middle_10]) if self.middle_10 else float('inf')
            if len(self.middle_10) < 10 or dist < max_dist:
                target = os.path.join(self.save_dir, "middle", f"fact_{case_id}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copytree(temp_dir, target)
                self.middle_10.append((efficacy_score, target))
                self.middle_10.sort(key=lambda x: abs(x[0] - 0.5))
                if len(self.middle_10) > 10:
                    _, prune_path = self.middle_10.pop(-1)
                    shutil.rmtree(prune_path)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def _insert_and_prune(self, tracker_list, temp_dir, case_id, score, bucket, reverse):
        target = os.path.join(self.save_dir, bucket, f"fact_{case_id}")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copytree(temp_dir, target)
        tracker_list.append((score, target))
        tracker_list.sort(key=lambda x: x[0], reverse=reverse)
        if len(tracker_list) > 10:
            _, prune_path = tracker_list.pop(-1)
            shutil.rmtree(prune_path)


# =====================================================================
# 3. Data Batching (128 Train / 32 Validation)
# =====================================================================
def load_and_batch_carrier_queries(tokenizer, wiki_path="data/wiki_sentences.json", batch_size=32):
    with open(wiki_path, "r", encoding="utf-8") as f:
        sentences = json.load(f)
        
    train_texts = sentences[:128] # 4 batches of 32
    val_texts = sentences[128:160] # 1 batch of 32
    
    # Pre-tokenize static student queries to save GPU runtime
    train_batches = []
    for i in range(0, 128, batch_size):
        b_texts = train_texts[i:i+batch_size]
        student_inputs = tokenizer(
            b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to("cuda")
        train_batches.append((b_texts, student_inputs))
        
    student_val_inputs = tokenizer(
        val_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to("cuda")
    val_batch = (val_texts, student_val_inputs)
    
    return train_batches, val_batch


# =====================================================================
# 4. Core Training and Evaluation Routine
# =====================================================================
def train_and_evaluate_fact(model, fact_record, train_batches, val_batch, peft_config):
    case_id = fact_record["case_id"]
    edit_statement = fact_record["edit_statement"]
    
    # --- Step A: Pre-Edit Baseline ---
    # Probe probabilities on base weights before injecting adapter
    base_p_true, base_p_new = probe_probabilities(
        model, model.tokenizer, fact_record['prompt'], 
        fact_record['target_true'], fact_record['target_new']
    )
    
    # --- Step B: Inject Blank LoRA Adapter ---
    model.peft_model.add_adapter("current_fact", peft_config)
    model.peft_model.set_adapter("current_fact")
    
    optimizer = torch.optim.AdamW(model.peft_model.parameters(), lr=5e-5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    loss_curve = []
    
    # --- Step C: Training Loop with Validation Early Stopping ---
    for step in range(1, 101):
        model.peft_model.train()
        step_train_loss = 0.0
        
        # Train on the 4 batches of 32
        for batch_texts, student_inputs in train_batches:
            teacher_texts = [f"{edit_statement} {t}" for t in batch_texts]
            teacher_inputs = model.tokenizer(
                teacher_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to("cuda")
            
            optimizer.zero_grad()
            loss = model.compute_masked_kl_loss(teacher_inputs, student_inputs)
            loss.backward()
            optimizer.step()
            step_train_loss += loss.item()
            
        avg_train_loss = step_train_loss / len(train_batches)
        
        # Validation on held-out 32 queries
        model.peft_model.eval()
        with torch.no_grad():
            val_texts, student_val_inputs = val_batch
            teacher_val_texts = [f"{edit_statement} {t}" for t in val_texts]
            teacher_val_inputs = model.tokenizer(
                teacher_val_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to("cuda")
            val_loss = model.compute_masked_kl_loss(teacher_val_inputs, student_val_inputs).item()
            
        loss_curve.append({
            "step": step, 
            "train_loss": avg_train_loss, 
            "val_loss": val_loss
        })
        
        # Check plateau
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {
                k: v.cpu().clone() 
                for k, v in get_peft_model_state_dict(model.peft_model).items()
            }
        else:
            patience_counter += 1
            
        if patience_counter >= 3:
            break
            
    # --- Step D: Rollback to Optimal Weights ---
    set_peft_model_state_dict(model.peft_model, best_weights)
    
    # --- Step E: Post-Edit Evaluation ---
    model.peft_model.eval()
    eval_metrics = run_evaluation_suite(model, model.tokenizer, fact_record)
    
    # Assemble full record
    full_metrics = {
        "case_id": case_id,
        "pre_edit_target_true_prob": base_p_true,
        "pre_edit_target_new_prob": base_p_new,
        "pre_edit_success": int(base_p_true > base_p_new),
        "steps_to_converge": len(loss_curve),
        "best_val_loss": best_val_loss,
        "loss_curve": loss_curve,
        **eval_metrics
    }
    
    return full_metrics


# =====================================================================
# 5. Main Orchestrator
# =====================================================================
def main():
    model_path = "./models/Qwen2.5-1.5B"
    wiki_path = "data/wiki_sentences.json"
    dataset_path = "data/counterfact_plus_static.json"
    output_log_path = "results_log.jsonl"
    
    # 1. Initialize Base Model and PEFT container
    model = context_baking_model(model_path=model_path)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    # Wrap base model in PEFT
    model.peft_model = get_peft_model(model.base_model, peft_config)
    
    # 2. Setup Carrier Queries and Tracker
    train_batches, val_batch = load_and_batch_carrier_queries(model.tokenizer, wiki_path)
    tracker = outlier_tracker(save_dir="./models/saved_adapters")
    
    # 3. Load CounterFact Records
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:5]  # Standard evaluation subset
        
    print(f"Beginning training loop across {len(dataset)} isolated facts...")
    
    # 4. Master Loop
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        for idx, fact in enumerate(dataset, 1):
            print(f"[{idx}/{len(dataset)}] Case {fact['case_id']}: '{fact['edit_statement']}'")
            
            full_metrics = train_and_evaluate_fact(
                model, fact, train_batches, val_batch, peft_config
            )
            
            # Flush log to disk immediately
            log_file.write(json.dumps(full_metrics) + "\n")
            log_file.flush()
            
            # Save or prune adapters based on efficacy
            tracker.process_and_save(
                model, fact, full_metrics["efficacy_prob_new"], full_metrics
            )
            
            # Cleanly delete adapter from memory, resetting to base model
            model.peft_model.delete_adapter("current_fact")

    print(f"Completed run. Full logs saved to {output_log_path}.")

if __name__ == "__main__":
    main()