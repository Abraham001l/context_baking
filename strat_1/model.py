import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class context_baking_model(nn.Module):
    def __init__(self, model_path="./models/Qwen2.5-1.5B"):
        super().__init__()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ensure padding is on the right for training
        self.tokenizer.padding_side = "right"

        # Load frozen base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Explicitly freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # self.peft_model will be injected dynamically by train.py
        self.peft_model = None

    def get_teacher_logits(self, input_ids, attention_mask):
        # Disable adapter to get the unedited base weights' reaction to the context
        with self.peft_model.disable_adapter():
            with torch.no_grad():
                outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
    
    def get_student_logits(self, input_ids, attention_mask):
        # Standard forward pass utilizing the active LoRA adapter
        outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, teacher_inputs, student_inputs):
        # 1. Get raw distributions
        teacher_logits = self.get_teacher_logits(
            input_ids=teacher_inputs['input_ids'],
            attention_mask=teacher_inputs["attention_mask"]
        )
        student_logits = self.get_student_logits(
            input_ids=student_inputs["input_ids"],
            attention_mask=student_inputs["attention_mask"]
        )

        # 2. Align sequence lengths (fixed slice syntax)
        seq_len = student_logits.size(1)
        teacher_logits_aligned = teacher_logits[:, -seq_len:, :]

        # 3. Convert to log probabilities (fixed variable reference)
        log_probs_teacher = F.log_softmax(teacher_logits_aligned, dim=-1)
        log_probs_student = F.log_softmax(student_logits, dim=-1)

        # 4. Compute token-wise KL Divergence
        kl_per_token = F.kl_div(
            log_probs_student, 
            log_probs_teacher, 
            reduction="none", 
            log_target=True
        ).sum(dim=-1) # Shape: (batch_size, seq_len)
        
        # 5. Mask out padding tokens
        mask = student_inputs['attention_mask']
        masked_kl = kl_per_token * mask
        
        # 6. Return normalized loss over valid tokens
        loss = masked_kl.sum() / mask.sum().clamp(min=1.0)

        return loss