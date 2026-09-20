import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class context_baking_model(nn.Module):
    def __init__(self, model_path="./models/Qwen2.5-1.5B", lora_rank=8, lora_alpha=16):
        super().__init__()

        # loading tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        # making sure we support input batches with varying scentence lengths
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load transofrmer model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # configuring & injecting LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )

        # build model with LoRA-injected architecture
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        # KL divergence loss function
        # reduction="batchmean" needed for mathematical accuracy
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    def get_teacher_logits(self, input_ids, attention_mask):
        # takes in context and has no gradient (disable LoRA)
        with self.model.disable_adapter():
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
    
    def get_student_logits(self, input_ids, attention_mask):
        # takes in no context, has gradient (with LoRA)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, teacher_inputs, student_inputs):
        # get teacher distribution (target)
        teacher_logits = self.get_teacher_logits(
            input_ids=teacher_inputs['input_ids'],
            attention_mask=teacher_inputs["attention_mask"]
        )

        # get student distribution (prediction)
        student_logits = self.get_student_logits(
            input_ids=student_inputs["input_ids"],
            attention_mask=student_inputs["attention_mask"]
        )

        # slicing off context tokens to compare only
        # tokens generated for carrier query
        seq_len = student_logits.size(1)
        teacher_logits_aligned = teacher_logits[:, -seq_len, :]

        # convert logits to log_probabilites (required for KLDivLoss)
        # log_prob avoid rounding to 0 errors
        log_probs_teacher = F.log_softmax(teacher_logits, dim=-1)
        log_probs_student = F.log_softmax(student_logits, dim=-1)

        # calculate loss
        loss = self.kl_loss_fn(log_probs_student, log_probs_teacher)

        return loss



        

        
