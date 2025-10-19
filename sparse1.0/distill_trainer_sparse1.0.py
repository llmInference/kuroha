import transformers
import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from specInfer.generator import Generator
from specInfer.common import pad_to_2d, sychronize_time
from enum import Enum
import random
from torch.utils.data import DataLoader

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SampleSource(Enum):
    Student = 1
    Teacher = 2
    MixRequest = 3
    MixToken = 4


SAMPLE_SOURCE_MAP = {
    "student": SampleSource.Student,
    "teacher": SampleSource.Teacher,
    "mix_request": SampleSource.MixRequest,
    "mix_token": SampleSource.MixToken,
}


class KLMethod(Enum):
    Forward = 1
    Reverse = 2
    JSD = 3


KL_METHOD_MAP = {
    "forward": KLMethod.Forward,
    "reverse": KLMethod.Reverse,
    "jsd": KLMethod.JSD
}

eval_cnt = 0
copy_model = transformers.AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-160m")
copy_model.cuda()


class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model
        self.generator = Generator(
            self.model, self.teacher_model, self.tokenizer, args.max_propose_num, False
        )
        self.train_step_cnt = 0

        # online related params
        self.mode = args.mode
        self.online_eval_interval = args.online_eval_interval
        self.online_update_interval = args.online_update_interval
        self.buffer = []
        self.alphas = []
        self.alphas_by_dataset = {}
        self.alphas_by_language = {}
        self.alphas_by_topic = {}
        self.sample_steps = []

        self.sample_source = SAMPLE_SOURCE_MAP[args.sample_source]
        self.kl_method = KL_METHOD_MAP[args.kl_method]

        # 稀疏 teacher logits 的开关与超参
        self.use_sparse_teacher_logits = True
        self.sparse_k = 12
        self.fill_neg_inf = -1e9  # 近似 -inf 的大负值

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        if self.mode == "offline":
            return self.offline_training_step(model, inputs)
        elif self.mode == "online":
            return self.online_training_step(model, inputs)
        else:
            raise ValueError()

    def online_training_step(self, model, inputs):
        max_new_tokens = 128
        bsz = inputs["input_ids"].shape[0]
        assert (
                bsz == 1
        ), f"Does not support batch size > 1 in online setting, input batch size: {bsz}"
        assert (
                self.args.gradient_accumulation_steps == 1
        ), f"Does not support grad_acc > 1 in online setting, grad_acc: {self.args.gradient_accumulation_steps}"

        # ----------------- Robust attention_mask handling -----------------
        # inputs["attention_mask"] 常见形状为 (1, seq_len) 或 (seq_len,) 等
        attn_mask = inputs["attention_mask"]
        # 把 mask 规范为一维布尔向量 flat_mask，表示第 0 个样本的有效 token 位置
        if attn_mask.dim() == 2 and attn_mask.shape[0] == 1:
            flat_mask = attn_mask[0].bool()
        elif attn_mask.dim() == 1:
            flat_mask = attn_mask.bool()
        else:
            # 万一出现其他形状，尝试拉平并取第一个样本
            flat_mask = attn_mask.view(bsz, -1)[0].bool()

        # 若全部是 padding（没有有效 token），跳过本 step，避免传入空 input 触发模型内部 reshape 错误
        if flat_mask.sum().item() == 0:
            # 返回一个 0.0 损失张量（放在模型所在设备上）
            # 使用 eval 模式以与原逻辑保持一致
            self.model.eval()
            return torch.tensor(0.0, device=model.device)

        # 根据 mask 选出非 padding tokens，并恢复 batch 维
        # inputs["input_ids"] 形状 (1, seq_len)
        input_ids = inputs["input_ids"][0, flat_mask].unsqueeze(0)

        # ----------------- 继续原来的 speculative decoding -----------------
        # use speculative decoding to generate tokens
        output = self.generator.generate(
            input_ids,
            max_new_tokens,
            attention_mask=torch.ones_like(input_ids),
        )

        token_ids = torch.cat([input_ids, output.generated_ids], dim=-1)
        wrong_token_ids = [input_ids.shape[-1] + t for t in output.wrong_token_ids]

        if "dataset" in inputs:
            dataset = inputs["dataset"][0]
            if dataset not in self.alphas_by_dataset:
                self.alphas_by_dataset[dataset] = []
            if self.train_step_cnt <= 2000:
                if dataset == "gsm8k":
                    self.buffer.append((token_ids, wrong_token_ids))
            else:
                if dataset == "finance":
                    self.buffer.append((token_ids, wrong_token_ids))
        else:
            self.buffer.append((token_ids, wrong_token_ids))

        self.alphas.append(output.alpha_sum)
        self.sample_steps.append(output.sample_steps)
        if "dataset" in inputs:
            self.alphas_by_dataset[dataset].append(
                output.alpha_sum * 1.0 / output.sample_steps
            )
        if "language" in inputs:
            language = inputs["language"][0]
            if language not in self.alphas_by_language:
                self.alphas_by_language[language] = []
            self.alphas_by_language[language].append(
                output.alpha_sum * 1.0 / output.sample_steps
            )
        if "topic" in inputs:
            topic = inputs["topic"][0]
            if topic not in self.alphas_by_topic:
                self.alphas_by_topic[topic] = []
            self.alphas_by_topic[topic].append(
                output.alpha_sum * 1.0 / output.sample_steps
            )

        if self.train_step_cnt % self.online_eval_interval == 0:
            window_size = 1
            avg_alpha = sum(self.alphas[-window_size:]) * 1.0 / sum(
                self.sample_steps[-window_size:]
            )
            wandb.log({"alpha": avg_alpha})
            if "dataset" in inputs:
                wandb.log({f"alpha_{dataset}": self.alphas_by_dataset[dataset][-1]})
            if "language" in inputs:
                language_alpha = self.alphas_by_language[language][-1]
                wandb.log({f"alpha_{language}": language_alpha})
            if "topic" in inputs:
                topic_alpha = self.alphas_by_topic[topic][-1]
                wandb.log({f"alpha_{topic}": topic_alpha})

        if len(self.buffer) >= self.online_update_interval:
            self.model.train()

            input_ids = pad_to_2d([x[0] for x in self.buffer], self.tokenizer.pad_token_id)
            attn_mask = torch.ones_like(input_ids)

            student_logits = self.get_logits(model, input_ids, attn_mask).float()
            with torch.no_grad():
                teacher_logits = self.get_logits(self.teacher_model, input_ids, attn_mask).float()

            # 稀疏 Logit 采样 + 局部归一化（仅当开关开启）
            # 目标：无偏（importance / multinomial）近似 teacher 分布
            if self.use_sparse_teacher_logits:
                # 形状: [B, T, V]
                bsz2, seq_len, vocab_size = teacher_logits.shape

                # 在每个 (b, t) 位置上，从 teacher 的概率分布按权重随机采样 k 个不重复的词
                teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
                flat_probs = teacher_probs.view(-1, vocab_size)  # [B*T, V]

                k = min(self.sparse_k, vocab_size)
                try:
                    sampled_indices = torch.multinomial(flat_probs, num_samples=k, replacement=False)
                except RuntimeError:
                    sampled_indices = torch.multinomial(flat_probs, num_samples=k, replacement=True)

                flat_teacher_logits = teacher_logits.view(-1, vocab_size)  # [B*T, V]
                gathered_logits = flat_teacher_logits.gather(1, sampled_indices)  # [B*T, k]

                sparse_flat = torch.full_like(flat_teacher_logits, self.fill_neg_inf)
                sparse_flat.scatter_(1, sampled_indices, gathered_logits)
                sparse_teacher_logits = sparse_flat.view_as(teacher_logits)
                targets_for_loss = sparse_teacher_logits
            else:
                targets_for_loss = teacher_logits

            # 仅在 wrong predictions 位置参与损失
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            for i, data in enumerate(self.buffer):
                cur_wrong_token_ids = data[1]
                mask[i, cur_wrong_token_ids] = False

            loss = self.soft_cross_entropy(student_logits, targets_for_loss, mask)
            loss.backward()
            self.buffer = []
            return loss.detach()
        else:
            self.model.eval()
            return torch.tensor(-1).cuda()

    def offline_training_step(self, model, inputs):
        max_new_tokens = 128
        student_temperature = 1.0
        teacher_temperature = 1.0
        debug = False
        if debug:
            step_start_time = sychronize_time()

        if self.sample_source == SampleSource.MixRequest:
            student_request_ratio = 0.5

        if self.sample_source == SampleSource.MixToken:
            student_token_ratio = 0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.9

        sample_mix_token = False

        # sample tokens
        if self.sample_source == SampleSource.Teacher:
            sample_student = False
        elif self.sample_source == SampleSource.Student:
            sample_student = True
        elif self.sample_source == SampleSource.MixRequest:
            sample_student = True if random.random() < student_request_ratio else False
        elif self.sample_source == SampleSource.MixToken:
            sample_mix_token = True

        if debug:
            sample_time_start = sychronize_time()

        if sample_mix_token:
            copy_model.load_state_dict(self.remove_module_prefix(model.state_dict()))
            generated_ids = self.get_mix_generated_ids(
                copy_model,
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                student_token_ratio,
            )
            generated_ids = generated_ids.clone().detach()
        elif sample_student:
            copy_model.load_state_dict(self.remove_module_prefix(model.state_dict()))
            generated_ids, _ = self.get_generated_ids(
                copy_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
            generated_ids = generated_ids.clone().detach()
        else:
            generated_ids = inputs["input_ids"]
        if debug:
            print(f"Sample time: {sychronize_time() - sample_time_start}")

        if debug:
            prepare_time_start = sychronize_time()

        # prepare masks
        if sample_mix_token or sample_student:
            prompt_len = inputs["prompt_ids"].shape[-1]
            attention_mask = generated_ids != self.tokenizer.pad_token_id
            output_mask = generated_ids[..., 1:] == self.tokenizer.pad_token_id
            # Ignore prompt when calculating loss
            output_mask[..., :prompt_len - 1] = True
        else:
            attention_mask = inputs["attention_mask"]
            output_mask = inputs["labels"][..., 1:] == IGNORE_TOKEN_ID
        if debug:
            print(f"Prepare time: {sychronize_time() - prepare_time_start}")

        # logits
        student_logits = self.get_logits(model, generated_ids, attention_mask)
        with torch.no_grad():
            teacher_logits = self.get_logits(self.teacher_model, generated_ids, attention_mask)
        student_logits = student_logits[..., :-1, :].float()
        teacher_logits = teacher_logits[..., :-1, :].float()

        # loss
        if self.kl_method == KLMethod.Forward:
            loss = self.soft_cross_entropy(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask,
            )
        elif self.kl_method == KLMethod.Reverse:
            loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask,
            )
        elif self.kl_method == KLMethod.JSD:
            reverse_loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask,
            )
            fwd_loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask,
            )
            fwd_loss_ratio = 0.9
            loss = fwd_loss_ratio * fwd_loss + (1 - fwd_loss_ratio) * reverse_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def log(self, logs):
        if "loss" in logs and logs["loss"] == -1:
            del logs["loss"]
        super().log(logs)

    def get_train_dataloader(self):
        shuffle = False if self.mode == "online" else True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        output = self.generator.generate(
            inputs["input_ids"], 128, attention_mask=torch.ones_like(inputs["input_ids"])
        )
        find = False
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, DistillTrainerCallback):
                callback.correct_cnt += output.correct_tokens.shape[-1]
                callback.propose_cnt += output.propose_steps
                callback.alpha += output.alpha_sum
                callback.sample_steps += output.sample_steps
                find = True
        assert find
        return None, None, None

    def optimizer_step(self, model, optimizer, optimizer_idx, closure=None, **kwargs):
        print("============I am in Optimizer================")
        if closure is not None:
            closure()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    ###################### Helper Functions #############################

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_kl(self, predicts, targets, padding_mask):
        kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        predict_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.log_softmax(targets, dim=-1)
        output = kl_loss(predict_prob, targets_prob)
        expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
        output.masked_fill_(expand_mask, 0)
        mean_output = output.sum() / (~padding_mask).sum()
        return mean_output

    @torch.inference_mode()
    def get_generated_ids(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        require_logits,
    ):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=require_logits,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if require_logits:
            logits = torch.cat([score.unsqueeze(1) for score in outputs["scores"]], dim=1)
        else:
            logits = None
        return outputs["sequences"], logits

    @torch.inference_mode()
    def get_mix_generated_ids(
        self,
        student_model,
        teacher_model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        mix_ratio,
    ):
        org_input_ids = input_ids.clone()

        def sample_token_from_logits(logits):
            tau = 0.001  # argmax-ish
            distribution = torch.softmax(logits / tau, dim=-1)
            next_token_id = torch.multinomial(distribution, num_samples=1)
            return next_token_id

        def generate_one(model, input_ids, attention_mask, past_key_values):
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            next_token = sample_token_from_logits(outputs.logits[:, -1, :])
            return next_token, past_key_values

        bsz, prompt_len = input_ids.shape
        student_first_token, student_key_values = generate_one(
            student_model, input_ids, attention_mask, None
        )
        teacher_first_token, teacher_key_values = generate_one(
            teacher_model, input_ids, attention_mask, None
        )

        torch.manual_seed(1)
        input_ids = student_first_token if random.random() < mix_ratio else teacher_first_token
        attention_mask = torch.cat(
            [attention_mask, torch.ones(bsz, 1, dtype=torch.long, device="cuda")], dim=1
        )

        for _ in range(max_new_tokens - 1):
            sample_model, past_key_values = (
                (student_model, student_key_values)
                if random.random() < mix_ratio
                else (teacher_model, teacher_key_values)
            )
            next_token, _ = generate_one(sample_model, input_ids, attention_mask, past_key_values)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(bsz, 1, dtype=torch.long, device="cuda")], dim=1
            )

        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, col in zip(*eos_positions):
            mask[row, col + 1 :] = True
        input_ids[mask] = tokenizer.pad_token_id
        return torch.cat((org_input_ids, input_ids), dim=-1).cuda()

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


class DistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.correct_cnt = 0
        self.propose_cnt = 0
        self.alpha = 0
        self.sample_steps = 0

    def on_evaluate(self, args, state, control, **kwargs):
        global eval_cnt
        print(f"[{eval_cnt}] {self.correct_cnt}/{self.propose_cnt}")

        if self.correct_cnt > 0:
            with open("out", "a") as f:
                f.write(f"[{eval_cnt}] {self.correct_cnt}/{self.propose_cnt}\n")
            wandb.log({"generated_token": self.correct_cnt * 1.0 / self.propose_cnt})
            wandb.log({"alpha": self.alpha * 1.0 / self.sample_steps})

        eval_cnt += 1
        self.correct_cnt = 0
        self.propose_cnt = 0
        self.alpha = 0
        self.sample_steps = 0