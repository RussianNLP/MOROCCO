from rsg_funcs import *

import json
import os, sys
import torch
import argparse

from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

DANETQA = 'danetqa'
LIDIRUS = 'lidirus'
MUSERC = 'muserc'
PARUS = 'parus'
RCB = 'rcb'
RUCOS = 'rucos'
RUSSE = 'russe'
RWSD = 'rwsd'
TERRA = 'terra'
TASKS = [
    DANETQA,
    MUSERC,
    PARUS,
    RCB,
    RUCOS,
    RUSSE,
    RWSD,
    TERRA,
    LIDIRUS,
]
TASK_TITLES = {
    DANETQA: 'DaNetQA',
    LIDIRUS: 'LiDiRus',
    MUSERC: 'MuSeRC',
    PARUS: 'PARus',
    RCB: 'RCB',
    RUCOS: 'RuCoS',
    RUSSE: 'RUSSE',
    RWSD: 'RWSD',
    TERRA: 'TERRa',
} 


def format_json(item):
    return json.dumps(item, ensure_ascii=False)


def format_jsonl(items):
    for item in items:
        yield format_json(item)


class GPT2LMHeadModelLossBatch(GPT2LMHeadModel):
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )  


def infer(args):
    task_dict = {DANETQA: danetqa_get_answer,
                 LIDIRUS: lidirus_get_answer,
                 MUSERC: muserc_get_answer,
                 PARUS: parus_get_answer,
                 RCB: rcb_get_answer,
                 RUCOS: rucos_get_answer,
                 RUSSE: russe_get_answer,
                 RWSD: rwsd_get_answer,
                 TERRA: terra_get_answer}
                 
    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = torch.device(args.device)
    
    bench_path =  os.path.join("/workspace", args.model_path) # '/home/jovyan/vetrov/rsg_docker/checkpoint-212000'
    
    tokenizer = AutoTokenizer.from_pretrained(bench_path, use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = GPT2LMHeadModelLossBatch.from_pretrained(bench_path)
    model.config.use_cache = True
    model.to(device)

    data = [json.loads(line) for line in sys.stdin.readlines()]

    # with open(os.path.join(bench_path,'bench_params.json'), 'r') as json_file:
    #     bench_params = json.load(json_file)
    # task = bench_params["task"]

    preds = task_dict[args.task](data, args.batch_size, tokenizer, model, device)

    for line in format_jsonl(preds):
        print(line) 


def main(args):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.set_defaults(function=None)
    subs = parser.add_subparsers()

    sub = subs.add_parser('infer')
    sub.set_defaults(function=infer)
    sub.add_argument('--task', choices=TASKS)
    sub.add_argument('--model-path', type=str)
    sub.add_argument('--batch-size', type=int)
    sub.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args(args)
    if not args.function:
        parser.print_help()
        parser.exit()
    try:
        args.function(args)
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
    
