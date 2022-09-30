import copy
import torch
import re

def calc_ppl_batch(phrases,
                   tokenizer,
                   model,
                   device):
    padded_batch = tokenizer.batch_encode_plus(phrases, padding=True)
    padded_batch["input_ids"] = torch.tensor(padded_batch["input_ids"], device=device)
    padded_batch["attention_mask"] = torch.tensor(padded_batch["attention_mask"], device=device)
    
    with torch.no_grad():
        loss = model(input_ids=padded_batch["input_ids"],
                     labels=padded_batch["input_ids"])

    corrected_loss = padded_batch["attention_mask"][:,1:]*loss[0]
    corrected_loss = corrected_loss.sum(dim=1) / padded_batch["attention_mask"][:,1:].sum(dim=1)
    
    perplexity = torch.exp(corrected_loss)
    
    return perplexity


def clean(text):
    text = re.sub(r'\((\d+)\)', '', text)
    return text


def muserc_decide_batch(q_ans):
    if len(q_ans)>3:
        num_correct = 2
    else:
        num_correct = 1
    top_inds = sorted(q_ans, key=q_ans.get)[:num_correct]
    return top_inds


def muserc_process_item(item, text_batches, dict_perplexity, tokenizer, model, device):
    new_item = copy.deepcopy(item)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr = calc_ppl_batch(batch['query_batch'], tokenizer, model, device)
        for perp_idx in range(len(perp_arr)): 
            dict_perplexity[batch['question_id'][perp_idx]]["answers"][batch['answer_id'][perp_idx]] = perp_arr[perp_idx]

    for q_idx in dict_perplexity:
        top_ind_res = muserc_decide_batch(dict_perplexity[q_idx]["answers"])
        for item_q_idx in range(len(new_item['passage']['questions'])):
            if new_item['passage']['questions'][item_q_idx]["idx"] == q_idx:
                for item_q_ans_idx in range(len(new_item['passage']['questions'][item_q_idx]["answers"])):
                    q_ans_in_idx = new_item['passage']['questions'][item_q_idx]["answers"][item_q_ans_idx]["idx"]
                    new_item['passage']['questions'][item_q_idx]['answers'][item_q_ans_idx]['label'] = 1 if q_ans_in_idx in top_ind_res else 0
                           
    return new_item


def muserc_get_item(item, batch_size):    
    text_batches = [{"query_batch":[], "answer_batch":[], "question_id":[], "answer_id":[]}]
    dict_perplexity = {}
    new_item = copy.deepcopy(item)

    text = clean(item['passage']['text'])
    questions = item['passage']['questions']
    
    for q in range(len(questions)):
        quest = questions[q]['question']
        ans = questions[q]['answers']
        question_id = questions[q]['idx']
        dict_perplexity[question_id] = {"answers":{}}
        for a in ans:
            answer_id = a['idx']
            answer = a['text']
            query = text+' Вопрос: '+quest+' Ответ: '+a['text']
            dict_perplexity[question_id]["answers"][answer_id] = 0
            if len(text_batches[-1]['query_batch']) >= batch_size:
                text_batches.append({"query_batch":[], "question_id":[], "answer_id":[]})
            text_batches[-1]['query_batch'].append(query)
            text_batches[-1]['question_id'].append(question_id)
            text_batches[-1]['answer_id'].append(answer_id)
                    
    return text_batches, dict_perplexity


def muserc_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    for i in range(len(new_data)):
        batches, perp_dict = muserc_get_item(new_data[i], batch_size)
        new_data[i] = muserc_process_item(new_data[i], batches, perp_dict, tokenizer, model, device)
    for line in new_data:
        del line['passage']['text']
    return new_data


def rcb_process_items(items, text_batches, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr = calc_ppl_batch(batch['query_batch'], tokenizer, model, device)
        for perp_idx in range(len(perp_arr)): 
            if perp_arr[perp_idx]>20:
                new_items[batch["rcb_id"][perp_idx]]['label'] = "neutral"
            else:
                new_items[batch["rcb_id"][perp_idx]]['label'] = "entailment"        
            new_items[batch["rcb_id"][perp_idx]] = {'idx':int(new_items[batch["rcb_id"][perp_idx]]['idx']), 
                                                    'label':str(new_items[batch["rcb_id"][perp_idx]]['label'])}                 
    return new_items


def rcb_get_items(items, batch_size):    
    text_batches = [{"query_batch":[], "rcb_id":[]}] 
    
    for rcb_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        text = clean(new_item['premise'])
        question = new_item['hypothesis']
        X = text+' Из этого следует, что '+question[0].lower()+question[1:]
        if len(text_batches[-1]['query_batch']) >= batch_size:
            text_batches.append({"query_batch":[], "rcb_id":[]})
        text_batches[-1]["query_batch"].append(X)
        text_batches[-1]["rcb_id"].append(rcb_id)
        
    return text_batches


def rcb_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = rcb_get_items(new_data, batch_size)
    new_data = rcb_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data


def danetqa_process_items(items, text_batches, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        yes_batch = [x + ' Да' for x in batch['query_batch']]
        no_batch = [x + ' Нет' for x in batch['query_batch']]
        perp_arr_yes = calc_ppl_batch(yes_batch, tokenizer, model, device)
        perp_arr_no = calc_ppl_batch(no_batch, tokenizer, model, device)
        for perp_idx in range(len(perp_arr_yes)): 
            if perp_arr_yes[perp_idx]<perp_arr_no[perp_idx]:
                new_items[batch["danetqa_id"][perp_idx]]['label'] = True
            else:
                new_items[batch["danetqa_id"][perp_idx]]['label'] = False      
            new_items[batch["danetqa_id"][perp_idx]] = {'idx':int(new_items[batch["danetqa_id"][perp_idx]]['idx']),
                                                        'label':str(new_items[batch["danetqa_id"][perp_idx]]['label']).lower()}                 
    return new_items


def danetqa_get_items(items, batch_size):    
    text_batches = [{"query_batch":[], "danetqa_id":[]}]

    for danetqa_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        question = new_item['question']
        if len(text_batches[-1]['query_batch']) >= batch_size:
            text_batches.append({"query_batch":[], "danetqa_id":[]})
        text_batches[-1]["query_batch"].append(question)
        text_batches[-1]["danetqa_id"].append(danetqa_id)
        
    return text_batches


def danetqa_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = danetqa_get_items(new_data, batch_size)
    new_data = danetqa_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data


def parus_process_items(items, text_batches, tokenizer, model, device):
    question_conv = {'effect': ' Из-за этого ', 'cause': ' Потому что '}
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        choice1_batch = [clean(batch['premise_batch'][idx])+\
                         question_conv[batch['query_batch'][idx]]+\
                         batch['choice1'][idx][0].lower()+\
                         batch['choice1'][idx][1:] for idx in range(len(batch['query_batch']))]
        choice2_batch = [clean(batch['premise_batch'][idx])+\
                         question_conv[batch['query_batch'][idx]]+\
                         batch['choice2'][idx][0].lower()+\
                         batch['choice2'][idx][1:] for idx in range(len(batch['query_batch']))]
        perp_arr_choice1 = calc_ppl_batch(choice1_batch, tokenizer, model, device)
        perp_arr_choice2 = calc_ppl_batch(choice2_batch, tokenizer, model, device)
        for perp_idx in range(len(perp_arr_choice1)): 
            new_items[batch["parus_id"][perp_idx]]['label'] = int(perp_arr_choice1[perp_idx]>perp_arr_choice2[perp_idx])
            del new_items[batch["parus_id"][perp_idx]]['premise']
            del new_items[batch["parus_id"][perp_idx]]['choice1']
            del new_items[batch["parus_id"][perp_idx]]['choice2']
            del new_items[batch["parus_id"][perp_idx]]['question']               
    return new_items


def parus_get_items(items, batch_size):    
    text_batches = [{"query_batch":[], "premise_batch":[],
                     "choice1":[], "choice2":[], "parus_id":[]}]
    
    for parus_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        question = new_item['question']
        premise = new_item['premise']
        choice1 = new_item['choice1']
        choice2 = new_item['choice2']
        if len(text_batches[-1]['query_batch']) >= batch_size:
            text_batches.append({"query_batch":[], "premise_batch":[], 
                                 "choice1":[], "choice2":[], "parus_id":[]})
        text_batches[-1]["query_batch"].append(question)
        text_batches[-1]["premise_batch"].append(premise)
        text_batches[-1]["choice1"].append(choice1)
        text_batches[-1]["choice2"].append(choice2)
        text_batches[-1]["parus_id"].append(parus_id)
        
    return text_batches


def parus_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = parus_get_items(new_data, batch_size)
    new_item = parus_process_items(new_data, text_batches, tokenizer, model, device)
    return new_item


def rucos_get_items(items, batch_size):
    text_batches = [{"query_batch":[], "answer_batch":[], "passage_id":[]}]
    dict_perplexity = {}
    
    for p in range(len(items)):
        new_item = copy.deepcopy(items[p])
        passage_id = new_item["idx"]
        dict_perplexity[passage_id] = {} 
        text = new_item['passage']['text']
        question = new_item['qas'][0]['query']    
        for idx, ans in enumerate(new_item['passage']['entities']):
            a = text[ans['start']:ans['end']]
            query = text+' Заголовок: '+re.sub('@placeholder', a, question)    
            dict_perplexity[passage_id][a] = 0
            if len(text_batches[-1]['query_batch']) >= batch_size:
                text_batches.append({"query_batch":[], "answer_batch":[], "passage_id":[]})
            text_batches[-1]["query_batch"].append(query)
            text_batches[-1]["answer_batch"].append(a)
            text_batches[-1]["passage_id"].append(passage_id)

    return text_batches, dict_perplexity


def rucos_decide_batch(q_ans):
    if len(q_ans)>3:
        num_correct = 2
    else:
        num_correct = 1
    top_inds = sorted(q_ans, key=q_ans.get)[:num_correct]
    return top_inds


def rucos_process_items(items, text_batches, dict_perplexity, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr = calc_ppl_batch(batch['query_batch'], tokenizer, model, device)
        for perp_idx in range(len(perp_arr)): 
            dict_perplexity[batch['passage_id'][perp_idx]][batch['answer_batch'][perp_idx]] = perp_arr[perp_idx]
    
    for q_idx in dict_perplexity:
        top_ind_res = rucos_decide_batch(dict_perplexity[q_idx])
        for item_p_idx in range(len(new_items)):
            if new_items[item_p_idx]["idx"] == q_idx:
                new_items[item_p_idx] = {'idx': new_items[item_p_idx]["idx"], 'label': top_ind_res[0]}
                           
    return new_items


def rucos_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches, dict_perp = rucos_get_items(new_data, batch_size)
    new_data = rucos_process_items(new_data, text_batches, dict_perp, tokenizer, model, device)
    return new_data


def russe_process_items(items, text_batches, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr_choice1 = calc_ppl_batch([x[0] for x in batch["choices"]], tokenizer, model, device)
        perp_arr_choice2 = calc_ppl_batch([x[1] for x in batch["choices"]], tokenizer, model, device)
        for perp_idx in range(len(perp_arr_choice1)):
            new_items[batch["russe_id"][perp_idx]] = {"idx": new_items[batch["russe_id"][perp_idx]]["idx"],
                                                      'label': str(bool((perp_arr_choice1[perp_idx]>perp_arr_choice2[perp_idx]).item())).lower()}             
    return new_items


def russe_get_items(items, batch_size):    
    text_batches = [{"choices":[], "russe_id":[]}]
    
    for russe_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        choices = []
        choices.append(clean('Слово ' + new_item['word'] +\
                             ' значит одно и то же в предложениях: Предложение 1: ' +\
                             new_item['sentence1'] + ' Предложение 2: '+\
                             new_item['sentence2'] + " Ответ: верно"))
        choices.append(clean('Слово ' + new_item['word']+\
                             ' значит одно и то же в предложениях: Предложение 1: '+\
                             new_item['sentence1'] +' Предложение 2: '+\
                             new_item['sentence2']+" Ответ: неверно"))
        if len(text_batches[-1]['choices']) >= batch_size:
            text_batches.append({"choices":[], "russe_id":[]})
        text_batches[-1]["choices"].append(choices)
        text_batches[-1]["russe_id"].append(russe_id)
        
    return text_batches


def russe_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = russe_get_items(new_data, batch_size)
    new_data = russe_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data


def rwsd_get_items(items, batch_size):
    text_batches = [{"query_batch":[], "item_idx":[], "span":[]}]
    dict_perplexity = {}
    
    for item_idx in range(len(items)):
        new_item = copy.deepcopy(items[item_idx])
        span = new_item['text']
        target1 = new_item['target']['span1_text']
        target2 = new_item['target']['span2_text'] 
        query = span+' Имеется в виду, что '+target1+' - '+target2
        if len(text_batches[-1]['query_batch']) >= batch_size:
            text_batches.append({"query_batch":[], "item_idx":[], "span":[]})
        text_batches[-1]["query_batch"].append(query)
        text_batches[-1]["item_idx"].append(item_idx)
        text_batches[-1]["span"].append(span)

    return text_batches


def rwsd_decide_batch(q_ans):
    num_correct = 1
    top_inds = sorted(q_ans, key=q_ans.get)[:num_correct]
    return top_inds


def rwsd_process_items(items, text_batches, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    spans = {}
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr = calc_ppl_batch(batch['query_batch'], tokenizer, model, device)
        for perp_idx in range(len(perp_arr)): 
            if batch['span'][perp_idx] not in spans:
                spans[batch['span'][perp_idx]] = {}
            spans[batch['span'][perp_idx]][batch["item_idx"][perp_idx]] = perp_arr[perp_idx]
            
    for span in spans:
        top_ind_res = rwsd_decide_batch(spans[span])
        for span_idx in spans[span]:
            new_items[span_idx] = {"idx": new_items[span_idx]["idx"], "label": span_idx in top_ind_res}
                           
    return new_items


def rwsd_get_answer(data, batch_size, tokenizer, model, device):    
    new_data = copy.deepcopy(data)
    text_batches = rwsd_get_items(new_data, batch_size)
    new_data = rwsd_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data


def terra_process_items(items, text_batches, tokenizer, model, device):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr = calc_ppl_batch(batch['query_batch'], tokenizer, model, device)
        for perp_idx in range(len(perp_arr)): 
            if perp_arr[perp_idx]<22:
                new_items[batch["terra_id"][perp_idx]]['label'] = "entailment"
            else:
                new_items[batch["terra_id"][perp_idx]]['label'] = "not_entailment"        
            new_items[batch["terra_id"][perp_idx]] = {'idx':int(new_items[batch["terra_id"][perp_idx]]['idx']), 'label':new_items[batch["terra_id"][perp_idx]]['label']}                 
    return new_items


def terra_get_items(items, batch_size):    
    text_batches = [{"query_batch":[], "terra_id":[]}]
    
    for terra_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        text = clean(new_item['premise'])
        question = new_item['hypothesis']
        X = text+' '+question
        if len(text_batches[-1]['query_batch']) >= batch_size:
            text_batches.append({"query_batch":[], "terra_id":[]})
        text_batches[-1]["query_batch"].append(X)
        text_batches[-1]["terra_id"].append(terra_id)
        
    return text_batches


def terra_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = terra_get_items(new_data, batch_size)
    new_data = terra_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data


def lidirus_process_items(items, text_batches, tokenizer, model, args):
    new_items = copy.deepcopy(items)
    
    for batch_idx, batch in enumerate(text_batches):
        perp_arr_choice1 = calc_ppl_batch([x[0] for x in batch["choices"]], tokenizer, model, args)
        perp_arr_choice2 = calc_ppl_batch([x[1] for x in batch["choices"]], tokenizer, model, args)
        for perp_idx in range(len(perp_arr_choice1)):
            label = 'not_entailment'
            if perp_arr_choice1[perp_idx]<perp_arr_choice2[perp_idx]:
                label = 'entailment'
            new_items[batch["lidirus_id"][perp_idx]] = {"idx": new_items[batch["lidirus_id"][perp_idx]]["idx"], 'label': label}             
    return new_items


def lidirus_get_items(items, batch_size):    
    text_batches = [{"choices":[], "lidirus_id":[]}]
    
    for russe_id, item in enumerate(items):
        new_item = copy.deepcopy(item)
        choices = []
        choices.append(new_item['sentence1']+' Из этого следует, что '+new_item['sentence2'])
        choices.append(new_item['sentence1']+' Из этого не следует, что '+new_item['sentence2'])
        if len(text_batches[-1]['choices']) >= batch_size:
            text_batches.append({"choices":[], "lidirus_id":[]})
        text_batches[-1]["choices"].append(choices)
        text_batches[-1]["lidirus_id"].append(russe_id)
        
    return text_batches


def lidirus_get_answer(data, batch_size, tokenizer, model, device):
    new_data = copy.deepcopy(data)
    text_batches = lidirus_get_items(new_data, batch_size)
    new_data = lidirus_process_items(new_data, text_batches, tokenizer, model, device)
    return new_data
