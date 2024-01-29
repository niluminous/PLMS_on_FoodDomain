import collections
import json
import logging
import os
import re
import string
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count
import spacy
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from processors.utils import DataProcessor

class CoqaExample(object):
    """Single CoQA example"""
    def __init__(
            self,
            qas_id,
            question_text,
            doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            rational_start_position=None,
            rational_end_position=None,
            additional_answers=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.additional_answers = additional_answers
        self.rational_start_position = rational_start_position
        self.rational_end_position = rational_end_position

class CoqaFeatures(object):
    """Single CoQA feature"""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 cls_idx=None,
                 rational_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.cls_idx = cls_idx
        self.rational_mask = rational_mask

class Result(object):
    def __init__(self, unique_id, start_logits, end_logits, yes_logits, no_logits, unk_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.yes_logits = yes_logits
        self.no_logits = no_logits
        self.unk_logits = unk_logits

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """
    doc_tokens: A list of tokenized words from the context passage.
    input_start: The starting index of the predicted answer span.
    input_end: The ending index of the predicted answer span.
    tokenizer: A tokenizer used to tokenize the original answer text.
    orig_answer_text: The original answer text (non-tokenized).

    to refine the answer span predicted by the model
    by iterating over possible start and end positions within the doc_tokens
    to find the span of tokens that matches the tokenized orig_answer_text

    
    The function compares tokenized versions of the answer text with tokenized sub-spans of the context passage to find the best match.
    If a matching span is found, it returns the new start and end positions for the answer span that match the original answer text. 
    If no match is found, it returns the original input_start and input_end positions.
    
    """
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    doc_spans: A list of document spans. Each document span has information about its start, length, and other properties.
    cur_span_index: The current span index being considered.
    position: The position of the token within the document.

    to check if the current span (cur_span_index) contains the maximum context for the given position
    It determines if the current span is the most contextually relevant one for the given token position.
    
    It calculates a score for each document span based on the proximity of the token position to the start and end of the span, 
    giving more importance to spans with more context. 
    
    The span with the highest score is considered the one with the maximum context.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class Processor(DataProcessor):
    """
    DataProcessor 클래스를 상속받아 CoQA (Conversational Question Answering) 데이터셋을 처리하는 클래스인 Processor

    CoQA 데이터셋에서 각 데이터 포인트에 대한 CoqaExample을 생성하며, 
    히스토리 길이, 데이터셋 유형, 어텐션 여부 등을 고려하여 다양한 처리를 수행
    """

    # 클래스 변수
    train_file = "coqa-train-v1.0.json"
    dev_file = "coqa-dev-v1.0.json"

    def is_whitespace(self, c):
        # 주어진 문자 c가 공백 문자인지 여부를 확인
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        """
        PTB (Penn Treebank) 토큰을 일반 토큰으로 변환합니다.
        특정 토큰에 대한 변환 규칙이 정의되어 있습니다.
        """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    def space_extend(self, matchobj):
        """
        정규 표현식 패턴에 일치하는 문자열에 공백을 추가합니다.
        주로 특수 문자들을 공백으로 감싸는 용도로 사용됩니다.
        """
        return ' ' + matchobj.group(0) + ' '

    def pre_proc(self, text):
        text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', self.space_extend, text)
        text = text.strip(' \n')
        text = re.sub('\s+', ' ', text)
        return text

    def process(self, parsed_text):
        # 문장을 처리하여 단어, 오프셋, 문장 정보를 가진 딕셔너리를 반환
        output = {'word': [], 'offsets': [], 'sentences': []}
        for token in parsed_text:
            output['word'].append(self._str(token.text))
            output['offsets'].append((token.idx, token.idx + len(token.text)))
        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)
        assert word_idx == len(output['word'])
        return output

    def get_raw_context_offsets(self, words, raw_text):
        """
        단어와 원본 텍스트를 받아 각 단어의 원본 텍스트 상에서의 오프셋을 계산합니다.
        계산된 오프셋은 리스트로 반환됩니다.
        """
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:', raw_text)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)
        return raw_context_offsets

    def find_span(self, offsets, start, end):
        # 오프셋을 찾아 튜플로 반환 
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)

    def normalize_answer(self, s):
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def find_span_with_gt(self, context, offsets, ground_truth):
        """
        주어진 컨텍스트와 오프셋, 그리고 실제 정답을 사용하여 최적의 스팬을 찾습니다.
        정답과 일치하는 스팬 중에서 F1 점수가 가장 높은 스팬을 선택합니다.
        """
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(self.pre_proc(ground_truth)).split()

        ls = [
            i for i in range(len(offsets))
            if context[offsets[i][0]:offsets[i][1]].lower() in gt
        ]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = self.normalize_answer(
                    self.pre_proc(
                        context[offsets[ls[i]][0]:offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (ls[i], ls[j])
        return best_span

    def cut_sentence(self,doc_tok, r_start, r_end,dataset_type, nlp_context):
        if dataset_type in ['TS','RG']:
            if dataset_type == "TS":
                edge,inc = r_end,True
            elif dataset_type == "RG":
                edge,inc = r_start,False
            for i,j in nlp_context:
                if edge >= i and edge <= j:
                    sent = j if inc else i
            doc_tok = doc_tok[:sent]
            return doc_tok
        else:
            return doc_tok

    def get_examples(self, data_dir, history_len, filename=None, threads=1,dataset_type = None, attention = False):
        """
        주어진 데이터 디렉토리에서 훈련 데이터를 읽어 CoQAExample 객체를 생성하여 반환합니다.
        SpaCy를 사용하여 문장을 처리하고, 필요에 따라 문서를 잘라내거나 추가 답변을 고려합니다.
        """
        if data_dir is None:
            data_dir = ""

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]

        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self._create_examples, history_len=history_len, dataset_type = dataset_type, attention = attention)
            examples = list(tqdm(
                p.imap(annotate_, input_data),
                total=len(input_data),
                desc="Preprocessing examples",
            ))
        examples = [item for sublist in examples for item in sublist]
        return examples

    def _create_examples(self, input_data, history_len,dataset_type = None, attention = False):
        """
        각 데이터 포인트에 대해 CoQAExample을 생성합니다.
        히스토리 길이, 데이터셋 유형, 어텐션 여부 등을 고려하여 처리합니다.
        """
        nlp = spacy.load('en_core_web_sm', parser=False) # SpaCy를 사용하여 영어 텍스트를 처리하기 위해 SpaCy 모델을 로드
        examples = [] # 생성된 CoqaExample 객체들을 담을 리스트를 초기화
        datum = input_data
        context_str = datum['story'] # 현재 데이터 포인트의 이야기(본문)를 가져옵니다
        _datum = { # 초기화
            'context': context_str, # 'context': 이야기를 담고 있는 키.
            'source': datum['source'], # 'source': 데이터 소스를 나타내는 키.
            'id': datum['id'],
            'filename': datum['filename']
        }

        # SpaCy를 사용하여 이야기를 처리하고, 전처리 함수를 통해 정제된 텍스트를 얻습니다.
        nlp_context = nlp(self.pre_proc(context_str))

        # 전처리된 이야기를 process 함수로 처리하여 얻은 결과를 _datum에 추가합니다.
        _datum['annotated_context'] = self.process(nlp_context)

        # 문장의 각 단어에 대한 원본 텍스트 상에서의 오프셋을 계산하여 _datum에 추가합니다.
        _datum['raw_context_offsets'] = self.get_raw_context_offsets(_datum['annotated_context']['word'], context_str)
        
        # 질문과 답변의 개수가 동일한지 확인합니다.
        assert len(datum['questions']) == len(datum['answers'])
        additional_answers = {}
        if 'additional_answers' in datum:
            for k, answer in datum['additional_answers'].items():
                if len(answer) == len(datum['answers']):
                    for ex in answer:
                        idx = ex['turn_id']
                        if idx not in additional_answers:
                            additional_answers[idx] = []
                        additional_answers[idx].append(ex['input_text'])
        for i in range(len(datum['questions'])): # 각 질문에 대해 처리를 시작
            question, answer = datum['questions'][i], datum['answers'][i]
            assert question['turn_id'] == answer['turn_id']

            idx = question['turn_id']
            _qas = {
                'turn_id': idx,
                'question': question['input_text'],
                'answer': answer['input_text']
            }
            if idx in additional_answers: # 현재 turn_id에 대한 추가적인 답변이 있는 경우 _qas에 추가
                _qas['additional_answers'] = additional_answers[idx]

            _qas['raw_answer'] = answer['input_text'] # 원본 답변 텍스트를 _qas에 추가

            if _qas['raw_answer'].lower() in ['yes', 'yes.']: # 원본 답변이 'yes' 또는 'yes.'인 경우, 'yes'로 변경
                _qas['raw_answer'] = 'yes'
            if _qas['raw_answer'].lower() in ['no', 'no.']:
                _qas['raw_answer'] = 'no'
            if _qas['raw_answer'].lower() in ['unknown', 'unknown.']:
                _qas['raw_answer'] = 'unknown'

            _qas['answer_span_start'] = answer['span_start'] # 답변의 시작 위치를 _qas에 추가
            _qas['answer_span_end'] = answer['span_end'] # 답변의 끝 위치를 _qas에 추가
            start = answer['span_start'] # 답변의 시작 위치 가져오기 
            end = answer['span_end']
            chosen_text = _datum['context'][start:end].lower()
            while len(chosen_text) > 0 and self.is_whitespace(chosen_text[0]):
                """
                선택된 텍스트의 첫 문자가 공백인 경우, 공백을 제거하고 시작 위치를 조정하여
                시작 위치를 정확하게 지정
                """
                chosen_text = chosen_text[1:]
                start += 1
            while len(chosen_text) > 0 and self.is_whitespace(chosen_text[-1]):
                """
                위와 마찬가지 방법대로 끝 위치 조정
                """
                chosen_text = chosen_text[:-1]
                end -= 1
            r_start, r_end = self.find_span(_datum['raw_context_offsets'], start, end)
            input_text = _qas['answer'].strip().lower()

            if input_text in chosen_text:
                p = chosen_text.find(input_text)
                _qas['answer_span'] = self.find_span(_datum['raw_context_offsets'], start + p, start + p + len(input_text))
            else:
                if dataset_type == 'TS':
                    pos = _datum['raw_context_offsets'][r_end][1]
                    _qas['answer_span'] = self.find_span_with_gt( _datum['context'][:pos], _datum['raw_context_offsets'][:r_end+1], input_text)
                else:
                    _qas['answer_span'] = self.find_span_with_gt(_datum['context'], _datum['raw_context_offsets'], input_text)

            long_questions = [] # 질문 리스트 초기화
            for j in range(i - history_len, i + 1):
                long_question = '' # 현재 인덱스가 0 미만인 경우, 현재 질문만을 고려
                if j < 0:
                    continue

                long_question += '|Q| ' + datum['questions'][j]['input_text']
                # 현재 질문을 긴 질문에 추가
                if j < i:
                    # 현재 인덱스가 현재 질문보다 작은 경우, 현재 질문 이전의 답변을 추가
                    long_question += '|A| ' + datum['answers'][j]['input_text'] + ' '

                long_question = long_question.strip()
                long_questions.append(long_question)

            # 문맥의 단어 토큰을 가져옴
            doc_tok = _datum['annotated_context']['word']
            # 문장의 일부분만 선택
            doc_tok = self.cut_sentence(doc_tok, r_start, r_end, dataset_type,_datum['annotated_context']['sentences'])
            if len(doc_tok) == 0:
                continue

            if dataset_type == "RG":
                r_start,r_end = -1,-1 # 시작과 끝 위치 초기화
                gt = _qas['raw_answer'] # 원본 답변 가져오기
                gt_context = nlp(self.pre_proc(gt))
                _gt = self.process(gt_context)['word'] # 처리된 답변의 토큰을 가지고 옴
                found = " ".join(doc_tok).find(gt) # 답변의 문맥에 있는지 확인하고 위치를 찾음
                if gt not in ['unknown','yes','no']:
                    if found == -1 and not attention: # 답변에 문맥이 없고 어텐션 수행하지 않는 경우
                        doc_tok.append(gt) # 문맥에 답변 추가 
                    elif found != -1 and not attention: # 답변에 문맥이 있고 어텐션을 수행하지 않는 경우
                        r_start,r_end = -1,-1 # 시작과 끝 위치 초기화
                    elif found == -1 and attention: # 답변이 문맥에 없고 어텐션을 수행하는 경우 
                        r_start,r_end = len(doc_tok),len(doc_tok)+len(_gt)-1 # 시작과 끝 위치를 추가된 답변의 범위로 설정하여 문맥에 추가 
                        doc_tok.extend(_gt) # 처리된 답변의 단어 토큰을 문맥에 추가 
                    else: #답변이 문맥에 있고 어텐션을 수행하는 경우
                        for i in range(0,len(doc_tok)):
                            if doc_tok[i:i+len(_gt)] == _gt:
                                r_start = i 
                                r_end = r_start + len(_gt)-1
                        if r_start == r_end:
                            continue
                elif attention:
                    continue

            example = CoqaExample( # CoqaExample 객체를 생성
                qas_id = _datum['id'] + ' ' + str(_qas['turn_id']),
                question_text = long_questions,
                doc_tokens = doc_tok,
                orig_answer_text = _qas['raw_answer'] if dataset_type in [None,'TS'] else 'unknown',
                start_position = _qas['answer_span'][0] if dataset_type in [None,'TS'] else 0, 
                end_position = _qas['answer_span'][1] if dataset_type in [None,'TS'] else 0,
                rational_start_position = r_start,
                rational_end_position = r_end,
                additional_answers=_qas['additional_answers'] if 'additional_answers' in _qas else None,
            )
            examples.append(example)

        return examples


def Extract_Feature_init(tokenizer_for_convert): # 토크나이저 초기화
    global tokenizer
    tokenizer = tokenizer_for_convert

def Extract_Feature(example, tokenizer, max_seq_length = 512, doc_stride = 128, max_query_length = 64):
    features = []
    query_tokens = []
    for question_answer in example.question_text:
        query_tokens.extend(tokenizer.tokenize(question_answer))

    cls_idx = 3
    if example.orig_answer_text == 'yes':
        cls_idx = 0  # yes
    elif example.orig_answer_text == 'no':
        cls_idx = 1  # no
    elif example.orig_answer_text == 'unknown':
        cls_idx = 2  # unknown

    if len(query_tokens) > max_query_length:
        # keep tail
        query_tokens = query_tokens[-max_query_length:]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)


    """
    example.doc_tokens에 대해 토크나이징하고, 각 토큰의 원본 인덱스와 토큰화된 인덱스를 기록
    """
    tok_r_start_position = orig_to_tok_index[example.rational_start_position]
    if example.rational_end_position < len(example.doc_tokens) - 1:
        tok_r_end_position = orig_to_tok_index[example.rational_end_position + 1] - 1
    else:
        tok_r_end_position = len(all_doc_tokens) - 1
    if cls_idx < 3:
        tok_start_position, tok_end_position = 0, 0
    else:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
        
    # The -4 accounts for <s>, </s></s> and </s>
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 4

    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        slice_cls_idx = cls_idx
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        tokens.append("<s>")
        for token in query_tokens:
            tokens.append(token)
        tokens.extend(["</s>","</s>"])

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans,
                                                   doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
        tokens.append("</s>")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)
        segment_ids = [0]*max_seq_length
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # rational_part
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if example.rational_start_position == -1 or not (
                tok_r_start_position >= doc_start and tok_r_end_position <= doc_end):
            out_of_span = True
        if out_of_span:
            rational_start_position = 0
            rational_end_position = 0
        else:
            doc_offset = len(query_tokens) + 3
            rational_start_position = tok_r_start_position - doc_start + doc_offset
            rational_end_position = tok_r_end_position - doc_start + doc_offset
        # rational_part_end

        rational_mask = [0] * len(input_ids)
        if not out_of_span:
            rational_mask[rational_start_position:rational_end_position + 1] = [1] * (
                        rational_end_position - rational_start_position + 1)

        if cls_idx >= 3:
            # For training, if our document chunk does not contain an annotation we remove it
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                slice_cls_idx = 2
            else:
                doc_offset = len(query_tokens) + 3
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        else:
            start_position = 0
            end_position = 0

        features.append(
            CoqaFeatures(example_index=0,
                         unique_id=0,
                         doc_span_index=doc_span_index,
                         tokens=tokens,
                         token_to_orig_map=token_to_orig_map,
                         token_is_max_context=token_is_max_context,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         start_position=start_position,
                         end_position=end_position,
                         cls_idx=slice_cls_idx,
                         rational_mask=rational_mask))
    return features


def Extract_Features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training,threads=1):
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=Extract_Feature_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            Extract_Feature,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=16),
                total=len(examples),
                desc="Extracting features from dataset",
            )
        )
    # 병렬 처리를 위해 multiprocessing.Pool을 사용

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="Tag unique id to each example"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    """
    각 feature에 고유한 ID를 부여하고, 새로운 feature 리스트에 저장
    이때, 비어 있는 feature는 무시
    모든 feature의 input_ids, input_mask, tokentype_ids를 PyTorch Tensor로 변환
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_tokentype_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_tokentype_ids, all_input_mask, all_example_index)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        all_rational_mask = torch.tensor([f.rational_mask for f in features], dtype=torch.long)
        all_cls_idx = torch.tensor([f.cls_idx for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_tokentype_ids, all_input_mask, all_start_positions,
                                all_end_positions, all_rational_mask, all_cls_idx)

    return features, dataset
