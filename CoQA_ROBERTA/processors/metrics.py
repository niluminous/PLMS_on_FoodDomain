import collections
import json
import logging
import math
import re
import string
from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer


#   This code has been very heavily adapted/used for our CoQA model from the hugging face's Bert implmentation on SQuaD dataset. 
#   https://github.com/huggingface/transformers/blob/master/src/transformers/data/metrics/squad_metrics.py

def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, output_prediction_file, verbose_logging, tokenizer):
    """
    1. 주어진 예제와 모델의 예측 결과를 효율적으로 매핑하기 위해 데이터 구조를 초기화
    2. 'yes', 'no', 'unknown' 및 스팬 형식의 예측 결과에 대한 점수를 계산하고, 가장 높은 점수를 갖는 예측 결과의 인덱스를 추적
    3. 최종 예측을 생성하기 위해 'yes', 'no', 'unknown' 및 스팬 형식의 예측 결과를 종합
    4. 중복된 예측을 방지하고, n-best 형식으로 최종 예측을 구성
    5. 최종 예측 결과를 JSON 형식으로 저장하고, 예측 결과 및 n-best 결과를 반환
    
    """
    # 모델 예측 결과를 가져와서 최종 예측을 생성하는 함수
    # 주어진 예측 결과, 피처, 예제 및 기타 하이퍼파라미터를 기반으로 최종 예측을 생성
    
    # 예제 인덱스에 해당하는 피처들과 고유한 ID에 해당하는 모델 결과를 매핑
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result


    # 예비 예측의 데이터 구조를 정의
    _PrelimPrediction = collections.namedtuple("PrelimPrediction", ["feature_index", "start_index", "end_index", "score", "cls_idx",])

    # 최종 예측과 예비 예측 결과를 저장할 리스트 및 딕셔너리를 초기화
    all_predictions = []
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(tqdm(all_examples, desc="Writing preditions")):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        
        # 각 예측 결과의 점수 및 피처 인덱스를 저장할 변수를 초기화
        score_yes, score_no, score_span, score_unk = -float('INF'), -float('INF'), -float('INF'), float('INF')
        min_unk_feature_index, max_yes_feature_index, max_no_feature_index, max_span_feature_index = -1, -1, -1, -1
        

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            
            feature_yes_score, feature_no_score, feature_unk_score = \
                result.yes_logits[0] * 2, result.no_logits[0] * 2, result.unk_logits[0] * 2
            
            # 시작 지점과 종료 지점 예측에서 가장 높은 점수를 갖는 인덱스를 가져옴
            start_indexes, end_indexes = _get_best_indexes(result.start_logits, n_best_size), \
                                         _get_best_indexes(result.end_logits, n_best_size)

            # 가능한 스팬 예측 결과를 가져와서 예비 예측 리스트에 추가
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    feature_span_score = result.start_logits[start_index] + result.end_logits[end_index]
                    prelim_predictions.append(_PrelimPrediction(feature_index=feature_index,start_index=start_index,end_index=end_index,score=feature_span_score,cls_idx=3))
            
            # 'unknown', 'yes', 'no' 예측 중에서 가장 높은 점수를 갖는 예측 결과의 피처 인덱스를 갱신
            if feature_unk_score < score_unk:  
                score_unk = feature_unk_score
                min_unk_feature_index = feature_index
            if feature_yes_score > score_yes: 
                score_yes = feature_yes_score
                max_yes_feature_index = feature_index
            if feature_no_score > score_no: 
                score_no = feature_no_score
                max_no_feature_index = feature_index
                
        #including yes/no/unknown answers in preliminary predictions.
        prelim_predictions.append(_PrelimPrediction(feature_index=min_unk_feature_index,start_index=0,end_index=0,score=score_unk,cls_idx=2))
        prelim_predictions.append(_PrelimPrediction(feature_index=max_yes_feature_index,start_index=0,end_index=0,score=score_yes,cls_idx=0))
        prelim_predictions.append(_PrelimPrediction(feature_index=max_no_feature_index,start_index=0,end_index=0,score=score_no,cls_idx=1))
        prelim_predictions = sorted(prelim_predictions,key=lambda p: p.score,reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "score", "cls_idx"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            #   free-form answers (ie span answers)
            if pred.cls_idx == 3:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                # removing whitespaces
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
                nbest.append(_NbestPrediction(text=final_text,score=pred.score,cls_idx=pred.cls_idx))
                
            #   'yes'/'no'/'unknown' answers
            # 'yes', 'no', 'unknown' 예측 결과를 최종 예측 리스트에 추가
            else:
                text = ['yes', 'no', 'unknown']
                nbest.append(_NbestPrediction(text=text[pred.cls_idx], score=pred.score, cls_idx=pred.cls_idx))

        # 최종 예측이 없는 경우 'unknown'을 추가
        if len(nbest) < 1:
            nbest.append(_NbestPrediction(text='unknown', score=-float('inf'), cls_idx=2))

        assert len(nbest) >= 1

        probs = _compute_softmax([p.score for p in nbest])

        nbest_json = []

        # 최종 예측 결과에 대한 확률을 계산하고, JSON 형식으로 변환
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["score"] = entry.score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        # 예측 결과를 전체 예측 리스트에 추가하고, 예측 결과를 고유한 ID를 사용하여 딕셔너리에 저장
        _id, _turn_id = example.qas_id.split()
        all_predictions.append({
            'id': _id,
            'turn_id': int(_turn_id),
            'answer': confirm_preds(nbest_json)})
        all_nbest_json[example.qas_id] = nbest_json
    #   Writing all the predictions in the predictions.json file in the BERT directory
    # 최종 예측 결과를 JSON 파일로 저장
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """
    주어진 예측 텍스트와 원본 텍스트, 그리고 텍스트의 소문자 변환 여부를 고려하여 최종 예측 결과를 반환
    이때, 스페이스를 제외한 문자를 기준으로 위치 정보를 변환하여 최종 예측된 텍스트를 추출
    """
    # 스페이스를 제외한 문자를 추출하고, 매핑 정보를 생성하는 함수
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # BasicTokenizer를 이용하여 토큰화하는 객체 생성
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    # 원본 텍스트를 토큰화하여 스페이스를 제외한 형태로 변환
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    # 예측한 텍스트의 시작 위치를 찾음
    start_position = tok_text.find(pred_text)
    
    # 예측한 텍스트가 존재하지 않으면 원본 텍스트 반환
    if start_position == -1:
        return orig_text
    
    # 예측한 텍스트의 끝 위치 계산
    end_position = start_position + len(pred_text) - 1

    # 스페이스를 제외한 원본 텍스트와 토큰화된 텍스트의 정보 추출
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    # 원본 텍스트와 토큰화된 텍스트의 길이가 다르면 원본 텍스트 반환
    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text
    
    # 토큰화된 텍스트의 인덱스를 원본 텍스트의 인덱스로 매핑하는 딕셔너리 생성
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    # 토큰화된 텍스트 상의 시작 위치를 원본 텍스트 상의 위치로 변환
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    # 시작 위치가 없으면 원본 텍스트 반환
    if orig_start_position is None:
        return orig_text

    # 토큰화된 텍스트 상의 끝 위치를 원본 텍스트 상의 위치로 변환
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    # 끝 위치가 없으면 원본 텍스트 반환
    if orig_end_position is None:
        return orig_text

    # 최종 예측된 텍스트 추출
    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def confirm_preds(nbest_json):
    #unsuccessful attempt at trying to predict for how many and True or false type of questions
    subs = [ 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine','ten', 'eleven', 'twelve', 'true', 'false']
    ori = nbest_json[0]['text']
    if len(ori) < 2:  
        for e in nbest_json[1:]:
            if _normalize_answer(e['text']) in subs:
                return e['text']
        return 'unknown'
    return ori

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
