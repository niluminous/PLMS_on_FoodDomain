from models.bert.modeling import BertForQuAC, RobertaForQuAC
from transformers import AutoTokenizer
from models.bert.run_quac_dataset_utils import read_partial_quac_examples_extern, read_one_quac_example_extern, convert_one_example_to_features, recover_predicted_answer, RawResult

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class BertOrg():
    def __init__(self, args): # 객체 초기화 메서드
        self.args = args # 전달된 인자를 객체의 속성으로 저장
        self.model = BertForQuAC.from_pretrained(self.args.model_name_or_path) # BERT 모델 초기화
        self.QA_history = [] # 질문-답변 히스토리 초기화
        torch.manual_seed(args.seed) # 난수 생성기 시드 설정
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu") # CUDA 사용 가능 여부에 따라 디바이스 설정
        self.model = self.model.to(self.device) # 모델을 설정한 디바이스로 이동
    
    def tokenizer(self): # BERT 토크나이저 반환 메서드
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, do_lower_case=self.args.do_lower_case) # BERT 토크나이저 초기화
        return tokenizer
    
    def load_partial_examples(self, partial_eval_examples_file): # 부분적인 데이터셋 로드 메서드
        paragraphs = read_partial_quac_examples_extern(partial_eval_examples_file) # 외부 함수를 사용하여 데이터 로드
        return paragraphs
    
    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer): # 자동 턴에 대한 예측 수행 메서드
        question = partial_example.question_text # 질문 텍스트 추출
        turn = int(partial_example.qas_id.split("#")[1]) # 턴 정보 추출
        example = read_one_quac_example_extern(partial_example, self.QA_history, history_len=2, add_QA_tag=False) # 외부 함수를 사용하여 예제 읽기
        
        curr_eval_features, next_unique_id= convert_one_example_to_features(example=example, unique_id=unique_id, example_index=example_idx, tokenizer=tokenizer, max_seq_length=self.args.max_seq_length,
                                    doc_stride=self.args.doc_stride, max_query_length=self.args.max_query_length) # 예제를 모델 입력 특성으로 변환
        all_input_ids = torch.tensor([f.input_ids for f in curr_eval_features],
                                            dtype=torch.long) # 입력 아이디들을 텐서로 변환
        all_input_mask = torch.tensor([f.input_mask for f in curr_eval_features],
                                    dtype=torch.long) # 입력 마스크들을 텐서로 변환
        all_segment_ids = torch.tensor([f.segment_ids for f in curr_eval_features],
                                    dtype=torch.long) # 세그먼트 아이디들을 텐서로 변환
        all_feature_index = torch.arange(all_input_ids.size(0),
                                        dtype=torch.long) # 특성 인덱스를 텐서로 생성
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_feature_index) # 데이터셋 생성
        # 예측을 위한 데이터 로더 생성
        eval_dataloader = DataLoader(eval_data,
                                    sampler=None,
                                    batch_size=1)
        curr_results = [] # 현재 결과를 저장할 리스트
        # 현재 예제에 대한 예측 실행
        for input_ids, input_mask, segment_ids, feature_indices in eval_dataloader:

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            print(type(input_ids[0]), type(input_mask[0]), type(segment_ids[0]))
            # 로짓이 리스트의 한 항목인 것으로 가정
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_yes_logits, batch_no_logits, batch_unk_logits = self.model(
                    input_ids, segment_ids, input_mask)
            for i, feature_index in enumerate(feature_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                eval_feature = curr_eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                curr_results.append(
                    RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                yes_logits=yes_logits,
                                no_logits=no_logits,
                                unk_logits=unk_logits))
        predicted_answer = recover_predicted_answer(
            example=example, features=curr_eval_features, results=curr_results, tokenizer=tokenizer, n_best_size=self.args.n_best_size, max_answer_length=self.args.max_answer_length,
            do_lower_case=self.args.do_lower_case, verbose_logging=self.args.verbose_logging) # 예측된 답변 복원
        self.QA_history.append((turn, question, (predicted_answer, None, None))) # 히스토리에 현재 턴의 정보 추가
        return predicted_answer, next_unique_id

class RobertaOrg():
    def __init__(self, args, device):
        self.args = args
        self.model = RobertaForQuAC.from_pretrained(self.args.model_name_or_path)
        self.QA_history = []
        self.device = device
    
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)
        return tokenizer
    
    def load_partial_examples(self, cached_partial_eval_examples_file):
        paragraphs = read_partial_quac_examples_extern(cached_partial_eval_examples_file)
        return paragraphs
    
    def predict_one_human_turn(self, paragraph, question):
        return
    
    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx, tokenizer):
        question = partial_example.question_text
        turn = int(partial_example.qas_id.split("#")[1])
        example = read_one_quac_example_extern(partial_example, self.QA_history, history_len=2, add_QA_tag=False)
        
        curr_eval_features, next_unique_id= convert_one_example_to_features(example=example, unique_id=unique_id, example_index=example_idx, tokenizer=tokenizer, max_seq_length=self.args.max_seq_length,
                                    doc_stride=self.args.doc_stride, max_query_length=self.args.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in curr_eval_features],
                                            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in curr_eval_features],
                                    dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in curr_eval_features],
                                    dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0),
                                        dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_feature_index)
        # Run prediction for full data

        eval_dataloader = DataLoader(eval_data,
                                    sampler=None,
                                    batch_size=1)
        curr_results = []
        # Run prediction for current example
        for input_ids, input_mask, segment_ids, feature_indices in eval_dataloader:

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            # Assume the logits are a list of one item
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_yes_logits, batch_no_logits, batch_unk_logits = self.model(
                    input_ids, segment_ids, input_mask)
            for i, feature_index in enumerate(feature_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                eval_feature = curr_eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                curr_results.append(
                    RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                yes_logits=yes_logits,
                                no_logits=no_logits,
                                unk_logits=unk_logits))
        predicted_answer = recover_predicted_answer(
            example=example, features=curr_eval_features, results=curr_results, n_best_size=self.args.n_best_size, max_answer_length=self.args.max_answer_length,
            do_lower_case=self.args.do_lower_case, verbose_logging=self.args.verbose_logging)
        self.QA_history.append((turn, question, (predicted_answer, None, None)))
        return predicted_answer, next_unique_id