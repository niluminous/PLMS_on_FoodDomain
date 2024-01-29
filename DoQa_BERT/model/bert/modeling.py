from transformers import BertModel, BertPreTrainedModel, RobertaForQuestionAnswering
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import random
import torch

class Multi_linear_layer(nn.Module):
    def __init__(self,
                 n_layers,
                 input_size, # 입력 크기
                 hidden_size, # 은닉 크기
                 output_size, # 출력 크기
                 activation=None): # 활성화 함수
        super(Multi_linear_layer, self).__init__() # 상속받은 nn.Module의 초기화 메서드 호출
        self.linears = nn.ModuleList() # nn.ModuleList를 사용하여 선형 레이어를 저장할 리스트 생성
        self.linears.append(nn.Linear(input_size, hidden_size)) # 첫 번째 선형 레이어 추가 (입력 크기에서 은닉 크기로)
        for _ in range(1, n_layers - 1): # 주어진 은닉 레이어 수만큼 반복
            self.linears.append(nn.Linear(hidden_size, hidden_size)) # 은닉 크기에서 은닉 크기로의 선형 레이어 추가
        self.linears.append(nn.Linear(hidden_size, output_size)) # 마지막 선형 레이어 추가 (은닉 크기에서 출력 크기로)
        self.activation = getattr(F, activation) # 주어진 활성화 함수 문자열을 사용하여 활성화 함수 설정

    def forward(self, x): # 순전파 연산 정의
        for linear in self.linears[:-1]: # 마지막 레이어를 제외한 각 레이어에 대해 반복
            x = self.activation(linear(x)) # 활성화 함수를 적용한 선형 변환 수행
        linear = self.linears[-1] # 마지막 레이어 선택
        x = linear(x) # 마지막 레이어에 대해 선형 변환 수행
        return x # 최종 출력 반환


class BertForQuAC(BertPreTrainedModel):
    def __init__(
            self,
            config,
            output_attentions=False,
            keep_multihead_output=False,
            n_layers=2,
            activation='relu',
            beta=100,
    ):
        super(BertForQuAC, self).__init__(config) # 상속받은 부모 클래스의 초기화 메서드 호출
        self.output_attentions = output_attentions # 어텐션 출력 여부 설정
        self.bert = BertModel(config) # BERT 모델 생성
        hidden_size = config.hidden_size # BERT 모델의 은닉 크기
        self.rational_l = Multi_linear_layer(n_layers, hidden_size,
                                             hidden_size, 1, activation) # rationale 예측을 위한 다층 선형 레이어 생성
        self.logits_l = Multi_linear_layer(n_layers, hidden_size, hidden_size,
                                           2, activation) # 정답 예측을 위한 다층 선형 레이어 생성
        self.unk_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 1,
                                        activation) # unknown 예측을 위한 다층 선형 레이어 생성
        self.attention_l = Multi_linear_layer(n_layers, hidden_size,
                                              hidden_size, 1, activation) # 어텐션 가중치 예측을 위한 다층 선형 레이어 생성
        self.yn_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 2,
                                       activation) # yes/no 예측을 위한 다층 선형 레이어 생성
        self.beta = beta # rational 손실 가중치

        self.init_weights() # 가중치 초기화 메서드 호출

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx = None,
            head_mask=None,
    ):
        # mask some words on inputs_ids
        # if self.training and self.mask_p > 0:
        #     batch_size = input_ids.size(0)
        #     for i in range(batch_size):
        #         len_c, len_qc = token_type_ids[i].sum(
        #             dim=0).detach().item(), attention_mask[i].sum(
        #                 dim=0).detach().item()
        #         masked_idx = random.sample(range(len_qc - len_c, len_qc),
        #                                    int(len_c * self.mask_p))
        #         input_ids[i, masked_idx] = 100

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            # output_all_encoded_layers=False,
            head_mask=head_mask,
        )
        # print(outputs)
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs # 어텐션 출력이 필요한 경우
        else:
            final_hidden=outputs.last_hidden_state # 마지막 은닉 상태
            pooled_output =outputs.pooler_output # 풀링된 출력
        # print("Final_hidden:",final_hidden)
        rational_logits = self.rational_l(final_hidden) # rationale 예측을 위한 다층 선형 레이어 적용
        rational_logits = torch.sigmoid(rational_logits) # sigmoid 함수를 사용하여 0과 1 사이의 값으로 변환

        final_hidden = final_hidden * rational_logits # rationale을 곱하여 적용

        logits = self.logits_l(final_hidden) # 정답 예측을 위한 다층 선형 레이어 적용

        start_logits, end_logits = logits.split(1, dim=-1) # 시작 및 종료 로짓으로 분할

        start_logits, end_logits = start_logits.squeeze(
            -1), end_logits.squeeze(-1) # 마지막 차원을 제거하여 로짓을 얻음

        segment_mask = token_type_ids.type(final_hidden.dtype) # 어텐션 마스크의 데이터 타입을 은닉 상태의 데이터 타입으로 변경

        rational_logits = rational_logits.squeeze(-1) * segment_mask # rationale 예측을 마스킹

        start_logits = start_logits * rational_logits # 시작 로짓에 rationale 적용

        end_logits = end_logits * rational_logits # 종료 로짓에 rationale 적용

        unk_logits = self.unk_l(pooled_output) # unknown 예측을 위한 다층 선형 레이어 적용

        attention = self.attention_l(final_hidden).squeeze(-1) # 어텐션 가중치 예측을 위한 다층 선형 레이어 적용

        attention.data.masked_fill_(attention_mask.eq(0), -float('inf')) # 어텐션 마스크가 0인 부분을 음의 무한대로 채움

        attention = F.softmax(attention, dim=-1) # 소프트맥스 함수를 사용하여 어텐션 가중치를 확률 분포로 변환

        attention_pooled_output = (attention.unsqueeze(-1) *
                                   final_hidden).sum(dim=-2) # 어텐션 가중치를 적용하여 어텐션된 은닉 상태를 얻음

        yn_logits = self.yn_l(attention_pooled_output) # yes/no 예측을 위한 다층 선형 레이어 적용

        yes_logits, no_logits = yn_logits.split(1, dim=-1) # yes와 no 로짓으로 분할

        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf')) # 어텐션 마스크가 0인 부분을 음의 무한대로 채움
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf')) # 어텐션 마스크가 0인 부분을 음의 무한대로 채움

        new_start_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, start_logits), dim=-1) # 예측된 값들을 하나의 텐서로 합침
        new_end_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, end_logits), dim=-1)
        
        if start_positions is not None and end_positions is not None:
            # Ground truth start와 end 위치가 제공되는 경우

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
            # [CLS] 토큰의 인덱스를 제공된 start 및 end 위치에 추가

            # 만약 multi-GPU 환경이면 squeeze가 차원을 추가
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 때로는 start/end 위치가 모델 입력 범위를 벗어날 수 있으므로 이러한 항목들을 무시
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # CrossEntropyLoss는 모델을 훈련하는 데 사용
            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # 손실의 일부로서의 rationale 부분
            alpha = 0.25
            gamma = 2.
            rational_mask = rational_mask.type(final_hidden.dtype)

            rational_loss = -alpha * (
                (1 - rational_logits)**gamma
            ) * rational_mask * torch.log(rational_logits + 1e-7) - (
                1 - alpha) * (rational_logits**gamma) * (
                    1 - rational_mask) * torch.log(1 - rational_logits + 1e-7)

            rational_loss = (rational_loss *
                             segment_mask).sum() / segment_mask.sum()

            # 총 손실은 start, end 및 rationale 손실의 조합 -> rationale 손실에 가중치(factor)가 적용됩니다.
            total_loss = (start_loss +
                          end_loss) / 2 + rational_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits
        # Ground truth가 제공되지 않으면 예측된 logits을 반환





class RobertaForQuAC(RobertaForQuestionAnswering):
    def __init__(
            self,
            config,
            output_attentions=False,
            keep_multihead_output=False,
            n_layers=2,
            activation='relu',
            beta=100,
    ):
        super(RobertaForQuAC, self).__init__(config)
        self.output_attentions = output_attentions
        hidden_size = config.hidden_size
        self.rational_l = Multi_linear_layer(n_layers, hidden_size,
                                             hidden_size, 1, activation)
        self.logits_l = Multi_linear_layer(n_layers, hidden_size, hidden_size,
                                           2, activation)
        self.unk_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 1,
                                        activation)
        self.attention_l = Multi_linear_layer(n_layers, hidden_size,
                                              hidden_size, 1, activation)
        self.yn_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 2,
                                       activation)
        self.beta = beta

        self.init_weights()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx = None,
            head_mask=None,
    ):
        # mask some words on inputs_ids
        # if self.training and self.mask_p > 0:
        #     batch_size = input_ids.size(0)
        #     for i in range(batch_size):
        #         len_c, len_qc = token_type_ids[i].sum(
        #             dim=0).detach().item(), attention_mask[i].sum(
        #                 dim=0).detach().item()
        #         masked_idx = random.sample(range(len_qc - len_c, len_qc),
        #                                    int(len_c * self.mask_p))
        #         input_ids[i, masked_idx] = 100

        outputs = self.roberta(
            input_ids,
            token_type_ids=None, # warning: should we use token_type_ids in roberta?
            attention_mask=attention_mask,
            # output_all_encoded_layers=False,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs
        else:
            final_hidden, pooled_output = outputs

        rational_logits = self.rational_l(final_hidden)
        rational_logits = torch.sigmoid(rational_logits)

        final_hidden = final_hidden * rational_logits

        logits = self.logits_l(final_hidden)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits, end_logits = start_logits.squeeze(
            -1), end_logits.squeeze(-1)

        segment_mask = token_type_ids.type(final_hidden.dtype)

        rational_logits = rational_logits.squeeze(-1) * segment_mask

        start_logits = start_logits * rational_logits

        end_logits = end_logits * rational_logits

        unk_logits = self.unk_l(pooled_output)

        attention = self.attention_l(final_hidden).squeeze(-1)

        attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        attention = F.softmax(attention, dim=-1)

        attention_pooled_output = (attention.unsqueeze(-1) *
                                   final_hidden).sum(dim=-2)

        yn_logits = self.yn_l(attention_pooled_output)

        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        new_start_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, start_logits), dim=-1)
        new_end_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, end_logits), dim=-1)

        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # rational part
            alpha = 0.25
            gamma = 2.
            rational_mask = rational_mask.type(final_hidden.dtype)

            rational_loss = -alpha * (
                (1 - rational_logits)**gamma
            ) * rational_mask * torch.log(rational_logits + 1e-7) - (
                1 - alpha) * (rational_logits**gamma) * (
                    1 - rational_mask) * torch.log(1 - rational_logits + 1e-7)

            rational_loss = (rational_loss *
                             segment_mask).sum() / segment_mask.sum()
            # end

            assert not torch.isnan(rational_loss)

            total_loss = (start_loss +
                          end_loss) / 2 + rational_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits