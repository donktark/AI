# LSTM Text Generation
> [🔗LSTM 텍스트 생성](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/)   
[🔗Winnie-the-Pooh 텍스트 파일](https://www.gutenberg.org/ebooks/67098)
<pre>
<b>  작업 환경 </b> 
OS      : Ubuntu Linux 20.04
CPU     : 8
Memory  : 64GB
GPU     : 비활성화
</pre>
## 라이브러리
```python
import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
```
## 소문자 변환 및 단어사전 추출
```python
# 파일 읽기, 소문자로 변환
file_name ='./Winnie-the-Pooh.txt'
raw_text = open(file_name, 'r', encoding='utf-8').read()
lower_text = raw_text.lower()

# chars -> integer mapping
chars = sorted(list(set(lower_text))) 
chars_to_int = dict((c, i) for i, c in enumerate(chars))
print(chars_to_int) #파일 내에 있는 고유 문자를 단어사전으로 저장함

n_chars = len(lower_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
```
파일을 소문자로 변환한 뒤에 고유한 문자열 하나씩 각각의 단어사전으로 저장한다.  
학습 시 이 단어사전을 기준으로 토큰을 만들게 된다.

## 학습 데이터 전처리
```python
seq_len = 100
X_data = []
Y_data = []
for i in range(0, n_chars - seq_len, 1):
    seq_in = lower_text[i:i + seq_len]
    seq_out = lower_text[i + seq_len]
    X_data.append([chars_to_int[c] for c in seq_in])
    Y_data.append(chars_to_int[seq_out])
n_patterns = len(X_data)
print("Total Patterns: ", n_patterns)
```
X와 Y 데이터를 만든다. 시퀀스 길이만큼 잘라 X를 만들고 그 위에 오는 단일 문자열 하나를 Y로 지정해 넣는다. 단어 사전에 따라 문자열을 숫자 벡터로 전환한다.

```python
X = torch.tensor(X_data, dtype=torch.float32).reshape(n_patterns, seq_len, 1) #tensor를 통해 각 문자들을 나눔
X = X / float(n_vocab) #vocab수로 나눠서 정규화 (0-1로 만듦 <- Pytorch는 0~1 값 선호)
y = torch.tensor(Y_data)
print(X.shape, y.shape)
```
Torch에서 학습할 수 있도록 Tensor로 바꾼 뒤에 정규화를 한다.

## 모델 학습
```python
class BuildModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
```
모델을 빌드한다. LSTM, 드랍아웃, 출력층을 만든다.  
은닉층은 한 개만 사용한다.
```python
epochs = 20
batch_size = 128
model = BuildModel()

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #validation
    model.eval()
    loss = 0
    with torch.no_grad():                   #학습을 멈추고 모델 평가 (validation)
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:                #현재 저장된 loss보다 저 작게 나오면 모델 저장
            best_loss = loss
            best_model = model.state_dict()
        print('Epoch %d: Cross-entrophy: %.4f' % (epoch, loss))

torch.save([best_model, chars_to_int], "./model_checkpoints/text_generator.pth")
```
학습을 진행한다. 옵티마이저는 Adam, 손실함수는 Cross Entrophy로 지정한다.  
한 epoch마다 loss 값을 출력해 학습상태를 표시한다.
## 텍스트 생성
```python
best_model, chars_to_int = torch.load("./model_checkpoints/text_generator.pth")
n_vocab = len(chars_to_int)
int_to_chars = dict((i, c) for c, i in chars_to_int.items())

start = np.random.randint(0, len(raw_text)-seq_len)
prompt = lower_text[start:start+seq_len]            #문서 중 문구를 랜덤으로 뽑아서 seq_length 만큼 프롬프트로 입력
pattern = [chars_to_int[c] for c in prompt]

model.eval()
print("Prompt: \n %s  _____" % prompt)
with torch.no_grad():
    for i in range(1000):           #모델을 1000번 반복함. 즉 글자수가 1000이 될때까지 텍스트를 생성
        # 입력을 tensor로
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype = torch.float32)

        # 문자 logit 계산
        prediction = model(x)

        # 인덱스를 문자로 변환
        index = int(prediction.argmax())
        result = int_to_chars[index]
        print(result, end = "")

        # 프롬프트에 단어를 추가
        pattern.append(index)
        pattern=pattern[1:]
print()
print("__Fin__")
```
<pre>
___<b>프롬프트</b>
 balloon?"

"yes, i just said to myself coming along: 'i wonder if christopher robin
has such a thing  

___<b>출력</b>
 and then the boll hs toe tore 

"ioo the tooe   "a larpy hirteey," 
"yhs, yhu kave ho in " 
"the oor toe lo the hertoon " 
"yhs, yhu iave g vorl ho in an hnnettor b aoa oo ho so bo the bootoe " 
"the oere po he poe"thre " soo tere airistopher robin sas shit the borto tf the pore of the sore of the horest, 
"io whu den tooh " 
"yhet so he soo to lave a lort of toeek to an toeetllng "

</pre>
현재 모델은 입력이 주어지면 한 글자를 생성해내므로 원하는 글자수 만큼의 출력을 얻기위해서는 그만큼 모델을 반복하여 실행한다.  
각 반복마다 로짓을 계산하여 문자로 변환한 뒤 프롬프트에 추가해 다음 회차에서 다시 예측을 한다.