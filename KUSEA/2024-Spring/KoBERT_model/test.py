import torch
import numpy as np

model= torch.load('model.pht')

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

#train된 모델을 통해 output을 확인할 때 사용
def testModel(model, seq):
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(tok, max_len, vocab, pad=True, pair=False)
    tokenized = transform(tmp)

    model.eval()
    result = model(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
    idx = result.argmax().cpu().item()
    print("입력 내용의 연관 경제 용어는 \'"+ result_dict[idx] + '\'입니다.')
    #print("신뢰도는:", "{:.2f}%".format(softmax(result,idx)))

seq = input('경제 용어에 대한 설명을 쓰시오.')
testModel(model, seq)
