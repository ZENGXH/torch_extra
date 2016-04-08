-- test_selfFeedSequencer.lua
require 'nn'
require 'rnn'
dofile 'SelfFeedSequencer.lua'
net = nn.SelfFeedSequencer(nn.Mul(torch.Tensor{2}))
tt = torch.Tensor(2):fill(0)
t = torch.Tensor(2):fill(1)
inputTable = {t, tt, tt}
out = net:forward({t, tt, tt})

print(out[1])
print(out[2])
print(out[3])

print('is input Table update?')
print(inputTable[2])