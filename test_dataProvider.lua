-- test_dataProvider.lua
dofile '../hko/opts-hko.lua'
dofile '../torch_extra/dataProvider.lua'

data_path = '../helper/'	
datasetSeq= getdataSeq_hko('train', data_path)
darasetSeq_valid = getdataSeq_hko('valid', data_path)
index = 10
sample = datasetSeq[index] 
print(sample)
sample = darasetSeq_valid[index]
print(sample)