require 'nngraph'

function lstm(prev_h, input, prev_c)
  rnn_size = input:size(1)
  local function sumInput()
    local h_input = nn.Linear( rnn_size, rnn_size)(input)
    local h_pre = nn.Linear( rnn_size, rnn_size)(prev_h)
    
    return sumIn = nn.CAddTable()({h_input, h_pre})
  end
  
  local input_gate = nn.Sigmoid()(sumInput())

  local forget_gate = nn.Sigmoid()(sumInput())
  local curr_cell = nn.CAddTable()({ nn.CMulTable()({forget_gate, prev_c}), nn.CMulTable()({input_gate, nn.Tanh()(sunInput())}) })
  local output_gate = nn.Sigmoid()(sumInput())
  local curr_h = nn.CAddTable()({output_gate, nn.Tanh()(curr_cell)})

  return curr_h, curr_cell
end

