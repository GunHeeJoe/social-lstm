import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        # social_tensor.shape = 이웃수(N) x (grid_size * grid_size * rnn_size)
        # 특정 이웃의 특정 (grid_x, grid_y)에서의 hiddent_state정보를 추출함(단 flatten구조로)
        # 그니까 social_tensor[0][0~rnn_size]는 0번 이웃의 (0,0)셀의 hiddent_state정보
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor

            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
    
        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]

        numNodes = len(look_up)
    
        # outputs.shape = [seq*numNode, 5] : 각 시점별 보행자들의 위치를 예측하기 위해 이변량가우시안분포를 추정
        # 이변량가우시안분포에서 사용할 Mean(x),Mean(y), VAR(x), VAR(y), corr(x,y)인 총 5개 예측
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):
            # Peds present in the current frame

            #print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            #print("list of nodes")

            #nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            #print("lookup table :%s"% look_up)
            # 보행자 id를 0번부터 재부여
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))

            if self.use_cuda:            
                corr_index = corr_index.cuda()

            #print("list of nodes: %s"%nodeIDs)
            #print("trans: %s"%corr_index)
            #if self.use_cuda:
             #   list_of_nodes = list_of_nodes.cuda()


            #print(list_of_nodes.data)
            # Select the corresponding input positions
            # 특정 시점 보행자의 (x,y)좌표
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            # grid_current.shape = N x N x (grid_size*grid_size)
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            # hidden_states.shape = N x hidden_dimension
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs
            # [N, 2] -> [N, embedding_dim]
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            # Embed the social tensor
            # [N, grid_size * grid_size * hidden_state_size] -> [N, embedding_dim]
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            
            # Concat input
            # [N, embedding_dim * 2]
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                # h_node : [N, rnn_size] 각 선수별 hidden_state
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))

            # Compute the output
            # h_nodes : [N, rnn_size] // 특정 시점 각 보행자들의 hidden_state
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            # 현재시점에 trajectory를 갖는 사람들만 hidden_state 업데이트
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
