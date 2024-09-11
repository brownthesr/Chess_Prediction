import torch
import networkx as nx
import chess
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gen_dataset import ChessDataset
from torch.utils.data import DataLoader
# Function to one-hot encode the position of a square
def encode_position(square):
    rank = square // 8
    file = square % 8
    position_encoding = torch.zeros(16)
    position_encoding[rank] = 1
    position_encoding[8+file] = 1
    return position_encoding

# Function to one-hot encode the piece on a square
def encode_piece(piece):
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                      'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    piece_encoding = torch.zeros(12)
    if piece is not None:
        piece_encoding[piece_to_index[piece.symbol()]] = 1
    return piece_encoding

# Function to generate legal moves for a piece on a given square
def generate_moves(board, square):
    piece = board.piece_at(square)
    if piece is None:
        return []
    legal_moves = []
    for move in board.legal_moves:
        if move.from_square == square:
            legal_moves.append(move.to_square)
    return legal_moves

# Function to generate both edge index and feature tensor for the chess board
def generate_graph_data(board):
    num_nodes = 64
    num_features = 16 + 12  # Position encoding (16) + Piece embedding (12)
    edge_index = []
    features = torch.zeros(num_nodes, num_features)
    mask = torch.zeros(num_nodes)  # Initialize mask

    # Generate features
    for square in range(64):
        features[square, :16] = encode_position(square)
        features[square, 16:] = encode_piece(board.piece_at(square))

    # Generate edge index
    for square in range(64):
        moves = generate_moves(board, square)
        if moves:  # If there are legal moves for the piece on this square
            mask[square] = 1  # Set the corresponding index in mask to 1
            for move in moves:
                edge_index.append([square, move])
        # board.push(chess.Move.null())
        # moves = generate_moves(board, square)
        # if moves:  # If there are legal moves for the piece on this square after making a null move
        #     for move in moves:
        #         edge_index.append([square, move])
        # board.pop()  # Revert the null move

    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index, features, mask
def square_to_num(square):
    """
    Converts a square from algebraic notation (e.g., 'd4') to a numerical representation (0-63).
    """
    file, rank = ord(square[0]) - ord('a'), int(square[1]) - 1
    return rank * 8 + file

def num_to_square(num):
    """
    Converts a numerical representation (0-63) to algebraic notation (e.g., 'd4').
    """
    file, rank = num % 8, num // 8
    return chr(file + ord('a')) + str(rank + 1)
def draw_chessboard_with_graph(board, edge_index):
    G = nx.DiGraph()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[:, i]
        G.add_edge(source.item(), target.item())

    pos = {i: (i % 8 + 0.5,  i // 8 + 0.5) for i in range(64)}

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw chessboard
    colors = ['lightgray', 'white']
    for i in range(8):
        for j in range(8):
            square_color = colors[(i + j) % 2]
            ax.add_patch(plt.Rectangle((i, j), 1, 1, color=square_color, zorder=0))
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                ax.text(i + 0.5, j + 0.5, piece.symbol(), fontsize=20, ha='center', va='center', zorder=1)

    # Draw graph connections
    nx.draw(G, pos, ax=ax, with_labels=False, node_size=0, edge_color='blue', width=0.5)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()
"""
Ok so te idea for what I am going to do is I will use cross entropy to train the loss. The data
X is the state of the board. I will have a two part process. First the model will predict which piece to move
from among the pieces that can move. It will then predict which place to move, from the places it can move to.
I will accomplish this with two models.

the first will output a single number for each node, we then softmax over then and choose the right one. Then we
run the same database but with a one appended to the end if we chose the node to move and a zero otherwise. We
then softmax over the available nodes to move to
"""
def func(x,e):
    return x[:,0]

def get_san(start_square, end_square):
    start_square_name = chess.square_name(start_square)
    end_square_name = chess.square_name(end_square)
    return start_square_name + end_square_name
def move(board,model1,model2):
    edge_index, features,mask = generate_graph_data(board)
    logits = model1(features)
    inf_mask = torch.full(fill_value=-torch.inf,size=(len(features),))
    inf_mask[mask.bool()] = 0
    choice = torch.multinomial(torch.softmax(inf_mask+logits,dim=-1),1)
    # Append a collumn of 0s to the features, making the chosen node have a one there
    new_feats = torch.hstack([features,torch.zeros(64).unsqueeze(-1)])
    new_feats[choice,-1] = 1
    
    logits = model2(new_feats)
    move_mask = generate_moves(board,choice)
    inf_mask = torch.full(fill_value=-torch.inf,size=(len(features),))
    inf_mask[move_mask] = 0
    # print(inf_mask)
    move_choice = torch.multinomial(torch.softmax(logits+inf_mask,dim=-1),1)
    return get_san(choice,move_choice)
def train_models(board,model1,model2,correct_piece,correct_move):
    edge_index, features,mask = generate_graph_data(board)
    logits = model1(features,edge_index)
    inf_mask = torch.full(fill_value=-torch.inf,size=(len(features),))
    inf_mask[mask.bool()] = 0
    choice_prob = torch.softmax(inf_mask+logits,dim=-1)[correct_piece]
    # Append a collumn of 0s to the features, making the chosen node have a one there
    new_feats = torch.hstack([features,torch.zeros(64).unsqueeze(-1)])
    new_feats[correct_piece,-1] = 1
    
    logits = model2(new_feats,edge_index)
    move_mask = generate_moves(board,correct_piece)
    inf_mask = torch.full(fill_value=-torch.inf,size=(len(features),))
    inf_mask[move_mask] = 0
    # print(inf_mask)
    move_choice_prob = torch.softmax(logits+inf_mask,dim=-1)[correct_move]
    return choice_prob,move_choice_prob

class Trans(torch.nn.Module):
    def __init__(self,n_layers,n_in,n_hidden=64,nhead=4):
        super().__init__()
        self.embedder = torch.nn.Linear(n_in,n_hidden)
        decoder_layer = nn.TransformerEncoderLayer(n_hidden,nhead,batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer,n_layers)
        self.out = nn.Linear(n_hidden,1)
        
    def forward(self, X):
        out = self.embedder(X)
        out = self.decoder(out)
        out = self.out(out).squeeze(-1)
        return out
import pandas as pd
from tqdm import tqdm
# max allowable data size (currently 1 million)
import sys
device = "cuda"
piece_picker = Trans(6,28).to(device)
move_picker = Trans(6,29).to(device)
batch_size = 64
opt = torch.optim.Adam(list(piece_picker.parameters())+list(move_picker.parameters()),lr=5e-4)
board = chess.Board()
# for index, row in tqdm(df.iterrows()):
#     board = chess.Board()
#     san_move = row[i+2]
#     xuci_move = board.parse_san(san_move)
print("obatining dataset")
dataset = ChessDataset("chess_dataset.pkl")
print("obtained dataset")
dl = DataLoader(dataset,batch_size=batch_size,drop_last=True)
losses = []
for epoch in range(10):
    for features,ys,mask1,mask2,meta in tqdm(dl):
        features,ys,mask1,mask2 = features.to(device),ys.to(device),mask1.to(device),mask2.to(device)
        opt.zero_grad()
        logits = piece_picker(features)
        inf_mask = torch.full(fill_value=-torch.inf,size=(batch_size,features.shape[-2],)).to(device)
        inf_mask[mask1.bool()] = 0
        choice_prob = torch.softmax(inf_mask+logits,dim=-1)[torch.arange(batch_size),ys[:,0]]
        # Append a collumn of 0s to the features, making the chosen node have a one there
        new_feats = torch.cat([features,torch.zeros(64).unsqueeze(-1).unsqueeze(0).repeat(batch_size,1,1).to(device)],dim=-1).to(device)
        # sys.exit()
        new_feats[torch.arange(batch_size),ys[:,0],-1] = 1
        
        logits = move_picker(new_feats)
        inf_mask = torch.full(fill_value=-torch.inf,size=(batch_size,64,)).to(device)
        inf_mask[mask2.bool()] = 0
        # print(inf_mask)
        move_choice_prob = torch.softmax(logits+inf_mask,dim=-1)[torch.arange(batch_size),ys[:,1]]
        loss = -torch.log(choice_prob) - torch.log(move_choice_prob)
        loss = loss.sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()
plt.plot(losses)
plt.savefig("here.png")  
move_picker,piece_picker = move_picker.to("cpu"),piece_picker.to("cpu")         
board = chess.Board()
for i in range(60):
    a = move(board,piece_picker,move_picker)
    board.push_san(a)
    edge_index,_,_ = generate_graph_data(board)
    draw_chessboard_with_graph(board,edge_index)