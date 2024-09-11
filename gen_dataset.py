from torch.utils.data import Dataset, DataLoader
import pickle
import chess
import torch
from tqdm import tqdm

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
        board.push(chess.Move.null())
        moves = generate_moves(board, square)
        if moves:  # If there are legal moves for the piece on this square after making a null move
            for move in moves:
                edge_index.append([square, move])
        board.pop()  # Revert the null move

    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index, features, mask
class ChessDataset (Dataset):
    def __init__(self,file):
        with open(file, 'rb') as f:
            self.dataset_list = pickle.load(f)
    def __len__(self):
        return len(self.dataset_list)
    def __getitem__(self,idx):
        # Return features, labels, mask1,mask2, metadata
        return self.dataset_list[idx][0],self.dataset_list[idx][1], self.dataset_list[idx][2], self.dataset_list[idx][3],self.dataset_list[idx][4]
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

if __name__ == "__main__":
    # for a in dl:
    #     break
    # print(a[3].shape)
    max_data_size = 30000

    path = './all_with_filtered_anotations_since1998.txt'

    data_final = []

    with open(path) as file:
        count = 0
        for i,line in tqdm(enumerate(file)):
            # print(i)
            # first few lines are not useful
            if count <= 4:
                count += 1
                continue

            row = line.split(' ')
            
            # row[1] is date
            # row[2] is result
            # row[17] is W1
            # row[18] is B1 and so on ....
            # taking first 6 moves
            board = chess.Board()
            for k in range(100):
                try:
                    san_move = row[k+17].split(".")[1]
                    uci_move = board.parse_san(san_move)

                    explicit_move = str(chess.Move.from_uci(uci_move.uci()))
                    piece = square_to_num(explicit_move[:2])
                    movement = square_to_num(explicit_move[2:])
                    y = torch.Tensor([piece,movement]).long()
                    meta_data =[row[2],(k+1)%2]
                    edge_index, features,mask = generate_graph_data(board)
                    mask2 = torch.zeros(64)
                    mask2[generate_moves(board,piece)]=1
                    data_final.append((features,y,mask,mask2,meta_data))
                    board.push_san(san_move)
                except:
                    continue
            if i > max_data_size:
                break
    print(len(data_final))
    with open('chess_dataset.pkl', 'wb') as f:
        pickle.dump(data_final, f)

            