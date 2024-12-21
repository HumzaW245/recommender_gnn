
import GNNRecommender as gnn
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import config
import data.data as pipeLine
import wandb
import argparse

def evaluate(args):
  
  wandb.login()
  #------------------------------------------------------>INITIALIZING WANDB PROJECT NAME AND NAME OF RUN <--------------------------------------------------
  wandb.init(project="GNN_Project_MAT6495", name='MovieLens1M: ' + args.wandbNameSuffix)

  #Making sure config dictionary is update to have values expected
  print(f'\n\n\nThe configuration for this run is as follows: \n {args} \n\n\n')


  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print(device)

  x_train, edge_index_train, ratings_train, x_test, edge_index_test, ratings_test = pipeLine.get_movielens1m_train_test(args)
  ratings_tensor_train = torch.tensor(ratings_train['rating'].values, dtype=torch.float)
  print(f'Ratings tensor shape (used to calc loss with model output) {ratings_tensor_train.shape}')

  model = gnn.GNNRecommender(x_train.size(1), args.hidden_channels)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  
  for epoch in range(args.epochs):
      model.train()
      optimizer.zero_grad()
      out = model(x_train, edge_index_train)
      print(out.shape)
      print(ratings_tensor_train.shape)
      assert False
      loss = criterion(out, ratings_tensor_train)
      loss.backward()
      optimizer.step()
      wandb.log(f'Epoch {epoch+1}, Loss: {loss.item()}')
      print(f'Epoch {epoch+1}, Loss: {loss.item()}')
      

  ratings_tensor_test = torch.tensor(ratings_test['rating'].values, dtype=torch.float)
  model.eval() 
  with torch.no_grad(): 
    test_out = model(x_test, edge_index_test) 
    test_loss = criterion(test_out, ratings_tensor_test) 
    print(f'Epoch {epoch+1}, Test Loss: {test_loss.item()}')
    # Calculate accuracy 
    # Example threshold, adjust as necessary 
    threshold = args.accuracyTheshold 
    test_predictions = torch.round(test_out).cpu().numpy() 
    test_actuals = ratings_tensor_test.cpu().numpy() 
    accuracy = (test_predictions == test_actuals).mean() 
    print(f'Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%')
    wandb.log(f'Epoch {epoch+1}, Test Accuracy: {accuracy * 100:.2f}%')


'''
Executing Run (optional: with a custom config)
'''
if __name__ == '__main__':
    print('STARTING PROGRAM')
    args = config.parse_args()

    evaluate(args)