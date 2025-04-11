import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from einops import rearrange, repeat

class ProfitDataset(Dataset):
    def __init__(self, features, profits):
        self.features = features
        self.profits = profits

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'profit': torch.tensor(self.profits[idx], dtype=torch.float)
        }

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

def main():
    num_epochs = 100

    variables = ['PerFTSE20', 'AVGPTS3_20', 'SharpeRatio20', 'DurationRate20',
                 'ProfitRate20', 'WinTradeRate20', 'ProfitxDur20', 'PassAvgReturn',
                 'AvgShortSales20', 'TradFQ20', 'Period', 'accountid',
                 'NumTrades', 'AvgOpen20', 'DurationRatio20', 'OrderCloseRate20']

    train_set = pd.read_csv('data/lcg_train.csv')[:1000]
    val_set = pd.read_csv('data/lcg_val.csv')
    test_set = pd.read_csv('data/lcg_test.csv')

    train_set = ProfitDataset(train_set[variables].values, train_set['anomaly'])
    val_set = ProfitDataset(val_set[variables].values, val_set['anomaly'])
    test_set = ProfitDataset(test_set[variables].values, test_set['anomaly'])

    # Create a DataLoader
    batch_size = 32  # You can adjust the batch size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = NumericalEmbedder(32, len(variables))
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            features = batch['features']
            profits = batch['profit']

            # Forward pass
            print(features.shape)
            outputs = model(features)
            print(outputs.shape)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), profits)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loss
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                profits = batch['profit']
                outputs = model(features)

                loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), profits)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        print('Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss))

    # Test loss
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            profits = batch['profit']

            outputs = model(features)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), profits)
            test_losses.append(loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    print('Test Loss: {:.4f}'.format(test_loss))

    # Save model
    torch.save(model.state_dict(), 'storage/profit_embedding.pt')
    # predict on test set and evaluate the F1 score
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            outputs = model(features)
            y_pred += list(torch.sigmoid(outputs.squeeze()).numpy())
    test_set = pd.read_csv('data/lcg_test.csv')
    test_set['pred_proba'] = y_pred
    test_set['pred'] = test_set['pred_proba'].apply(lambda x: 1 if x > 0.5 else 0)
    # calculate F1
    from sklearn.metrics import f1_score
    f1 = f1_score(test_set['anomaly'], test_set['pred'], average='macro')


if __name__ == '__main__':
    main()