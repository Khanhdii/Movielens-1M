import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, movies_file, binary_classification=1):
        self.ratings = pd.read_csv(ratings_file,
                                   header=None,
                                   sep='::',
                                   names=['userId', 'movieId', 'rating', 'timestamp'],
                                   engine='python',
                                   encoding='latin-1')

        self.movies = pd.read_csv(movies_file,
                                  header=None,
                                  sep='::',
                                  names=['movieId', 'title', 'genres'],
                                  engine='python',
                                  encoding='latin-1')

        # Mã hóa dữ liệu người dùng và phim
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        self.ratings['user'] = self.user_encoder.fit_transform(self.ratings['userId'])
        self.ratings['movie'] = self.movie_encoder.fit_transform(self.ratings['movieId'])

        # Convert rating based on binary classification flag
        if binary_classification == 1:
            self.ratings['rating'] = (self.ratings['rating'] > 3).astype(int)
        else:
            self.ratings['rating'] = (self.ratings['rating'] - 1).astype(int)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = torch.tensor(self.ratings.iloc[idx]['user'], dtype=torch.long)
        movie = torch.tensor(self.ratings.iloc[idx]['movie'], dtype=torch.long)
        rating = torch.tensor(self.ratings.iloc[idx]['rating'], dtype=torch.long)
        return user, movie, rating


def get_data_loaders(ratings_file, movies_file, batch_size=64, binary_classification=1):
    dataset = MovieLensDataset(ratings_file, movies_file, binary_classification)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optionally, you can create validation loader too
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
