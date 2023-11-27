import torch
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
import pandas as pd
from sklearn.decomposition import PCA


class SequenceGeneratorCV:
    '''
    data: initial dataframe
    '''
    def __init__(
        self, 
        numeric_features, 
        categorical_features, 
        output_features,
        device,
        normalize_features = [], # feature that should be transformed to resemble normal distribution
        input_width = int(7*24), 
        output_width = 48, 
        val_size = .2, 
        batch_size = 256,
    ): 
        self.val_size, self.batch_size, self.device = (val_size, batch_size, device)
        # zakresy do indeksowania sekwencji
        self.input_width, self.output_width = (input_width, output_width)
        # nazwy kolumn numerycznych, kategoryczne i wyjsciowych do predykcji do modelu
        self.all_numeric_features = numeric_features
        self.numeric_features, self.normalize_features, self.categorical_features = (
            # to split out ones that need normalization
            [i for i in numeric_features if i not in normalize_features],  
            [i for i in normalize_features if i in numeric_features],
            categorical_features
        )
        self.output_features = output_features


    def init_preprocessor(self, sample_df):
        # data processing unit
        preprocessor = ColumnTransformer(transformers=[
            ('pass_city', 'passthrough', ['city']),
            ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True), self.normalize_features),
            ('numeric_transformerr', StandardScaler(), self.numeric_features),
            ('categorical_transformer', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_features),
        ], remainder='drop', verbose_feature_names_out=False)
        preprocessor.set_output(transform='pandas')

        self.preprocessor = Pipeline([
            ('standardize', preprocessor),
        ])

        sample_df = self.preprocessor.fit_transform(sample_df)
        self.input_columns_idx = [i for i,n in enumerate(sample_df.columns) if n not in ['city']]
        self.output_columns_idx = [i for i,n in enumerate(sample_df.columns) if n in self.output_features]

    def split_data(self, df, folds, cv_incremental):
        # finding the city with longest available data so that we can
        # generate correct dataset without data leaking
        cities = df.city.unique()
        longest = len(df)
        for city in cities:
            city_df = df[df.city == city]
            if len(city_df) < longest: longest = len(city_df)  
                
        # spliting by city, assigning new index for correct slicing
        cities_dfs = []

        for city in cities:
            city_df = df[df.city == city].reset_index(drop=True)
            index = pd.RangeIndex(longest - len(city_df), longest)
            city_df.index = index
            cities_dfs.append(city_df.copy())

        # creating cross validation indices
        cv_indices, train_start, split_size = [], 0, longest / folds

        for i in range(1, folds):
            val_start, val_end = int(i*split_size), int((i+1)*split_size)
            cv_indices.append( (slice(train_start, val_start), slice(val_start, val_end) ))
            if cv_incremental: train_start = val_start
        return cities_dfs, cv_indices


    def get_dataloaders(self, data_train, data_val=None):
        dataloader_train = DataLoader(list(zip(*self.sliding_windows(data_train))), self.batch_size, False, num_workers=2)
        if data_val is None: return dataloader_train
        dataloader_val = DataLoader(list(zip(*self.sliding_windows(data_val))), self.batch_size, False, num_workers=2)
        return dataloader_train, dataloader_val


    def sliding_windows(self, df):
        # generuje wektory X i y na podstawie podanego dataframu
        cities = df.city.unique()
        X, y = [], []
        # przez kazde miasto, target na x i y
        for city in cities:
            df_city = df[df.city == city]
            Xs = torch.tensor(df_city.iloc[:, self.input_columns_idx].values, dtype=torch.float32)
            ys = torch.tensor(df_city.iloc[:, self.output_columns_idx].values, dtype=torch.float32)
            
            # po wszystkich mozliwych sekwencjach
            for i in range(self.input_width + self.output_width):
                in_seq_idx = slice(i, (i + self.input_width))
                out_seq_idx = slice((i + self.input_width), (i + self.input_width + self.output_width))
                X.append(Xs[in_seq_idx, :]), y.append(ys[out_seq_idx , :])
        return torch.stack(X).to(self.device), torch.stack(y).to(self.device)


class SequenceGeneratorCVPCA(SequenceGeneratorCV):
    def init_preprocessor(self, sample_df):
        # data processing unit
        preprocessor = ColumnTransformer(transformers=[
            ('pass_city', 'passthrough', ['city']),
            ('power_transformer', PowerTransformer(method='yeo-johnson', standardize=True), self.normalize_features),
            ('numeric_transformerr', StandardScaler(), self.numeric_features),
            ('categorical_transformer', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_features),
        ], remainder='drop', verbose_feature_names_out=False)
        preprocessor.set_output(transform='pandas')

        dim_reduction = ColumnTransformer(transformers=[
            ('pass_city', 'passthrough', ['city']),
            ('reduce_dim', PCA(n_components = 10), self.all_numeric_features),
            ('pass', 'passthrough', self.output_features),
        ], remainder='passthrough', verbose_feature_names_out=False)
        dim_reduction.set_output(transform='pandas')

        self.preprocessor = Pipeline([
            ('standardize', preprocessor),
            ('dim_red', dim_reduction),
        ])

        sample_df = self.preprocessor.fit_transform(sample_df)
        self.input_columns_idx = [i for i,n in enumerate(sample_df.columns) if n not in ['city', *self.output_features]]
        self.output_columns_idx = [i for i,n in enumerate(sample_df.columns) if n in self.output_features]
