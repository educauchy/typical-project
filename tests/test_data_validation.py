import pandas as pd
import pytest


class TestDataValidation(object):
    splits = {
        'train': {
            'path': './data/processed/winequality-red-train.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'quality']
        },
        'test': {
            'path': './data/processed/winequality-red-test.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'quality']
        },
        'scoring': {
            'path': './data/processed/winequality-red-scoring.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'id']
        },
        'scoring_result': {
            'path': './data/processed/winequality-red-scoring-result.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'score', 'id']
        }
    }

    def test_columns_presence(self):
        error_splits = []
        for curr_split, curr_metadata in self.splits.items():
            df_columns = (pd.read_csv(curr_metadata['path'], sep=';')).columns.tolist()
            needed_cols = curr_metadata['needed_cols']
            df_columns.sort()
            needed_cols.sort()
            if df_columns != needed_cols:
                error_splits.append(curr_split)

        assert not error_splits, "Splits without necessary columns: {}".format("\n".join(error_splits))

    def test_columns_intervals(self):
        col_ranges = {
            'quality': [1, 10]
        }
        errors = []
        for curr_split, curr_metadata in self.splits.items():
            df = pd.read_csv(curr_metadata['path'], sep=';')
            for curr_col, rng in col_ranges.items():
                if curr_col in df.columns:
                    available_range_df = df.query('{} <= {} <= {}'.format(rng[0], curr_col, rng[1]))
                    if df.shape[0] != available_range_df.shape[0]:
                        errors.append('{}__{}'.format(curr_split, curr_col))

        assert not errors, "Errors occured in splits/columns: {}".format("\n".join(errors))

    def test_data_drift(self):
        # check shift in distribution before and after some date
        pass

    def test_data_freshness(self):
        # splits_to_test = self.splits.copy()
        # splits_to_test.pop('scoring')
        # splits_to_test.pop('scoring_result')
        # error_splits = []
        # curr_dates = ['']
        #
        # for curr_split, curr_metadata in self.splits.items():
        #     report_dts = (pd.read_csv(curr_metadata['path'], sep=';', usecols=['report_dt'])).to_numpy().unique().astype(pd.datetime)
        # assert not error_splits, "Splits without fresh data: {}".format("\n".join(error_splits))
        pass

    def test_split_intervals_consistency(self):
        pass


if __name__ == '__main__':
    # TestDataQuality().test_columns_presence()
    # TestDataQuality().test_columns_intervals()
    TestDataValidation().test_data_freshness()
