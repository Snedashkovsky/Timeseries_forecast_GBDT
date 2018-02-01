import pandas as pd
import numpy as np
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn import ensemble, cross_validation, learning_curve, metrics
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def add_columns_to_df(input_df,
                      columns_for_analysis,
                      datetime_column):

    holydays =      np.array(pd.read_csv('holydays_data/holydays.csv', header=0))
    big_holydays =  np.array(pd.read_csv('holydays_data/big_holydays.csv', header=0))
    near_holydays = np.array(pd.read_csv('holydays_data/near_holydays.csv', header=0))
    pre_holydays =  np.array(pd.read_csv('holydays_data/pre_holydays.csv', header=0))

    if 'year' in columns_for_analysis:
        input_df.loc[:, 'year'] = input_df[datetime_column].map(lambda x: x.year)
    if 'month' in columns_for_analysis:
        input_df.loc[:, 'month'] = input_df[datetime_column].map(lambda x: x.month)
    if ('day' in columns_for_analysis) or \
       ('week_of_month' in columns_for_analysis):
        input_df.loc[:, 'day'] = input_df[datetime_column].map(lambda x: x.day)
    if 'day_of_the_year' in columns_for_analysis:
        input_df.loc[:, 'day_of_the_year'] = input_df[datetime_column].map(lambda x: int(x.strftime('%j')))
    if 'weekday' in columns_for_analysis:
        input_df.loc[:, 'weekday'] = input_df[datetime_column].map(lambda x: x.weekday())
    if 'week' in columns_for_analysis:
        input_df.loc[:, 'week'] = input_df[datetime_column].map(lambda x: x.week)
    if 'week_of_month' in columns_for_analysis:
        input_df.loc[:, 'week_of_month'] = input_df['day'] // 7
    if 'holydays' in columns_for_analysis:
        input_df.loc[:, 'holydays'] = input_df[datetime_column].map(lambda x: [x.day, x.month] in holydays)
    if 'big_holydays' in columns_for_analysis:
        input_df.loc[:, 'big_holydays'] = input_df[datetime_column].map(lambda x: [x.day, x.month] in big_holydays)
    if 'near_holydays' in columns_for_analysis:
        input_df.loc[:, 'near_holydays'] = input_df[datetime_column].map(lambda x: [x.day, x.month, x.year] in near_holydays)
    if 'pre_holydays' in columns_for_analysis:
        input_df.loc[:, 'pre_holydays'] = input_df[datetime_column].map(lambda x: [x.day, x.month, x.year] in pre_holydays)
    if ('hour' in columns_for_analysis) or\
       ('minute_of_day' in columns_for_analysis):
        input_df['hour'] = input_df[datetime_column].map(lambda x: x.hour)
    if ('minute' in columns_for_analysis) or\
       ('minute_of_day' in columns_for_analysis):
        input_df['minute'] = input_df[datetime_column].map(lambda x: x.minute)
    if 'minute_of_day' in columns_for_analysis:
        input_df['minute_of_day'] = input_df['minute'] + 60 * input_df['hour']

    return input_df


def predict_by_date_test(input_df,
                         date_start_predict,
                         date_column='date',
                         value_column='value',
                         columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                               'holydays', 'near_holydays', 'big_holydays', 'pre_holydays'),
                         learning_rate=0.02,
                         max_depth=3,
                         n_estimators=900,
                         min_child_weight=4,
                         analysis_algoritm='XGB'):

    return predict_test(input_df,
                        datetime_start_predict=date_start_predict,
                        datetime_column=date_column,
                        value_column=value_column,
                        columns_for_analysis=columns_for_analysis,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        min_child_weight=min_child_weight,
                        analysis_algoritm=analysis_algoritm)


def predict_by_datetime_test(input_df,
                             datetime_start_predict,
                             datetime_column='datetime',
                             value_column='value',
                             columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                                   'holydays', 'near_holydays', 'big_holydays', 'pre_holydays',
                                                   'hour', 'minute', 'minute_of_day'),
                             learning_rate=0.02,
                             max_depth=3,
                             n_estimators=900,
                             min_child_weight=4,
                             analysis_algoritm='XGB'):

    return predict_test(input_df,
                        datetime_start_predict=datetime_start_predict,
                        datetime_column=datetime_column,
                        value_column=value_column,
                        columns_for_analysis=columns_for_analysis,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        min_child_weight=min_child_weight,
                        analysis_algoritm=analysis_algoritm)


def predict_test(input_df,
                 datetime_start_predict,
                 datetime_column='date',
                 value_column='value',
                 columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                       'holydays', 'near_holydays', 'big_holydays', 'pre_holydays'),
                 learning_rate=0.02,
                 max_depth=3,
                 n_estimators=900,
                 min_child_weight=4,
                 analysis_algoritm='XGB'):

    input_df = add_columns_to_df(input_df,
                                columns_for_analysis,
                                datetime_column)

    input_df_filtred_fit =     input_df[input_df[datetime_column] < datetime_start_predict]
    input_df_filtred_predict = input_df[input_df[datetime_column] >= datetime_start_predict]

    input_df_filtred_predict = input_df_filtred_predict.sort_values([datetime_column], ascending=[True])

    if analysis_algoritm == 'XGB':
        model = xgb.XGBRegressor(learning_rate=learning_rate,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 min_child_weight=min_child_weight)
    elif analysis_algoritm == 'LGB':
        model = lgb.LGBMRegressor(objective='regression',
                                  max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  min_child_weight=min_child_weight)
    else:
        print('No this methods: {} Please input "XGB" or "LGB".'.format(analysis_algoritm))
        return 0
    model.fit(input_df_filtred_fit[list(columns_for_analysis)], input_df_filtred_fit[value_column])
    input_df_filtred_predict.loc[:, 'predict'] = model.predict(input_df_filtred_predict[list(columns_for_analysis)])
    accuracy = 1 - mean_squared_error(input_df_filtred_predict['predict'],
                                      input_df_filtred_predict[value_column]) ** 0.5/\
                   input_df_filtred_predict[value_column].mean()

    return input_df_filtred_predict[[value_column, 'predict', datetime_column]], accuracy


def predict_by_date(input_df,
                    predict_days,
                    date_start_predict,
                    date_column='date',
                    value_column='value',
                    columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                          'holydays', 'near_holydays', 'big_holydays', 'pre_holydays'),
                    learning_rate=0.02,
                    max_depth=3,
                    n_estimators=900,
                    min_child_weight=4,
                    analysis_algoritm='XGB'):

    predict_df = pd.DataFrame(data=[[None, date_start_predict + timedelta(days=i)] for i in range(predict_days)],
                              columns=['predict', date_column])

    return predict(input_df,
                   predict_df,
                   datetime_start_predict=date_start_predict,
                   datetime_column=date_column,
                   value_column=value_column,
                   columns_for_analysis=columns_for_analysis,
                   learning_rate=learning_rate,
                   max_depth=max_depth,
                   n_estimators=n_estimators,
                   min_child_weight=min_child_weight,
                   analysis_algoritm=analysis_algoritm)


def predict_by_datetime(input_df,
                        period_in_seconds,
                        predict_number_of_perionds,
                        datetime_start_predict,
                        datetime_column='datetime',
                        value_column='value',
                        columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                              'holydays', 'near_holydays', 'big_holydays', 'pre_holydays',
                                              'hour', 'minute', 'minute_of_day'),
                        learning_rate=0.02,
                        max_depth=3,
                        n_estimators=900,
                        min_child_weight=4,
                        analysis_algoritm='XGB'):

    predict_df = pd.DataFrame(data=[[None, datetime_start_predict + timedelta(seconds=period_in_seconds*i)] for i in range(predict_number_of_perionds)],
                              columns=['predict', datetime_column])



    return predict(input_df,
                   predict_df,
                   datetime_start_predict=datetime_start_predict,
                   datetime_column=datetime_column,
                   value_column=value_column,
                   columns_for_analysis=columns_for_analysis,
                   learning_rate=learning_rate,
                   max_depth=max_depth,
                   n_estimators=n_estimators,
                   min_child_weight=min_child_weight,
                   analysis_algoritm=analysis_algoritm)


def predict(input_df,
            predict_df,
            datetime_start_predict,
            datetime_column='datetime',
            value_column='value',
            columns_for_analysis=('year', 'month', 'day', 'day_of_the_year', 'weekday', 'week', 'week_of_month',
                                  'holydays', 'near_holydays', 'big_holydays', 'pre_holydays'),
            learning_rate=0.02,
            max_depth=3,
            n_estimators=900,
            min_child_weight=4,
            analysis_algoritm='XGB'):

    input_df = add_columns_to_df(input_df,
                                 columns_for_analysis,
                                 datetime_column)

    input_df_filtred_fit = input_df[input_df[datetime_column] < datetime_start_predict]

    predict_df = add_columns_to_df(predict_df,
                                   columns_for_analysis,
                                   datetime_column)

    predict_df = predict_df.sort_values([datetime_column], ascending=[True])

    if analysis_algoritm == 'XGB':
        model = xgb.XGBRegressor(learning_rate=learning_rate,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 min_child_weight=min_child_weight)
    elif analysis_algoritm == 'LGB':
        model = lgb.LGBMRegressor(objective='regression',
                                  max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators,
                                  min_child_weight=min_child_weight)
    else:
        print('No this methods: {} Please input "XGB" or "LGB".'.format(analysis_algoritm))
        return 0
    model.fit(input_df_filtred_fit[list(columns_for_analysis)], input_df_filtred_fit[value_column])
    predict_df.loc[:, 'predict'] = model.predict(predict_df[list(columns_for_analysis)])

    return predict_df[['predict', datetime_column]]

