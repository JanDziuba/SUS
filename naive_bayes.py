import pandas as pd
import math
from math import sqrt
from math import exp
from math import pi


# divide our set into a test and train sets
def my_train_test_split(df, test_size=0.25):
    if test_size >= 1 or test_size <= 0:
        raise NameError('Wrong test_size')

    # shuffle the DataFrame rows
    df = df.sample(frac=1)

    rows = df.shape[0]
    test_rows = math.ceil(rows * test_size)
    test_df = df.iloc[:test_rows, :]
    train_df = df.iloc[test_rows:, :]

    return (train_df, test_df)


def separate_by_play(train_df):
    play_df = train_df[train_df.Play.eq('Yes')]
    not_play_df = train_df[train_df.Play.eq('No')]
    return play_df, not_play_df


def separate_categorical_data(df):
    categorical = df[['Weather', 'Windy']]
    continuous = df[['Temperature', 'Humidity']]
    return categorical, continuous


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers))
    return sqrt(variance)


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def get_caterogical_p(categorical_data):
    rows = categorical_data.shape[0]
    caterogical_p = {}

    caterogical_p['Weather'] = categorical_data['Weather'].value_counts() / rows
    caterogical_p['Windy'] = categorical_data['Windy'].value_counts() / rows
    return caterogical_p


def get_means_and_stdevs(continuous_data):
    means_and_stdevs = {}
    for (columnName, columnData) in continuous_data.iteritems():
        means_and_stdevs[columnName] = (mean(columnData), stdev(columnData))

    return means_and_stdevs


def predict(row, p_play, play_caterogical_p, not_play_caterogical_p,
            play_means_and_stdevs, not_play_means_and_stdevs):
    row_caterogical, row_continuous = separate_categorical_data(row)
    play_p = p_play
    not_play_p = 1 - p_play

    for (columnName, columnData) in row_caterogical.iteritems():
        play_p *= play_caterogical_p[columnName].get(columnData, 0)
        not_play_p *= not_play_caterogical_p[columnName].get(columnData, 0)

    for (columnName, columnData) in row_continuous.iteritems():
        x = columnData
        play_mean, play_stdev = play_means_and_stdevs[columnName]
        play_p *= calculate_probability(x, play_mean, play_stdev)

        not_play_mean, not_play_stdev = not_play_means_and_stdevs[columnName]
        not_play_p *= calculate_probability(x, not_play_mean, not_play_stdev)

    if play_p > not_play_p:
        return 'Yes'
    else:
        return 'No'


def test_accuracy(test_df, p_play, play_caterogical_p, not_play_caterogical_p,
                  play_means_and_stdevs, not_play_means_and_stdevs):
    expected = test_df["Play"]
    good_pred_number = 0
    row_count = len(test_df.index)

    for row in range(test_df.shape[0]):
        prediction = predict(test_df.iloc[row], p_play, play_caterogical_p, not_play_caterogical_p,
                             play_means_and_stdevs, not_play_means_and_stdevs)
        if prediction == expected.iloc[row]:
            good_pred_number += 1

    print('accuracy: ', good_pred_number / row_count)


if __name__ == '__main__':
    weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
               'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
    temp = ['85', '80', '83', '70', '68', '65', '64', '72', '69', '75', '75', '72', '81', '87']
    humidity = ['85', '90', '86', '96', '80', '70', '65', '95', '70', '80', '70', '90', '75', '91']
    windy = [False, True, False, False, False, True, True, False, False, False, True, True, False, True]
    play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

    weather_dataset = list(zip(weather, temp, humidity, windy, play))

    # creating a pandas DataFrame
    df = pd.DataFrame(data=weather_dataset, columns=['Weather', 'Temperature', 'Humidity', 'Windy', 'Play'])
    # saving pandas DataFrame
    df.to_csv('weather.csv', index=False)

    df = pd.read_csv('weather.csv')

    train_df, test_df = my_train_test_split(df)

    play_df, not_play_df = separate_by_play(train_df)

    p_play = play_df.shape[0] / train_df.shape[0]

    play_categorical, play_continuous = separate_categorical_data(play_df)
    not_play_categorical, not_play_continuous = separate_categorical_data(not_play_df)

    play_caterogical_p = get_caterogical_p(play_categorical)
    not_play_caterogical_p = get_caterogical_p(not_play_categorical)

    play_means_and_stdevs = get_means_and_stdevs(play_continuous)
    not_play_means_and_stdevs = get_means_and_stdevs(not_play_continuous)

    test_accuracy(test_df, p_play, play_caterogical_p, not_play_caterogical_p,
                  play_means_and_stdevs, not_play_means_and_stdevs)
