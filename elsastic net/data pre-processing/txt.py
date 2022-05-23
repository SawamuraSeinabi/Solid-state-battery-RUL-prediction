import pickle

data = pickle.load(open(r'.\Dataset\test_data_multi.pkl', 'rb'))

with open("test2.txt", "w") as fw:
    for key in data:
        current = data[key]
        result = f"{current['var']},{current['minimum']},{current['skewness']},{current['kurtosis']},{current['cycle2']},{current['differ']},{current['cycle_life'][0][0]}\n"
        fw.write(result)


