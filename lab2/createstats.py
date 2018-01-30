import matplotlib
matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def boostrap(statistic_func, iterations, data):
	samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])
	#print samples.shape
	data_mean = data.mean()
	vals = []
	for sample in samples:
		sta = statistic_func(sample)
		#print sta
		vals.append(sta)
	b = np.array(vals)
	#print b
	lower, upper = np.percentile(b, [2.5, 97.5])
	return data_mean,lower, upper


# def permutation(statistic, error):


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')
	print((df.columns))
	# df1 = df.loc[:, 1]
	# df1['Number'] = df1.index
	# df1.loc[:, ] = range(1, len(df.index))
	# sns_plot = sns.lmplot(df1.columns[0], df1.columns[1], data=df1, fit_reg=False)

	df1 = df.copy()
	df1.drop(df1.columns[[1]], axis=1, inplace=True)
	df1['Number'] = df1.index

	sns_plot = sns.lmplot(df1.columns[1], df1.columns[0], data=df1, fit_reg=False)

	sns_plot.savefig("scaterplot.png", bbox_inches='tight')
	sns_plot.savefig("scaterplot.pdf", bbox_inches='tight')

	plt.clf()

	df2 = df.copy()
	df2.drop(df2.columns[[0]], axis=1, inplace=True)
	df2['Number'] = df2.index

	sns_plot = sns.lmplot(df2.columns[1], df2.columns[0], data=df2, fit_reg=False)

	sns_plot.savefig("scaterplot2.png", bbox_inches='tight')
	sns_plot.savefig("scaterplot2.pdf", bbox_inches='tight')

	data = df1.values[:,0]

	print((("Mean: %f") % (np.mean(data))))
	print((("Median: %f") % (np.median(data))))
	print((("Var: %f") % (np.var(data))))
	print((("std: %f") % (np.std(data))))
	print((("MAD: %f") % (mad(data))))

	plt.clf()

	sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()

	axes = plt.gca()
	axes.set_xlabel('MPG value old fleet')
	axes.set_ylabel('Count')

	sns_plot2.savefig("histogram.png", bbox_inches='tight')
	sns_plot2.savefig("histogram.pdf", bbox_inches='tight')

	plt.clf()

	data = df2.values[:,0]
	data = data[~np.isnan(data)]

	print((("Mean: %f") % (np.mean(data))))
	print((("Median: %f") % (np.median(data))))
	print((("Var: %f") % (np.var(data))))
	print((("std: %f") % (np.std(data))))
	print((("MAD: %f") % (mad(data))))

	sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()

	axes = plt.gca()
	axes.set_xlabel('MPG value new fleet')
	axes.set_ylabel('Count')

	sns_plot2.savefig("histogram2.png", bbox_inches='tight')
	sns_plot2.savefig("histogram2.pdf", bbox_inches='tight')


	plt.clf()

#-----------------------------



    #
	# df = pd.read_csv('./vehicles.csv')
	# print df.columns

	data = df.values.T[0]
	boots = []
	for i in range(100, 10000, 100):
		boot = boostrap(np.std, i, data)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0, )
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence_current.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence_current.pdf", bbox_inches='tight')

	plt.clf()

	data = df.values.T[1]
	data = data[~np.isnan(data)]
	boots = []
	for i in range(100, 10000, 100):
		boot = boostrap(np.std, i, data)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0, )
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence_new.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence_new.pdf", bbox_inches='tight')