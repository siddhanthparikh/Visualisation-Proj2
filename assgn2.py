import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def find_opt_k(data, max_k):
	wcss = []
	for i in range(1, max_k + 1):
		kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		kmeans.fit(data)
		wcss.append(kmeans.inertia_)
	plt.plot(range(1, max_k + 1), wcss)
	plt.xlabel('no. of  clusters')
	plt.show()

def stratified_sampling(data, no_clusters, frac):
	kmeans1 = KMeans(n_clusters = no_clusters, random_state = 0)
	result = kmeans1.fit(data)
	data['titles'] = kmeans1.labels_
	stratified_rows = []
	for i in range(no_clusters):
		length_of_labels = (int)(len(data[data['titles'] == i])*frac)
		index_of_labels = list(data[data['titles'] == i].index)
		rnd_sample = rnd.sample(index_of_labels, length_of_labels)
		stratified_rows.append(data.loc[rnd_sample])

	# print("Size of Each Cluster in Adaptive Sampling")
	# for i in range(no_clusters):
	# 	print('Size of Cluster '+str(i)+ ' is ' +str(len(stratified_rows[i])))

	stratified_sample = pd.concat(stratified_rows)
	del stratified_sample['titles']
	return stratified_sample

def perform_pca(data):
	pca1 = 	PCA(n_components = 2)

	data = pca1.fit_transform(data)
	exp_var_ratio = pca1.explained_variance_ratio_
	return data

def create_csv(rndsample, stratsample, name_of_file):
	rndsample['type'] = pd.Series('1', index=rndsample.index)
	stratsample['type'] = pd.Series('2', index=stratsample.index)
	if len(rndsample.columns) == 3:
	    rndsample.columns = ['x', 'y', 'type']
	    stratsample.columns = ['x', 'y', 'type']

	sample = pd.concat([rndsample, stratsample])
	sample.to_csv(file_name, sep=',', index=False)

def main():
	dataset = pd.read_csv('myFile.csv')
	random_sample = dataset.sample(frac = 0.25)
	# random_sample.to_csv("randomsample.csv", sep = ',')

	#find optimal value of K using elbow method
	find_opt_k(dataset, 10)

	optimal_k = 3

	stratified_samples = stratified_sampling(dataset, optimal_k, 0.25)
	# stratified_samples.to_csv('stratifiedsample.csv', sep = ',')
	# print(s)

	#PCA
	pca_random = perform_pca(random_sample)
	pca_random = pd.DataFrame(pca_random)

	pca_stratified = perform_pca(stratified_samples)
	pca_stratified = pd.DataFrame(pca_stratified)

	create_csv(pca_random, pca_stratified, 'pca_output')

main()