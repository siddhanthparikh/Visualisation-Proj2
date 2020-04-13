from flask import Flask, render_template, jsonify, request
import json 
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

dataset = pd.read_csv('myFile.csv')
dataset = dataset.drop(['Name'], axis=1)
dataset = dataset.iloc[:500,:]
minmax  = MinMaxScaler()
cols = dataset.columns
dataset = pd.DataFrame(minmax.fit_transform(dataset), columns = cols)

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
		rnd_sample = random.sample(index_of_labels, length_of_labels)
		stratified_rows.append(data.loc[rnd_sample])


	stratified_sample = pd.concat(stratified_rows)
	del stratified_sample['titles']
	return stratified_sample

@app.route('/screeplot_org', methods = ['GET', 'POST'])
def screeplot_org():
	exp_var = list(perform_pca(dataset))
	return jsonify({"key":exp_var})

@app.route('/screeplot_strat', methods = ['GET', 'POST'])
def screeplot_strat():
	sample = stratified_sampling(dataset, 3, 0.25)
	exp_var = list(perform_pca(sample))
	return jsonify({"key":exp_var})

@app.route('/screeplot_rnd', methods = ['GET', 'POST'])
def screeplot_rnd():
	sample = dataset.sample(frac = 0.25, replace = True)
	exp_var = list(perform_pca(sample))
	return jsonify({"key":exp_var})

def perform_pca(data):
	pca1 = 	PCA()
	data = pca1.fit_transform(data)
	exp_var = pca1.explained_variance_
	return exp_var

@app.route('/pca_org', methods = ['GET', 'POST'])
def pca_org():
	# sample = stratified_sampling(dataset, 3, 0.25)
	pca1 = PCA()
	pca_data = pca1.fit_transform(dataset)
	return jsonify({"key":pca_data})

@app.route('/pca_rnd', methods = ['GET', 'POST'])
def pca_rnd():
	sample = random.sample(dataset, 0.25)
	pca1 = PCA()
	pca_data = pca1.fit_transform(sample)
	return jsonify({"key":pca_data})

@app.route('/pca_strat', methods = ['GET', 'POST'])
def pca_strat():
	sample = stratified_sampling(dataset, 3, 0.25)
	pca1 = PCA()
	pca_data = pca1.fit_transform(sample)
	return jsonify({"key":pca_data})



@app.route("/")
def d3():
    return render_template('index.html')

@app.route("/ran", methods = ['GET', 'POST'])
def ran():
	df = dataset[["sepal width", "sepal length", "petal width"]]
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})	


@app.route("/scatterplot", methods = ['GET', 'POST'])
def scatterplot():

	pca1 = 	PCA(n_components = 2)
	
	data = pca1.fit_transform(dataset)
	exp_var_ratio = pca1.explained_variance_ratio_
	df = pd.DataFrame.from_records(data, columns = ["axis1", "axis2"])
	# df = dataset[["sepal width", "sepal length"]]
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/screeplot", methods = ['GET', 'POST'])
def screeplot():
	# df = dataset[["sepal width", "sepal length", "petal width"]]

	# df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	# # df = df.to_json(orient = 'records')
	# # df = json.loads(df)
	# data_list = df.values.tolist()
	# return jsonify({"key":df})
	l = [1,2,3,4,5]
	return jsonify({"key":l})

@app.route("/mds_euc_org", methods = ['GET', 'POST'])
def mds_euc_org():
	dis_mat = pairwise_distances(dataset, metric= 'euclidean')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/mds_euc_rnd", methods = ['GET', 'POST'])
def mds_euc_rnd():
	dis_mat = pairwise_distances(dataset.sample(frac = 0.25), metric= 'euclidean')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	print (df)
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/mds_euc_strat", methods = ['GET', 'POST'])
def mds_euc_strat():
	dis_mat = pairwise_distances(stratified_sampling(dataset, 3 ,0.25), metric= 'euclidean')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/mds_cor_org", methods = ['GET', 'POST'])
def mds_cor_org():
	dis_mat = pairwise_distances(dataset, metric= 'correlation')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/mds_cor_rnd", methods = ['GET', 'POST'])
def mds_cor_rnd():
	dis_mat = pairwise_distances(dataset.sample(frac = 0.25), metric= 'correlation')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})

@app.route("/mds_cor_strat", methods = ['GET', 'POST'])
def mds_cor_strat():
	dis_mat = pairwise_distances(stratified_sampling(dataset, 3 ,0.25), metric= 'correlation')
	mds = MDS(n_components=2, dissimilarity='precomputed')
	df = mds.fit_transform(dis_mat)
	df = pd.DataFrame.from_records(df, columns = ["axis1", "axis2"])
	df = df.to_json(orient = 'records')
	df = json.loads(df)
	return jsonify({"key":df})
	
@app.route("/scatterplot_high_loading", methods = ['GET', 'POST'])
def scatterplot_high_loading():
	pca1 = 	PCA(n_components = 11)
	loading_sum = []
	data = pca1.fit_transform(dataset)
	# exp_var_ratio = pca1.explained_variance_ratio_
	loadings = pca1.components_.T * np.sqrt(pca1.explained_variance_)
	for x in loadings:
		x= [i**2 for i in x]
		loading_sum.append(sum(x))
	print(loading_sum)

	s = sorted(range(len(loading_sum)), key=lambda i: loading_sum[i], reverse=True)[:3]

	print(s)
	# df = pd.DataFrame.from_records(data, columns = ["axis1", "axis2"])
	# df = dataset[["sepal width", "sepal length"]]
	df = dataset.iloc[:, s].values
	col_list = list(dataset.columns.values)
	# print (df)
	# print (col_list)
	result = []
	print (df.tolist())
	df = df.tolist()
	for x in df:
		print (x)
		g = {}
		g[col_list[s[0]]] = x[0]
		g[col_list[s[1]]] = x[1]
		g[col_list[s[2]]] = x[2]
		result.append(g)
	# df = df.to_json(orient = 'records')
	# df = json.loads(df)
	return jsonify({"key": result})

if __name__ == "__main__":
    app.run("localhost", 8000, debug = True)