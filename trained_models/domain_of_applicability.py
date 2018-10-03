"""Perform domain of applicability analysis."""

import numpy
import pandas
import sys
from sklearn.decomposition import PCA as sklearnPCA
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem, DataStructs
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

dataframe = pandas.read_csv("../pubchem_data/processed/pparg_ligand_data.txt",
                            sep="\t")
mols = []
fps = []

for index, row in dataframe.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    mols.append(mol)
    fps.append(fp)

np_fps = []
for fp in fps:
    arr = numpy.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    np_fps.append(arr)

np_fps_array = numpy.array(np_fps)

sklearn_pca = sklearnPCA(n_components=2)
y_sklearn = sklearn_pca.fit_transform(np_fps_array)
pccr = pandas.DataFrame(data=y_sklearn)
pccr.columns = ['PC1', 'PC2']
# sns.jointplot(x="PC1", y="PC2", data=pccr)
# plt.show()

mols2 = []
fps2 = []
chem_names = []
inputdata = pandas.read_csv(sys.argv[1], sep="\t")
for index, row in inputdata.iterrows():
    chem_name = row['Chemical_Name']
    mol = Chem.MolFromSmiles(row['SMILES'])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    mols2.append(mol)
    fps2.append(fp)
    chem_names.append(chem_name)

np_fps2 = []
for fp in fps2:
    arr = numpy.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    np_fps2.append(arr)

np_fps_array2 = numpy.array(np_fps2)

y_sklearn2 = sklearn_pca.fit_transform(np_fps_array2)
pccr2 = pandas.DataFrame(data=y_sklearn2)
pccr2.columns = ['PC1', 'PC2']

outfile = open(sys.argv[1] + "_domain_of_applicability.txt", 'w')
outfile.write("Chemical Name\tWithin Domain of Applicability\n")


def is_p_inside_points_hull(points, p):
    """Check if a point is inside the convex hull."""
    hull = ConvexHull(points)
    new_points = numpy.append(points, p, axis=0)
    new_hull = ConvexHull(new_points)
    if list(hull.vertices) == list(new_hull.vertices):
        return "true"
    else:
        return "false"


for i in range(0, len(pccr2)):
    outfile.write(chem_names[i] + "\t" +
                  is_p_inside_points_hull(pccr, pccr2.loc[[i]]) +
                  "\n")

outfile.close()
