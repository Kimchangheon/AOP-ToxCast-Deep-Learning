"""Run the pparg ligand model."""

import numpy
import pandas
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model


def main():
    """Run the main function."""
    outfile = open(sys.argv[1] + "_predictions_" +sys.argv[2] +".txt" , 'w')
    outfile.write("Chemical Name\tPrediction\n")
    #model = load_model("../TOX21_PPARg_BLA_antagonist_ratio(Input)_model(99.32).h5")
    model = load_model(sys.argv[2])
    dataframe = pandas.read_csv(sys.argv[1], sep="\t")
    mols = []
    fps = []

    for index, row in dataframe.iterrows():
        try : 
            mol = Chem.MolFromSmiles(row['Smiles'])
        except :
            continue
            
        try : 
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        except :
            continue
       
        mols.append(mol)
        fps.append(fp)

    np_fps = []
    for fp in fps:
        arr = numpy.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)

    np_fps_array = numpy.array(np_fps)
    predictions = model.predict(np_fps_array, batch_size=5)
    i = 0
    for prediction in predictions:
        y_prediction = ''
        if(prediction < 0.50):
            y_prediction = "ligand"
        else:
            y_prediction = "not_ligand"
        outfile.write(dataframe['Chemical_Name'][i] + "\t" + y_prediction + "\t" + repr(prediction) + "\n")
        i += 1
    outfile.close()


if __name__ == "__main__":
    main()
# """Run the pparg ligand model."""

# import numpy
# import pandas
# import sys
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import confusion_matrix
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import SimilarityMaps
# from rdkit import Chem, DataStructs
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from keras.models import load_model


# def main():
#     """Run the main function."""
#     outfile = open(sys.argv[1] + "_predictions.txt", 'w')
#     outfile.write("Chemical Name\tPrediction\n")
#     model = load_model("./models/BSK_KF3CT_IL1a_up(Input)_model(99.72).h5")
#     dataframe = pandas.read_csv(sys.argv[1], sep="\t")
#     mols = []
#     fps = []

#     for index, row in dataframe.iterrows():
#         mol = Chem.MolFromSmiles(row['Smiles'])
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
#         mols.append(mol)
#         fps.append(fp)

#     np_fps = []
#     for fp in fps:
#         arr = numpy.zeros((1,))
#         DataStructs.ConvertToNumpyArray(fp, arr)
#         np_fps.append(arr)

#     np_fps_array = numpy.array(np_fps)
#     predictions = model.predict(np_fps_array, batch_size=5)
#     i = 0
#     for prediction in predictions:
#         y_prediction = ''
#         if(prediction < 0.50):
#             y_prediction = "ligand"
#         else:
#             y_prediction = "not_ligand"
#         outfile.write(dataframe['Chemical_Name'][i] + "\t" + y_prediction + "\n")
#         #outfile.write(dataframe['Chemical_Name'][i] + "\t" + repr(prediction) + "\n")
#         i += 1
#     outfile.close()


# if __name__ == "__main__":
#     main()