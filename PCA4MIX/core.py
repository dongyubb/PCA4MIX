import pandas as pd
import numpy as np
import math
import scipy
from scipy.stats import zscore


class PCA4MIX (object):

    def splitdata(self, input_dataframe):
        """ Split dataframe to numerical and categorical dataframe

        Args:
            input_dataframe: pandas dataframe to be split
        Returns:
            data_numeric: pandas dataframe with numerical variables
            data_category: pandas dataframe with categorical variables
        """
        data_category = input_dataframe.select_dtypes(include="object")
        data_numeric = input_dataframe.select_dtypes(exclude="object")
        data_category = data_category.astype("category")
        return data_numeric, data_category

    def dataframe_recod(self, data_numeric, data_category):
        """Reconstruct dataframe to be used in pcamix.
        Note that some returned values are for the purpose of inspection and will not be used in other function
        Args:
            data_numeric: pandas dataframe with numerical varialbes
            data_category: pandas dataframe with categorical variables
        Returns:
            X: pandas dataframe numeric and category concatenate
            Y: pandas dataframe numeric and category indicator concatenate
            Z: pandas dataframe standardized numeric and category indicator concatenate
            W:
            , n, p, p1, p2, g, s, indexj, indicator_matrix, category_cod, data_numeric, data_category, category_num






        """
        #num of rows and columns, std and mean of columns, zscores of numeric matrix
        n1 = len(data_numeric.index)
        p1 = len(data_numeric.columns)
        reduc = math.sqrt(((len(data_numeric.index) - 1) / len(data_numeric.index)))
        numeric_std = data_numeric.std(axis=0) * reduc
        numeric_mean = data_numeric.mean(axis=0)
        numeric_zscores = data_numeric.apply(zscore)

        #num of rows and columns, std and mean of columns, zscores of categorical matrix
        indicator_matrix = pd.get_dummies(data_category)
        n2 = len(data_category.index)
        p2 = len(data_category.columns)
        category_mean = indicator_matrix.mean(axis=0)
        category_sum = indicator_matrix.sum(axis=0)
        category_std = (category_sum / n2).apply(np.sqrt)
        category_cod = indicator_matrix / category_std
        category_zscore = category_cod - category_cod.mean(axis=0)
        category_cent = indicator_matrix - category_sum / n2
        category_num = data_category.nunique()
        indexj2 = []
        for i in range(0, p2):
            for j in range(0, category_num[i]):
                indexj2.append(i + 1)
        indexj = list(range(1, len(data_numeric.columns) + 1))
        indexj2 = [x + len(data_numeric.columns) for x in indexj2]
        indexj = indexj + indexj2
        if n1 == n2:
            n = n1
            p = p1+p2
            Z = pd.concat([numeric_zscores,category_zscore], axis=1)
            Y = pd.concat([data_numeric,indicator_matrix], axis=1)
            W = pd.concat([numeric_zscores,category_cent],axis=1)
            g = pd.concat([numeric_mean, category_mean], axis=0)
            s = pd.concat([numeric_std, category_std], axis=0)
            X = pd.concat([data_numeric,data_category],axis=1)
        else :
            raise Exception("Unequal rows of numerical and categorical dataframe")
        return X, Y, Z, W, n, p, p1, p2, g, s, indexj, indicator_matrix, category_cod, data_numeric, data_category, category_num

    #Generalized sigular value decomposition
    def gsvd(self, W, N, Mcol):
        ncp = len(W.columns) # placeholder for chose number of PCs
        #build matrix for GSVD
        roweight = N / N.sum(axis=1).values[0]
        W = W.mul(Mcol.apply(np.sqrt).values, axis=1)
        W = W.mul(roweight.transpose().apply(np.sqrt).values, axis=0)
        #if columns > index, the matrix will be transposed to perform GSVD (in developing)
        if len(W.columns) < len(W.index):
            U, S, Vh = scipy.linalg.svd(W, full_matrices=False, lapack_driver="gesvd")#gesvd was used to match the Matlab svd routine
            #scipy gives the transposed V
            V = Vh.T
            multicoef = V.sum(axis=0)
            multicoef = np.sign(multicoef)
            multicoef[multicoef == 0] = 1
            U = U * multicoef
            V = V * multicoef
            roweight = np.array(roweight)
            roweight = np.sqrt(roweight)
            U = U / roweight.transpose()
            colweight = np.array(Mcol)
            colweight = np.sqrt(colweight)
            V = V / colweight.transpose()
            colnumV = np.where(S < 1e-15)
            U[:, colnumV[0]] = U[:, colnumV[0]] * S[colnumV[0]]
            V[:, colnumV[0]] = V[:, colnumV[0]] * S[colnumV[0]]
        return U, S, V

    def pcamix(self, data_numeric, data_category):
        X, Y, Z, W, n, p, p1, p2, g, s, indexj, indicator_matrix, category_cod, data_numeric, data_category, category_num = self.dataframe_recod(data_numeric,data_category)
        #construct weight dataframe
        columnames = data_numeric.columns.values.tolist() + indicator_matrix.columns.values.tolist()
        m = len(W.columns) - p1
        N = pd.DataFrame([1 / n for x in range(n)]).T
        M1 = pd.DataFrame([1 for x in range(p1)]).T
        P1 = M1
        M2 = n / indicator_matrix.sum(axis=0)
        M2 = M2.to_frame().T
        P2 = pd.DataFrame([1 for x in range(len(indicator_matrix.columns))]).T
        M = pd.concat([M1, M2], axis=1)
        M.columns = columnames
        P = pd.concat([P1, P2], axis=1)
        P.columns = columnames
        Mcol = P.multiply(M, axis=0)
        U, S, V = self.gsvd(W, N, Mcol)# call svd
        # number of PCs retained
        Wrank = np.linalg.matrix_rank(W)

        # eigenvalue proportion and cumulative
        S = S[0:Wrank]
        eigDF = np.zeros(shape=(Wrank, 3))
        eigDF = pd.DataFrame(eigDF, columns=["Eigenvalue", "Proportion", "Cumulative"],
                             index=["PC%s" % i for i in range(1, Wrank + 1)])
        eigDF["Eigenvalue"] = S ** 2
        eigDF["Proportion"] = 100 * (eigDF["Eigenvalue"] / eigDF["Eigenvalue"].sum())
        eigDF["Cumulative"] = [eigDF["Proportion"][0:i].sum() for i in range(1, Wrank + 1)]
        # calculate PC scores
        U = U[:, 0:Wrank]  # standardized scores
        U = pd.DataFrame(U, columns=["PC%s" % i for i in range(1, Wrank + 1)])
        F = U * S  # scores
        # loadings and contributions
        V = V[:, 0:Wrank]
        V = pd.DataFrame(V, columns=["PC%s" % i for i in range(1, Wrank + 1)], index=[columnames])
        # numerical
        if p1 > 0:
            V1 = V.loc[columnames[0:p1], :]
            A1 = V1 * S
            NumericContri = A1 ** 2
            NumericContriPct = NumericContri / (S ** 2)
        # categorical
        if p2 > 0:
            V2 = V.loc[columnames[p1:p1 + m], :]
            A2 = (V2 * S).mul(M2.T.values, axis=0)
            CategoryContri = (A2 ** 2).mul((indicator_matrix.sum(axis=0) / n).values, axis=0)
            CategoryContriPct = CategoryContri / (S ** 2)

        ReContriPct = pd.concat([NumericContriPct, CategoryContriPct])
        Sqloadings = pd.concat([NumericContri, CategoryContri])

        # coefficient of PCs
        Coef = V.mul(Mcol.T.values, axis=0)
        Coef.loc[columnames[0:p1], :] = Coef.loc[columnames[0:p1], :].divide(s.loc[columnames[0:p1]].values, axis=0)
        Coef.index = Coef.index.map("".join)
        CoefConstMat = Coef.mul(g.T.values, axis=0)
        CoefConst = -CoefConstMat.sum(axis=0)
        CoefConst = pd.DataFrame(CoefConst, columns=["Constant"]).T
        CoefMatrix = pd.concat([CoefConst, Coef])

        return eigDF, U, F, Sqloadings, ReContriPct, CoefMatrix
