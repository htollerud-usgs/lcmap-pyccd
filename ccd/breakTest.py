
# Functions used to test for breaks

# Order of dimensions are expected to always be: bands, time, coefficients

import numpy as np


def breakTestIncludingModelError(compareObservationResiduals,regressorsForCompareObservations,
        msrOfCurrentModels,nObservationsInModel,cutoffLookupTable,pValueForBreakTest,inverseMatrixXTX,nCompareForP):
    """Test for model breaks; include observation error and model error.
    Also allows for a variable number of comparison observations.
    """

    nBands,nCompareObservations = compareObservationResiduals.shape
    nCoefficients = regressorsForCompareObservations.shape[1]
    magnitudes = np.zeros((nCompareObservations))

#    inverseMatrixXTX = np.linalg.inv(matrixXTX)

    for i in range(nCompareObservations):

        xAtThisTime = np.copy(regressorsForCompareObservations[i,:])
        # This is the additional factor to account for model error
        modelErrorAdjustment = np.matmul(np.matmul(np.transpose(xAtThisTime),inverseMatrixXTX),xAtThisTime)

        for band in range(nBands):
            magnitudes[i] += np.power(compareObservationResiduals[band,i],2)/(msrOfCurrentModels[band]*(1+modelErrorAdjustment))

    # The input p-value is for the entire test, including all nCompareObservations points. The p-value for the minimum doesn't
    #    need to be as stringent, since we know that all the other observations have larger residuals. If they are independentThe chance that all of
    #    the compare observations have p<individualPValue is individualPValue^nCompareObservations
    individualPValue = np.power(pValueForBreakTest,1/nCompareForP)

    nDegreesOfFreedom = max(nObservationsInModel-nCoefficients,1)
    cutoff = cutoffLookupTable[501-int(individualPValue*1000),min(nDegreesOfFreedom,cutoffLookupTable.shape[1])-1]

    if nCompareObservations == nCompareForP:
        compareValue = min(magnitudes)
    else:
        sort = np.argsort(magnitudes)
        secondIndex = np.floor_divide(nCompareObservations,nCompareForP)
        mod = np.mod(nCompareObservations,nCompareForP)
        compareValue = (magnitudes[sort[secondIndex-1]]*(nCompareForP-mod) + magnitudes[sort[secondIndex]]*mod)/nCompareForP

    if compareValue > cutoff:
        return True, magnitudes
    else:
        return False, magnitudes

