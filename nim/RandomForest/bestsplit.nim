import ../linalg/matrix.nim

import bestthreshold

proc findBestSplit * [Scalar, GainStruct, TargetScalar] (
            data: Matrix[Scalar],
            targetValues: Matrix[TargetScalar],
            dataIndices: seq[int],
            featIndices: seq[int],
            gainFunc: proc(gAll, gL, gR: GainStruct): Scalar,
            zeroGainValue: GainStruct
        ): (float64, Scalar, int) = 
    
    let nSamples = dataIndices.len
    var valuesLocal  = matrix.zeros[Scalar](nSamples)
    var targetsLocal = matrix.zeros[TargetScalar](nSamples)

    for i in 0..nSamples - 1:
        targetsLocal[i] = targetValues[dataIndices[i]]

    var bestGain = -1e10
    var bestFeatInd = -1
    var bestThr: Scalar = 0

    for featInd in featIndices:
        for i in 0..nSamples - 1:
            valuesLocal[i] = data[dataIndices[i], featInd]
        let (gain, thr) = findBestThreshold(
                            valuesLocal,
                            targetsLocal,
                            gainFunc, 
                            zeroGainValue)
        if gain > bestGain:
            bestGain = gain
            bestThr = thr
            bestFeatInd = featInd
        
    return (bestGain, bestThr, bestFeatInd)
