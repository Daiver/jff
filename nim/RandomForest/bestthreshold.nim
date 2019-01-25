import ../linalg/matrix.nim
import common

proc findBestThreshold * [Scalar, GainStruct, TargetScalar] (
            values: Matrix[Scalar],
            targetValues: Matrix[TargetScalar],
            gainFunc: proc(gAll, gL, gR: GainStruct): Scalar,
            zeroGainValue: GainStruct
        ): (float64, Scalar) = 
    let nSamples = values.rows()
    let indices = values.argsort
    var gAll: GainStruct = zeroGainValue
    for i in 0..nSamples - 1:
        gAll = gAll + targetValues[i]
    var gL: GainStruct = zeroGainValue
    var gR: GainStruct = gAll

    var bestGain = -1e10
    var threshold = 0.0

    for indInd in 0 .. nSamples - 1:
        let sampleInd = indices[indInd]
        let curVal  = values[sampleInd]
        let targetVal = targetValues[sampleInd]
        gR = gR - targetVal
        gL = gL + targetVal
        if indInd < nSamples - 1 :
            let nextInd = indices[indInd + 1]
            let nextVal = values[nextInd]
            let diff = nextVal - curVal
            if diff.abs < 0.00001:
                continue
        let curGain = gainFunc(gAll, gL, gR)
        if curGain > bestGain:
            bestGain = curGain
            var thr = curVal
            if  indInd < nSamples - 1:
                let nextVal = values[indices[indInd + 1]]
                thr = (thr + nextVal)/2.0
            threshold = thr

    return (bestGain, threshold)
