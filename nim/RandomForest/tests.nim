import unittest

import sequtils

import ../linalg/linalg.nim
import ../linalg/matrix.nim

import bestthreshold
import bestsplit
import common
import gainstruct
import tree

suite "GainStruct tests":

    test "create test 01":
        let gs = gainstruct.zeroGS(5)
        let ans = @[0.0, 0, 0, 0, 0]
        check(gs.dict == ans)

    test "add/sub test 01":
        let gs1 = GainStruct(dict: @[1.0, 2, 3])
        let gs2 = GainStruct(dict: @[5.0, 2, 4])
        let gs3 = gs1 + gs2
        let gs4 = gs1 - gs2
        let ans1 = @[6.0, 4, 7]
        let ans2 = @[-4.0, 0, -1]
        check gs3.dict == ans1
        check gs4.dict == ans2
        
    test "gini imp test 01":
        let gs = GainStruct(dict: @[6.0, 4])
        let res = gs.giniImp
        let ans = 1.0 - 0.6*0.6 - 0.4*0.4
        check(fuzzyComp(res, ans))

suite "Common tests":

    test "sort test 01":
        let mat = matrix.fromValues([4, 1, 3, 5, 6, -7])
        let indices = argsort mat
        let ans = @[5, 1, 2, 0, 3, 4]
        check(ans == indices)

suite "Decision tree splits tests":
    test "best thr test 01":
        let values = fromValues(@[1.0, 2, 3, 4, 5])
        let labels = @[1, 1, 1, 0, 0]
        let res = findBestThreshold(values, fromValues labels, gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, 3.5))

    test "best thr test 02":
        let values = fromValues(@[4.0, 0, 3, 5, 2])
        let labels = @[0, 1, 1, 0, 1]
        let res = findBestThreshold(values, fromValues labels, gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, 3.5))

    test "best thr test 03":
        let values = fromValues(@[4.0, 0, 3, 5, 2])
        let labels = @[1, 1, 1, 0, 1]
        let res = findBestThreshold(values, fromValues labels, gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, 4.5))

    test "best thr test 04":
        let values = fromValues(@[4.0, 0, 3, -1, 2])
        let labels = @[1, 1, 1, 0, 1]
        let res = findBestThreshold(values, fromValues labels, gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, -0.5))

    test "best split test 01":
        let data = fromValues(@[
                    4.0, 1,
                    0,   1,
                    3,   1,
                   -1,   1,
                    2,   1]).reshape(5, 2)
        #echo($data)
        let labels = fromValues(@[1, 0, 1, 0, 0])
        let res = findBestSplit(data, labels, 
                                toSeq((0..labels.rows() - 1)),
                                @[0, 1],
                                gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, 2.5))
        let feat = res[2]
        check(feat == 0)

    test "best split test 02":
        let data = fromValues(@[
                    4.0, 1,
                    6,   2,
                    3,   3,
                   -1,   4,
                    2,   5]).reshape(5, 2)
        #echo($data)
        let labels = fromValues(@[0, 1, 1, 1, 1])
        let res = findBestSplit(data, labels, 
                                toSeq((0..labels.rows() - 1)),
                                @[0, 1],
                                gainstruct.gainFunc, zeroGS(2))
        let thr = res[1]
        check(fuzzyComp(thr, 1.5))
        let feat = res[2]
        check(feat == 1)

