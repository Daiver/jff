
template seqPointer *(v: seq[float32]): ptr float32 = cast[ptr float32](unsafeAddr(v[0]))

template seqPointer *(v: seq[float64]): ptr float64 = cast[ptr float64](unsafeAddr(v[0]))
