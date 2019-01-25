/*#define NPY_NO_DEPRECATED_API */
/*#define NPY_1_7_API_VERSION*/
#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>


/*static PyObject *fitFern(PyObject *dummy, PyObject *args)*/
/*{*/
    /*PyArrayObject *arrX     = NULL;*/
    /*PyArrayObject *arrR     = NULL;*/
    /*PyArrayObject *arrAlpha = NULL;*/
    /*PyArrayObject *arrBeta  = NULL;*/

    /*int width, height;*/
    /*if (!PyArg_ParseTuple(args, "O!O!O!O!ii", */
                /*&PyArray_Type, &arrX, */
                /*&PyArray_Type, &arrR,*/
                /*&PyArray_Type, &arrAlpha,*/
                /*&PyArray_Type, &arrBeta,*/
                /*&width,*/
                /*&height))*/
        /*return NULL;*/

    /*int dataLength = arrX->dimensions[0];*/
    /*npy_intp outDims[] = {height, width};*/
    /*PyArrayObject *arrResU = PyArray_SimpleNew(2, &outDims, PyArray_FLOAT32);*/
    /*PyArrayObject *arrResV = PyArray_SimpleNew(2, &outDims, PyArray_FLOAT32);*/

    /*int i, j, k;*/

    /*for(i = 0; i < height; ++i)*/
        /*for(j = 0; j < width; ++j){*/
            /*float valU = 0;*/
            /*float valV = 0;*/
            /*for(k = 0; k < dataLength; ++k){*/
                /*float x = *(float*)(arrX->data + */
                        /*k * arrX->strides[0] + 0 * arrX->strides[1]);*/
                /*float y = *(float*)(arrX->data + */
                        /*k * arrX->strides[0] + 1 * arrX->strides[1]);*/

                /*float alpha = *(float*)(arrAlpha->data + k * arrAlpha->strides[0]);*/
                /*float beta  = *(float*)(arrBeta->data  + k * arrBeta->strides[0]);*/
                /*float r     = *(float*)(arrR->data     + k * arrR->strides[0]);*/

                /*valU += alpha * exp(-(pow(i - x, 2.0) + pow(j - y, 2.0))/(r));*/
                /*valV += beta  * exp(-(pow(i - x, 2.0) + pow(j - y, 2.0))/(r));*/

                /*//valU += alpha * sqrt(pow(i - x, 2.0) + pow(j - y, 2.0) + pow(r, 2.0));*/
                /*//valV += beta  * sqrt(pow(i - x, 2.0) + pow(j - y, 2.0) + pow(r, 2.0));*/
            /*}*/
            /*valU += *(float*)(arrAlpha->data + (dataLength    )*arrAlpha->strides[0])*i;*/
            /*valU += *(float*)(arrAlpha->data + (dataLength + 1)*arrAlpha->strides[0])*j;*/
            /*valU += *(float*)(arrAlpha->data + (dataLength + 2)*arrAlpha->strides[0]);*/

            /*valV += *(float*)(arrBeta->data + (dataLength    )*arrBeta->strides[0])*i;*/
            /*valV += *(float*)(arrBeta->data + (dataLength + 1)*arrBeta->strides[0])*j;*/
            /*valV += *(float*)(arrBeta->data + (dataLength + 2)*arrBeta->strides[0]);*/

            /*//valU += *(float)*/
            /**(float*)(arrResU->data + i * arrResU->strides[0] + j * arrResU->strides[1]) = valU;*/
            /**(float*)(arrResV->data + i * arrResV->strides[0] + j * arrResV->strides[1]) = valV;*/
        /*}*/
    
    /*PyObject* ret = Py_BuildValue("OO", arrResV, arrResU);*/
    /*Py_DECREF(arrResU);*/
    /*Py_DECREF(arrResV);*/
    /*return ret;*/
/*}*/

inline int getIndexFromExternalFID(char *arrData, int dataStride, PyArrayObject *arrFInds)
{
    int index = 0;
    int i;
    int sizeOfFeatureIndices = arrFInds->dimensions[0];
    for(i = 0; i < sizeOfFeatureIndices; ++i){
        index <<= 1;
        int fId = *(int *)(arrFInds->data + i * arrFInds->strides[0]);
        float item = *(float *)(arrData + fId * dataStride);
        if(item > 0)
            ++index;
    }

    return index;
}

static PyObject *fitFernSimple(PyObject *dummy, PyObject *args)
{
    PyArrayObject *arrData   = NULL;
    PyArrayObject *arrValues = NULL;
    PyArrayObject *arrFInds  = NULL;

    int depthOfFern;
    if (!PyArg_ParseTuple(args, "O!O!O!i", 
                &PyArray_Type, &arrData, 
                &PyArray_Type, &arrValues,
                &PyArray_Type, &arrFInds,
                &depthOfFern))
        return NULL;

    int dataLength = arrData->dimensions[0];

    int sizeOfBins = pow(2, depthOfFern);
    npy_intp outDims[] = {sizeOfBins};
    PyArrayObject *arrBins = (PyArrayObject *)PyArray_SimpleNew(1, &outDims, PyArray_FLOAT32);

    int i;

    int *counts = malloc(sizeof(int) * sizeOfBins);
    for(i = 0; i < sizeOfBins; ++i){
        counts[i] = 0;
        *(float *)(arrBins->data + i * arrBins->strides[0]) = 0.0;
    }

    for(i = 0; i < dataLength; ++i){
        int index = getIndexFromExternalFID(
                arrData->data + i*arrData->strides[0], 
                arrData->strides[1], arrFInds);
        *(float *)(arrBins->data + index * arrBins->strides[0]) += 
            *(float *)(arrValues->data + i * arrValues->strides[0]);
        counts[index]++;
    }

    const float beta = 0.01;
    const float eps  = 0.0001;
    for(i = 0; i < sizeOfBins; ++i){
        float v1 = counts[i] + beta*dataLength;
        *(float *)(arrBins->data + i * arrBins->strides[0]) /= (v1 > eps) ? v1 : eps;
    }

    free(counts);
    return (PyObject *)arrBins;
}

static PyObject *predictFernSimple(PyObject *dummy, PyObject *args)
{
    PyArrayObject *arrData  = NULL;
    PyArrayObject *arrBins  = NULL;
    PyArrayObject *arrFInds = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!i", 
                &PyArray_Type, &arrData, 
                &PyArray_Type, &arrBins,
                &PyArray_Type, &arrFInds))
        return NULL;

    int dataLength = arrData->dimensions[0];

    npy_intp outDims[] = {dataLength};

    PyArrayObject *arrRes = (PyArrayObject *)PyArray_SimpleNew(1, &outDims, PyArray_FLOAT32);

    int i;
    for(i = 0; i < dataLength; ++i){
        int index = getIndexFromExternalFID(
                arrData->data + i*arrData->strides[0], 
                arrData->strides[1], arrFInds);
        float ans = *(float *)(arrBins->data + index * arrBins->strides[0]);
        *(float *)(arrRes->data + i * arrRes->strides[0]);
    }

    return (PyObject *)arrRes;
}



static struct PyMethodDef methods[] = {
    {"fitFernSimple", fitFernSimple, METH_VARARGS, "descript of example"},
    {"predictFernSimple", predictFernSimple, METH_VARARGS, "descript of example"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initFernBuiltins (void)
{
    (void)Py_InitModule("FernBuiltins", methods);
    import_array();
}

