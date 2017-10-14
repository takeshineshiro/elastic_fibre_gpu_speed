//////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef    _FILEULITY   

#define  FileUtil_API   __declspec(dllexport)
#else

#define  FileUtil_API   __declspec(dllimport)

#endif


#include "opencv/cv.h"

template<typename T> void ReadBinFile(const char *filepath, T *pBuffer, int nElems);

FileUtil_API  int  ReadRFData(const char *file_path, float *rf, int rows, int cols);
int  ReadRFDataT(const char *file_path, short *rf, int rows, int cols);
int  ReadRFDataB(const char *file_path, short *rf, int rows, int cols);
int  ReadMatFile(const char *file_path, float *rf, int rows, int cols);

void  SaveDataFile(const char *filename, CvMat *pmat);
void  SaveDataFile(const char *filename, cv::Mat *pmat);
void  SaveDataFile(const char *filename, cv::Mat &mat);

void  MakeBmpAndShow(const char *filename, const CvMat *pmat);
