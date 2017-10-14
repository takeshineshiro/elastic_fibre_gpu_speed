//////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2\opencv.hpp>

#include <opencv2\gpu\gpu.hpp>

#include "ElastoCudaExt.h"

using namespace cv;

using namespace cv::gpu;

struct EInput;

struct EOutput;

struct CvMat;

struct ConfigParam;

namespace  elasto_cuda
{
    //////////////////////////////////////////////////////////////////////////
    // GPU-Cuda 基本要求
    const   int    g_nCudaMajorVer = 1;

    const   int    g_nCudaMinorVer = 2;

	const   int    k_nThreadDimX = 32;

	const   int    k_nThreadDimY = 32;

    const   int    FilterParamElemMaxNum = 128; // 用于滤波处理的参数的最大数量，它指定一个数据区的长度，注意，它不是指字节数

	template<typename T> PtrStepSz<T> make_PtrStepSz(void *data, int rows, int cols, int step);


	template<typename T> PtrStep<T>   make_PtrStep(void *data, int step);


	int  TotalBytes(const Size &size);


	int  TotalBytes(const PtrStepSzf &ptr);


	int  TotalBytes(const cv::Mat &mat);


	int  TotalBytes(const CvMat &mat);

    //////////////////////////////////////////////////////////////////////////
    // Elasto CUDA Implement
    // Only One Instance, it is included in Wrapper Class .
    //////////////////////////////////////////////////////////////////////////


	class CudaComputer
	{


	public:


		CudaComputer();

		~CudaComputer();


		bool  IsAvaible() { return m_fAvaible; }


		int   Init(const ConfigParam &param);


        void  InitBandpassFilterParams(float *pDatas, int num); // 初始化带通滤波的参数


        void  InitLowpassFilterParams (float *pDatas, int num); // 初始化低通滤波的参数


		void  InitMatchFilterParams   (float *pDatas, int num); // 初始化匹配增强的参数苏



        void  AllocHostMem(); // allocate Host page-locked memory


        void  FreeHostMem();


        void  AllocCudaMem(); // allocate Cuda Device memory


        void  FreeCudaMem();


        void  DoBpFilter(cv::Mat &in_out);  // 带通滤波


        void  DoLpFilter(cv::Mat &in_out);  // 低通滤波


        void  DoCalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat); // 计算应变


        void  DoCalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int win, int step, int scale); // 计算位移


        void  DoRandonSum(const cv::Mat &in, cv::Mat &out); // 拉东变换


        void  MeanFilter(const cv::Mat &in, cv::Mat &out, int step); // 平滑滤波


		void  MatchFilter(cv::Mat &image); // 匹配增强滤波


		void  Rel2AbsDisp(cv::Mat &image); // 位移叠加。相对位移转为绝对位移


	protected:

    private:


		void DoFilter_1(cv::Mat &in_out, PtrStepSzf params); // （带通、低通）滤波的实际执行体

  
	private:

        bool         m_fAvaible;

		////////////////////////////////////////////
        // following is page-locked memory in Host

        PtrStepSzf   m_ptrInFrame;         // 主机端-从超声设备取得的数据帧，算法处理的输入数据矩阵

        PtrStepSzf   m_ptrDispMat;         // 主机端-位移数据矩阵

        PtrStepSzf   m_ptrStrainMat;        // 主机端-应变数据矩阵


		////////////////////////////////////////////
        // following memory in Cuda Device 

        PtrStepSzf   m_ptrInFrame_d;       // 设备端-从超声设备取得的数据帧，算法处理的输入数据矩阵

        PtrStepSzf   m_ptrDispMat_d;       // 设备端-位移数据矩阵

        PtrStepSzf   m_ptrStrainMat_d;     // 设备端-应变数据矩阵

        PtrStepSzf   m_ptrFilterBuf_d;     // 设备端-滤波处理的中间缓冲区

        PtrStepSzf   m_ptrBpParams_d;      // 设备端-带通滤波的参数

        PtrStepSzf   m_ptrLpParams_d;      // 设备端-低通滤波的参数

        PtrStepSzf   m_ptrMatchParams_d;   // 设备端-匹配滤波的参数

        PtrStepSzf   m_ptrTemplateMat_d;   // 设备端-位移处理，模板数据

        PtrStepSzf   m_ptrObjectMat_d;     // 设备端-位移处理，对象数据

        PtrStepSz<float*>  m_ptrResultMats_d;  // 设备端-位移处理，结果数据

        PtrStepSzf   m_ptrRandonIn_d;      // 设备端-应变处理，拉东变换的输入数据

        PtrStepSzf   m_ptrRandonOut_d;     // 设备端-应变处理，拉东变化的输出数据

	};
}