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
    // GPU-Cuda ����Ҫ��
    const   int    g_nCudaMajorVer = 1;

    const   int    g_nCudaMinorVer = 2;

	const   int    k_nThreadDimX = 32;

	const   int    k_nThreadDimY = 32;

    const   int    FilterParamElemMaxNum = 128; // �����˲�����Ĳ����������������ָ��һ���������ĳ��ȣ�ע�⣬������ָ�ֽ���

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


        void  InitBandpassFilterParams(float *pDatas, int num); // ��ʼ����ͨ�˲��Ĳ���


        void  InitLowpassFilterParams (float *pDatas, int num); // ��ʼ����ͨ�˲��Ĳ���


		void  InitMatchFilterParams   (float *pDatas, int num); // ��ʼ��ƥ����ǿ�Ĳ�����



        void  AllocHostMem(); // allocate Host page-locked memory


        void  FreeHostMem();


        void  AllocCudaMem(); // allocate Cuda Device memory


        void  FreeCudaMem();


        void  DoBpFilter(cv::Mat &in_out);  // ��ͨ�˲�


        void  DoLpFilter(cv::Mat &in_out);  // ��ͨ�˲�


        void  DoCalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat); // ����Ӧ��


        void  DoCalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int win, int step, int scale); // ����λ��


        void  DoRandonSum(const cv::Mat &in, cv::Mat &out); // �����任


        void  MeanFilter(const cv::Mat &in, cv::Mat &out, int step); // ƽ���˲�


		void  MatchFilter(cv::Mat &image); // ƥ����ǿ�˲�


		void  Rel2AbsDisp(cv::Mat &image); // λ�Ƶ��ӡ����λ��תΪ����λ��


	protected:

    private:


		void DoFilter_1(cv::Mat &in_out, PtrStepSzf params); // ����ͨ����ͨ���˲���ʵ��ִ����

  
	private:

        bool         m_fAvaible;

		////////////////////////////////////////////
        // following is page-locked memory in Host

        PtrStepSzf   m_ptrInFrame;         // ������-�ӳ����豸ȡ�õ�����֡���㷨������������ݾ���

        PtrStepSzf   m_ptrDispMat;         // ������-λ�����ݾ���

        PtrStepSzf   m_ptrStrainMat;        // ������-Ӧ�����ݾ���


		////////////////////////////////////////////
        // following memory in Cuda Device 

        PtrStepSzf   m_ptrInFrame_d;       // �豸��-�ӳ����豸ȡ�õ�����֡���㷨������������ݾ���

        PtrStepSzf   m_ptrDispMat_d;       // �豸��-λ�����ݾ���

        PtrStepSzf   m_ptrStrainMat_d;     // �豸��-Ӧ�����ݾ���

        PtrStepSzf   m_ptrFilterBuf_d;     // �豸��-�˲�������м仺����

        PtrStepSzf   m_ptrBpParams_d;      // �豸��-��ͨ�˲��Ĳ���

        PtrStepSzf   m_ptrLpParams_d;      // �豸��-��ͨ�˲��Ĳ���

        PtrStepSzf   m_ptrMatchParams_d;   // �豸��-ƥ���˲��Ĳ���

        PtrStepSzf   m_ptrTemplateMat_d;   // �豸��-λ�ƴ���ģ������

        PtrStepSzf   m_ptrObjectMat_d;     // �豸��-λ�ƴ�����������

        PtrStepSz<float*>  m_ptrResultMats_d;  // �豸��-λ�ƴ����������

        PtrStepSzf   m_ptrRandonIn_d;      // �豸��-Ӧ�䴦�������任����������

        PtrStepSzf   m_ptrRandonOut_d;     // �豸��-Ӧ�䴦�������仯���������

	};
}