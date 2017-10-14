//////////////////////////////////////////////////////////////////////////
// Elato Cuda 实现的输出头文件
//
//////////////////////////////////////////////////////////////////////////

#pragma  once


#include <string>
#include <opencv2/opencv.hpp>

#ifndef ELASTO_CUDA_API  
#define ELASTO_CUDA_API  __declspec(dllimport)
#endif

#ifndef ELASTO_CUDA_EX_CLASS  
#define ELASTO_CUDA_EX_CLASS  __declspec(dllimport)
#endif


struct ConfigParam;

namespace  elasto_cuda {

    class  CudaComputer;

    typedef  float  ElemType;// cuda计算过程，数据单元的数据类型
    typedef  float* PtrElemType;// cuda计算过程，数据单元的数据类型

	enum
	{
		ERR_OK = 0,
		ERR_NO_GPU = -1,                     // 当前系统没有GPU;
		ERR_CUDA_ARCH_NOT_MATCHED = -2,      // 当前GPU的计算能力不足；
	};

    //////////////////////////////////////////////////////////////////////////
    // 所有在cuda实现中需要的数据矩阵的size，在这里统一命名。
    //
    const char * const g_szMatSizeNamespace[] = {
        "RawFrame",        // 硬件设备上传的原始数据帧
        "MOI",             // matric of intersting, 感兴趣的数据区，真正进入弹性处理的数据矩形区，是RawFrame的子集
        "DispMat",         // 位移-数据帧
        "TmplMat",         // 模板数据矩阵
        "ObjectMat",       // 目标数据矩阵
        "ResultMat",       // （模板匹配）结果数据矩阵
        "ResultMats",      // （模板匹配）结果数据矩阵的矩阵，为了并行处理；每个线程必须有一个矩阵
        "StrainMat",       // 应变-数据帧
        "StrainImg",       // 应变-图像帧， 它是StrainMat的转置矩阵，显示的需要
        "RandonInMat",     // 应变计算， 拉东变换的输入矩阵
        "RandonOutMat",    // 应变计算， 拉东变换的输出矩阵
    };

    //////////////////////////////////////////////////////////////////////////
    //  弹性测量&计算库的CUDA实现的接口类（它封装了CUDA的实现过程）。
    //  它提供了必要的CUDA实现的算法实现，必须串行的部分则留给CPU实现。
    //
	class ELASTO_CUDA_EX_CLASS Wrapper {
	public:
		static Wrapper & Instance();
		
        ~Wrapper();
	
		// 初始化
		// false， 失败；不能进行计算； true, 成功，可以进行计算
		bool  Init(const ConfigParam *lpParam, const std::string &config_filepath);

        void  InitBandpassFilterParams(float *pDatas, int num);

        void  InitLowpassFilterParams (float *pDatas, int num);

		void  InitMatchFilterParams   (float *pDatas, int num);

		bool  IsCapable() const // fasle, 不能调用GPU加速； true， 可以调用GPU加速
		{
			return m_nInitFlag == 1;
		}

		int   GetLastError() const { return m_nError; }

		friend class CudaComputer;

    public:
        //////////////////////////////////////////////////////////////////////////
        // CUDA实现的算法接口
        void  CalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat, cv::Mat &strImage);

		void  CalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int step, int win, int scale);

        //////////////////////////////////////////////////////////////////////////
        //  在各个处理阶段，输入或者输出的数据矩阵有不同的size， 
        //  这里统一处理，使用名字进行区分
        cv::Size  GetMatSize(const std::string &name);

        void  BpFilter(cv::Mat &in_out);

        void  LpFilter(cv::Mat &in_out);

		void  MeanFilter(cv::Mat &image, int step);

		void  MatchFilter(cv::Mat &image);

		void  RadonSum(cv::Mat &in, cv::Mat &out);

		void  Rel2AbsDisp(cv::Mat &disp);

	protected:

	private:
		Wrapper();

	private:

		int          m_nError;

		int          m_nInitFlag; // 0, 未初始化；  1， 初始化成功

		std::string  m_strConfigFilepath;

		ConfigParam  *m_lpConfigParam;
		
		CudaComputer *m_lpCudaWorker;
	};

}

