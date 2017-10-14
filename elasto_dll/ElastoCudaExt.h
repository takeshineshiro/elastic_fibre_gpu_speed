//////////////////////////////////////////////////////////////////////////
// Elato Cuda ʵ�ֵ����ͷ�ļ�
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

    typedef  float  ElemType;// cuda������̣����ݵ�Ԫ����������
    typedef  float* PtrElemType;// cuda������̣����ݵ�Ԫ����������

	enum
	{
		ERR_OK = 0,
		ERR_NO_GPU = -1,                     // ��ǰϵͳû��GPU;
		ERR_CUDA_ARCH_NOT_MATCHED = -2,      // ��ǰGPU�ļ����������㣻
	};

    //////////////////////////////////////////////////////////////////////////
    // ������cudaʵ������Ҫ�����ݾ����size��������ͳһ������
    //
    const char * const g_szMatSizeNamespace[] = {
        "RawFrame",        // Ӳ���豸�ϴ���ԭʼ����֡
        "MOI",             // matric of intersting, ����Ȥ�����������������뵯�Դ�������ݾ���������RawFrame���Ӽ�
        "DispMat",         // λ��-����֡
        "TmplMat",         // ģ�����ݾ���
        "ObjectMat",       // Ŀ�����ݾ���
        "ResultMat",       // ��ģ��ƥ�䣩������ݾ���
        "ResultMats",      // ��ģ��ƥ�䣩������ݾ���ľ���Ϊ�˲��д���ÿ���̱߳�����һ������
        "StrainMat",       // Ӧ��-����֡
        "StrainImg",       // Ӧ��-ͼ��֡�� ����StrainMat��ת�þ�����ʾ����Ҫ
        "RandonInMat",     // Ӧ����㣬 �����任���������
        "RandonOutMat",    // Ӧ����㣬 �����任���������
    };

    //////////////////////////////////////////////////////////////////////////
    //  ���Բ���&������CUDAʵ�ֵĽӿ��ࣨ����װ��CUDA��ʵ�ֹ��̣���
    //  ���ṩ�˱�Ҫ��CUDAʵ�ֵ��㷨ʵ�֣����봮�еĲ���������CPUʵ�֡�
    //
	class ELASTO_CUDA_EX_CLASS Wrapper {
	public:
		static Wrapper & Instance();
		
        ~Wrapper();
	
		// ��ʼ��
		// false�� ʧ�ܣ����ܽ��м��㣻 true, �ɹ������Խ��м���
		bool  Init(const ConfigParam *lpParam, const std::string &config_filepath);

        void  InitBandpassFilterParams(float *pDatas, int num);

        void  InitLowpassFilterParams (float *pDatas, int num);

		void  InitMatchFilterParams   (float *pDatas, int num);

		bool  IsCapable() const // fasle, ���ܵ���GPU���٣� true�� ���Ե���GPU����
		{
			return m_nInitFlag == 1;
		}

		int   GetLastError() const { return m_nError; }

		friend class CudaComputer;

    public:
        //////////////////////////////////////////////////////////////////////////
        // CUDAʵ�ֵ��㷨�ӿ�
        void  CalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat, cv::Mat &strImage);

		void  CalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int step, int win, int scale);

        //////////////////////////////////////////////////////////////////////////
        //  �ڸ�������׶Σ����������������ݾ����в�ͬ��size�� 
        //  ����ͳһ����ʹ�����ֽ�������
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

		int          m_nInitFlag; // 0, δ��ʼ����  1�� ��ʼ���ɹ�

		std::string  m_strConfigFilepath;

		ConfigParam  *m_lpConfigParam;
		
		CudaComputer *m_lpCudaWorker;
	};

}

