
#define ELASTO_CUDA_EX_CLASS  __declspec(dllexport)
#define ELASTO_CUDA_API       __declspec(dllexport)
#include "stdafx.h"
#include "SysConfig.h"
#include "ElastoCudaExt.h"
#include "CudaComputer.cuh"
#include <assert.h>

namespace elasto_cuda
{


	Wrapper & Wrapper::Instance()
	{
		static Wrapper s_instance;;
        return s_instance;
	}



	Wrapper::Wrapper():m_nError(0), m_nInitFlag(0),m_lpCudaWorker(0),m_lpConfigParam(0)
	{
	}



	Wrapper::~Wrapper()
	{
		// 下面的次序不能改动
		if (m_lpCudaWorker)  delete m_lpCudaWorker;
		if (m_lpConfigParam) delete m_lpConfigParam;
	}



	bool Wrapper::Init(const ConfigParam *lpParam, const std::string &config_filepath)
	{
		if (m_lpConfigParam == 0)  
		{
			m_lpConfigParam = new ConfigParam;
		}
		assert(lpParam);
		assert(!config_filepath.empty());
		m_strConfigFilepath = config_filepath;
		*m_lpConfigParam = *lpParam;

		if (m_lpCudaWorker == 0)
		{
			m_lpCudaWorker = new CudaComputer();
		}
        return m_lpCudaWorker->Init(*lpParam) == 0;
	}



    cv::Size Wrapper::GetMatSize(const std::string &name)
    {
        int    multiWin  = m_lpConfigParam->multiWin;   //大窗口对小窗口的倍数
        int    winSize   = m_lpConfigParam->windowHW;
        int    stepSize  = m_lpConfigParam->step;
        int    winNum    = (m_lpConfigParam->box_w - multiWin * winSize) / stepSize;  //计算需要匹配的段数
        
        const int points = m_lpConfigParam->fitline_pts;	//用于拟合的点数

        cv::Size size;
        if (name.compare("RawFrame") == 0)
        {
            size.width  = m_lpConfigParam->sampleNumPerLine;
            size.height = m_lpConfigParam->shearFrameLineNum;
        }
        else if (name.compare("MOI") == 0)
        {
            size.width  = m_lpConfigParam->box_w;
            size.height = m_lpConfigParam->box_h;
        }
        else if (name.compare("DispMat") == 0)
        {
            size.width  = winNum;
            size.height = m_lpConfigParam->box_h - 1;
        }
        else if (name.compare("TmplMat") == 0)
        {
            size.width  = winSize;
            size.height = 1;
        }
        else if (name.compare("ObjectMat") == 0)
        {
            size.width  = winSize * multiWin;
            size.height = 1;
        }
        else if (name.compare("ResultMat") == 0)
        {
            size.width  = winSize + 1;
            size.height = 1;
        }
        else if (name.compare("ResultMats") == 0)
        {
            size = GetMatSize("DispMat");
        }
        else if (name.compare("StrainImg") == 0)
        {
            // 应变图-在位移矩阵的基础上做了转置，行列转了90度
            // 同时减去了做直线拟合的点的数量
            size = GetMatSize("DispMat");
            int   width  = size.height;
            int   height = size.width - points + 1;
           
            size.width   = width;
            size.height  = height;
        }
        else if (name.compare("StrainMat") == 0)
        {
            size = GetMatSize("DispMat");
            int   width  = size.width - points + 1;
            int   height = size.height;
            
            size.width = width;
            size.height = height;
        }
        else if (name.compare("RandonInMat") == 0)
        {
            size = GetMatSize("StrainMat");

            int width  = (m_lpConfigParam->sb_h < 0) ? size.height : m_lpConfigParam->sb_h;
            int heigth = (m_lpConfigParam->sb_w < 0) ? size.width  : m_lpConfigParam->sb_w;
            size.height = heigth;
            size.width  = width;
        }
        else if (name.compare("RandonOutMat") == 0)
        {
            size = GetMatSize("RandonInMat");
            int width  = size.width;
            int heigth = size.width - 1;
            size.height = heigth;
            size.width  = width;
        }
        else
        {
			size.width  = 0;
			size.height = 0;
        }
        return size;
    }




    void  Wrapper::CalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat, cv::Mat &strImage)
    {
        if (m_lpCudaWorker)
        {
            //  strain image 是 strain Mat的转置，因此只要计算strainMat即可，这样可以减少host和device的数据传输
            m_lpCudaWorker->DoCalcStrain(in, count, strainMat);

            cv::Mat tmp = strainMat.t();
			tmp.copyTo(strImage);
            strImage = strImage * 100.0f;
        }
    }




	void  Wrapper::CalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int win, int step, int scale)
	{	
		if (m_lpCudaWorker)
		{
			cv::Size size = GetMatSize("DispMat");
			dispMat.create(size.height, size.width, inData.type());
			m_lpCudaWorker->DoCalcDisplacement(inData, dispMat, win, step, scale);
		}
	}



    void  Wrapper::InitBandpassFilterParams(float *pDatas, int num)
    {
        if (m_lpCudaWorker)
        {
            m_lpCudaWorker->InitBandpassFilterParams(pDatas, num);
        }
    }



    void  Wrapper::InitLowpassFilterParams(float *pDatas, int num)
    {
        if (m_lpCudaWorker)
        {
            m_lpCudaWorker->InitLowpassFilterParams(pDatas, num);
        }
    }



    void  Wrapper::InitMatchFilterParams(float *pDatas, int num)
    {
        if (m_lpCudaWorker)
        {
            m_lpCudaWorker->InitMatchFilterParams(pDatas, num);
        }
    }



	void  Wrapper::BpFilter(cv::Mat &in_out)
	{
		if (m_lpCudaWorker)
		{
			m_lpCudaWorker->DoBpFilter(in_out);
		}
	}



	void  Wrapper::LpFilter(cv::Mat &in_out)
	{
		if (m_lpCudaWorker)
		{
			m_lpCudaWorker->DoLpFilter(in_out);
		}
	}



	void  Wrapper::RadonSum(cv::Mat &in, cv::Mat &out)
	{
		if (m_lpCudaWorker)
		{
			cv::Size size = GetMatSize("RandonOutMat");
			out = cv::Mat::zeros(size, in.type());
			m_lpCudaWorker->DoRandonSum(in, out);
		}
	}



	void  Wrapper::MeanFilter(cv::Mat &image, int step)
	{

		if (m_lpCudaWorker)
		{
			cv::Mat  tmpMat(image.rows, image.cols + step - 1, image.type());
			
#if 0
			// 下面的实现有问题，还没有找到原因
			int r, c;
			for (c = 0; c < step - 1; c++)
			{
				tmpMat.col(c) = image.col(0);
			}
			for (c = step - 1; c < tmpMat.cols; c++)
			{
				tmpMat.col(c) = image.col(c - step + 1);
			}
#else			
			for (int j = 0; j < step - 1; j++)
			{
				for (int i = 0; i < image.rows; i++)
				{
					*tmpMat.ptr<float>(i, j) = *image.ptr<float>(i, 0);
				}
			}

			for (int j = step - 1; j < image.cols + step - 1; j++)
			{
				for (int i = 0; i < image.rows; i++)
				{
					*tmpMat.ptr<float>(i, j) = *image.ptr<float>(i, j - step + 1);
				}
			}
#endif
			m_lpCudaWorker->MeanFilter(tmpMat, image, step);
		}
	}





	void  Wrapper::MatchFilter(cv::Mat &image)
	{
		if (m_lpCudaWorker)
		{
			m_lpCudaWorker->MatchFilter(image);
		}
	}



	void  Wrapper::Rel2AbsDisp(cv::Mat &disp)
	{
		if (m_lpCudaWorker)
		{
			m_lpCudaWorker->Rel2AbsDisp(disp);
		}
	}






}