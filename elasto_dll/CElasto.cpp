

#include "stdafx.h"
#include "CDisplacement.h"
#include "CData.h"
#include "CStrain.h"
#include "CFilter.h"
#include "CElasto.h"
#include "Elasto.h"
#include "opencv/highgui.h"
#include "FileUtility.h"

#include <iostream>
#include <time.h>
#include <stdio.h>
#include "Log.h"
#include "TestTime.h"
#include "SysConfig.h"
#include "ImageFunc.h"
#include <opencv2/gpu/gpu.hpp>
#include "ElastoCudaExt.h"


//#pragma comment(lib, "CudaDLL.lib")


ELASTO_DLL_API int ElastoInit_gpu(const char *configfile)
{
    return ElastoWorker::Instance().Init(configfile);
}

ELASTO_DLL_API void ElastoRelease_gpu()
{

}

ELASTO_DLL_API int ElastoProcess_gpu(const EInput &in, EOutput &out)
{
    return ElastoWorker::Instance().Do(in, out);
}

ELASTO_DLL_API EPHandler ElastoRegisterHandler_gpu(EPHandler handler, void *lpParam)
{
    return ElastoWorker::Instance().RegisterHandler(handler, lpParam);
}

ELASTO_DLL_API void ElastoUnregisterHandler_gpu(EPHandler handler)
{
    ElastoWorker::Instance().UnregisterHandler(handler);
}


ElastoWorker & ElastoWorker::Instance()
{
	static ElastoWorker  s_objElatoWorker;
	return s_objElatoWorker;
}

ElastoWorker::ElastoWorker()
{
	m_pConfigParam = NULL;
	m_pHandler     = NULL;
	m_pFilterBp    = NULL;
	m_pFilterLp    = NULL;
	m_pDataset     = NULL;
	m_pDisplacement= NULL;
	m_pStrain      = NULL;
	m_pHandlerParam= NULL;

	m_fCudaIsCapable= false;
	m_strConfigFilepath = "";
}

ElastoWorker::~ElastoWorker()
{
	Release();
}

void ElastoWorker::UpdateConfig(std::string &file)
{
	ReadSysConfig(file.c_str(), *m_pConfigParam);
	m_pConfigParam->prf = 1 / 300e-6;
}

int  ElastoWorker::Init(const std::string &config_filepath)
{
    // 初始化次序不要改变
    m_strConfigFilepath = config_filepath;

	m_pConfigParam = new ConfigParam;
	ReadSysConfig(config_filepath.c_str(), *m_pConfigParam);
	m_pConfigParam->prf = 1 / 300e-6;

	m_fCudaIsCapable = elasto_cuda::Wrapper::Instance().Init(m_pConfigParam, config_filepath);

	m_pDataset      = new CDataset(m_pConfigParam->shearFrameLineNum, m_pConfigParam->sampleNumPerLine);
	m_pFilterLp     = new CFilterPlus(m_pConfigParam->lpfilt_file.c_str());
	m_pFilterBp     = new CFilterPlus(m_pConfigParam->bpfilt_file.c_str());
	m_pDisplacement = new CDisplacement();
    m_pStrain       = new CStrainPlus();

	if (m_fCudaIsCapable)
	{
		float  params[256];
		int    n = 256;
		
		m_pFilterBp->GetParams(params, n);
		elasto_cuda::Wrapper::Instance().InitBandpassFilterParams(params, n);

		n = 256;// getparam内部会检查params数据区的长度
		m_pFilterLp->GetParams(params, n);
		elasto_cuda::Wrapper::Instance().InitLowpassFilterParams(params, n);

		n = 256;
		m_pDisplacement->GetMatchFilterParams(params, n);
		elasto_cuda::Wrapper::Instance().InitMatchFilterParams(params, n);
	}

	return 0;
}



void ElastoWorker::Release()
{
	if (m_pConfigParam)  delete m_pConfigParam;
	if (m_pDataset)      delete m_pDataset;
	if (m_pDisplacement) delete m_pDisplacement;
	if (m_pFilterBp)     delete m_pFilterBp;
	if (m_pFilterLp)     delete m_pFilterLp;
	if (m_pStrain)       delete m_pStrain;
}

//////////////////////////////////////////////////////////////////////////
//  把数据保存到硬盘上会耗费大量时间，在正常情况下是不必要的。
//  对于需要显示的图片，可以通过定义在内存中的对象来实现，而不是通过硬盘上的文件来传递。
//  我暂时关闭了保存数据和图片的代码，strain.bmp还存在。
//                         杨戈， 2016.07.27
//////////////////////////////////////////////////////////////////////////
int  ElastoWorker::Do(const EInput &in, EOutput &out)
{
    CTestTime   ttime;
    CString     info;
    long        timeout;

    m_pDataset->ReadData(in.pDatas);
    m_pDataset->GetSubData(m_outData, m_pConfigParam->box_x, m_pConfigParam->box_y, m_pConfigParam->box_w, m_pConfigParam->box_h);

	ttime.run();
	(m_fCudaIsCapable) ? elasto_cuda::Wrapper::Instance().BpFilter(m_outData) :	m_pFilterBp->Do();// 带通滤波
    timeout = ttime.stop();

    info.Format(_T("\nbpfilter-timeout=%dms\n"), timeout);
    std::cout << info.GetString();
	
	// 通知观察者
	if (m_pHandler)
	{
		(*m_pHandler)(EP_POST_FILTER, &CvMat(m_outData), m_pHandlerParam);
	}

	//MakeImage(bpfilt->outDataMat, in.filepath_d);
	//SaveDataFile("bpfilt.dat", m_outData);//保存滤波后文件，正常运行时没有必要保存。调试时可以打开

	ttime.run();
	m_pDisplacement->Do();  //求位移
    timeout = ttime.stop();

    info.Format(_T("displace-timeout=%dms\n"), timeout);
    std::cout << info.GetString();

	//CLog::Instance()->Write(info.GetString(), info.GetLength());

	ttime.run();
	(m_fCudaIsCapable) ? elasto_cuda::Wrapper::Instance().LpFilter(m_outData) :	m_pFilterLp->Do();//低通滤波
	timeout = ttime.stop();

	info.Format("lowpass-timeout=%dms\n", timeout);
    std::cout << info.GetString();
	//CLog::Instance()->Write(info.GetString(), info.GetLength());

	//MakeImage(lpfilt->outDataMat, "disp_lp.bmp");
	//SaveDataFile("disp_lp.dat", lpfilt->outDataMat); // 正常运行时没有必要保存。调试时可以打开

	//MakeImage(m_outData, in.filepath_d);
	//SaveDataFile("displace.dat", m_outData);//保存位移文件，正常运行时没有必要保存。调试时可以打开

	if (m_pHandler)// 通知观察者
	{
		(*m_pHandler)(EP_POST_DISPLACEMENT,  &CvMat(m_outData), m_pHandlerParam);
	}
    
	ttime.run();
	(m_fCudaIsCapable) ? m_pStrain->CalcStrain_cuda(in, out) : m_pStrain->CalcStrain(in, out);//求应变图及杨氏模量
    timeout = ttime.stop();

    info.Format("strain-timeout=%dms\n", timeout);
    std::cout << info.GetString();

	return 0;
}