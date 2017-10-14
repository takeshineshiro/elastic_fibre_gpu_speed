#pragma   once 

#include "opencv/cv.h"
#include <string>
#include "Elasto.h"

// 该宏开关仅仅是为了验证opencv的c++和c接口的区别，实际效果差不多；c稍稍快点。但是c++的接口明显更方便
#define  USE_CPP   1

class  CDataset;
class  CStrainPlus;
class  CDisplacement;
class  CFilterPlus;
struct ConfigParam;

//////////////////////////////////////////////////////////////////////////
//   
//
//////////////////////////////////////////////////////////////////////////
class ElastoWorker
{
public:
	static  ElastoWorker & Instance();

	int        Init(const std::string &config_filepath);

	// calculate Elato strain and veocity
	int        Do(const EInput &in, EOutput &out);

	EPHandler  RegisterHandler(EPHandler new_handler, void *param)
	{
		EPHandler old = m_pHandler;
		m_pHandler = new_handler;
		m_pHandlerParam = param;
		return old;
	}

	void       UnregisterHandler(EPHandler)
	{
		m_pHandler      = NULL;
		m_pHandlerParam = NULL;
	}

	bool          CudaIsCapable() const { return m_fCudaIsCapable; }

	ConfigParam & GetConfigParam()  { return *m_pConfigParam; }

	cv::Mat&      GetOutmat()  { return m_outData; }

protected:

private:

	ElastoWorker();

	~ElastoWorker();

	void    Release();

	// Update Config param as read config file
	void    UpdateConfig(std::string &);

private:

	std::string     m_strConfigFilepath;
	CDataset      * m_pDataset;         // 
	CStrainPlus   * m_pStrain;
	CDisplacement        * m_pDisplacement;
	CFilterPlus  * m_pFilterBp;     // band pass
	CFilterPlus   * m_pFilterLp;    // low  pass
	ConfigParam   * m_pConfigParam;
	EPHandler       m_pHandler;
	void          * m_pHandlerParam;

	bool            m_fCudaIsCapable; // 为了使用新的cuda类，设置的布尔变量；true表示可以使用cuda的设备

	cv::Mat         m_outData; // 用于在各个阶段传递数据。它就是一个处理单元的输入数据矩阵，也是处理后的结果数据矩阵。然后这个数据结果作为下一个处理单元的输入。诸如此类。
};