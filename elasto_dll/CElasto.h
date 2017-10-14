#pragma   once 

#include "opencv/cv.h"
#include <string>
#include "Elasto.h"

// �ú꿪�ؽ�����Ϊ����֤opencv��c++��c�ӿڵ�����ʵ��Ч����ࣻc���Կ�㡣����c++�Ľӿ����Ը�����
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

	bool            m_fCudaIsCapable; // Ϊ��ʹ���µ�cuda�࣬���õĲ���������true��ʾ����ʹ��cuda���豸

	cv::Mat         m_outData; // �����ڸ����׶δ������ݡ�������һ������Ԫ���������ݾ���Ҳ�Ǵ����Ľ�����ݾ���Ȼ��������ݽ����Ϊ��һ������Ԫ�����롣������ࡣ
};