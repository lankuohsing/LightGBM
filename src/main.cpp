#include <iostream>
#include <LightGBM/application.h>
#include <LightGBM/c_api.h>
#include<string>
#include<climits>
#include<LightGBM/dataset.h>
int main(int argc, char** argv) {
	
  try {
	  //char aa[2][200] = { "D:/Projects/github/LightGBM/windows/x64/Release/lightgbm.exe","config=D:/Projects/github/LightGBM/windows/x64/Release/train.conf" };
	  //char aa[2][200] = { "D:/Projects/github/LightGBM/windows/x64/Release/lightgbm.exe","config=D:/Projects/github/LightGBM/windows/x64/Release/predict.conf" };
	  char aa[2][200] = { "D:/Projects/github/LightGBM/windows/x64/Release/lightgbm.exe","config=D:/Projects/github/LightGBM/windows/x64/Release/train1.conf" };
	  int ac = 2;
	  char *pp [2] = { aa[0],aa[1] };
    LightGBM::Application app(ac, pp);
    app.Run();
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
  /*
	char filename[] = "D:/Projects/github/LightGBM/windows/rank.train";
	char parameters[] = "task=train\
 boosting_type=gbdt\
 objective=lambdarank\
 metric=ndcg\
 ndcg_eval_at=1,3,5\
 metric_freq=1\
 is_training_metric=true\
 max_bin=255\
 data=rank.train\
 valid_data=rank.test\
 num_trees=100\
 learning_rate=0.1\
 num_leaves=31\
 tree_learner=serial\
 feature_fraction=1.0\
 bagging_freq=1\
 bagging_fraction=0.9\
 min_data_in_leaf=50\
 min_sum_hessian_in_leaf=5.0\
 is_enable_sparse=true\
 use_two_round_loading=false\
 is_save_binary_file=false\
 output_model=LightGBM_model.txt\
 num_machines=1\
 local_listen_port=12400\
 machine_list_file=mlist.txt";
	//char *str1 = new char[INT_MAX];
	//DatasetHandle *datasetHandle1 = str1;
	//LightGBM::Dataset *dataset = new LightGBM::Dataset();
	DatasetHandle  dataset_p=NULL;
	DatasetHandle *dataset_pp=&dataset_p;

	int *out2 = new int();
	LGBM_DatasetCreateFromFile( filename, parameters, nullptr, dataset_pp );
	LGBM_DatasetGetNumData( dataset_pp, out2 );
	//delete[] str1;
	//delete dataset;
	*/
	system( "pause" );
	return 0;
	
	
}
