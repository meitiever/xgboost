#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>

#include <ctime>
#include <stdlib.h>
#include <rabit/rabit.h>

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include "../src/common/io.h"

#include <ipa_room_segmentation/features.h>
#include <ipa_room_segmentation/raycasting.h>

using namespace xgboost;

enum CLITask {
  kTrain = 0,
  kDumpModel = 1,
  kPredict = 2
};

struct CLIParam : public XGBoostParameter<CLIParam> {
  /*! \brief the task name */
  int task;
  /*! \brief whether evaluate training statistics */
  bool eval_train;
  /*! \brief number of boosting iterations */
  int num_round;
  /*! \brief the period to save the model, 0 means only save the final round model */
  int save_period;
  /*! \brief the path of training set */
  std::string train_path;
  /*! \brief path of test dataset */
  std::string test_path;
  /*! \brief the path of test model file, or file to restart training */
  std::string model_in;
  /*! \brief the path of final model file, to be saved */
  std::string model_out;
  /*! \brief the path of directory containing the saved models */
  std::string model_dir;
  /*! \brief name of predict file */
  std::string name_pred;
  /*! \brief data split mode */
  int dsplit;
  /*!\brief limit number of trees in prediction */
  int ntree_limit;
  int iteration_begin;
  int iteration_end;
  /*!\brief whether to directly output margin value */
  bool pred_margin;
  /*! \brief whether dump statistics along with model */
  int dump_stats;
  /*! \brief what format to dump the model in */
  std::string dump_format;
  /*! \brief name of feature map */
  std::string name_fmap;
  /*! \brief name of dump file */
  std::string name_dump;
  /*! \brief the paths of validation data sets */
  std::vector<std::string> eval_data_paths;
  /*! \brief the names of the evaluation data used in output log */
  std::vector<std::string> eval_data_names;
  /*! \brief all the configurations */
  std::vector<std::pair<std::string, std::string> > cfg;

  static constexpr char const* const kNull = "NULL";

  // declare parameters
  DMLC_DECLARE_PARAMETER(CLIParam) {
    // NOTE: declare everything except eval_data_paths.
    DMLC_DECLARE_FIELD(task).set_default(kTrain)
      .add_enum("train", kTrain)
      .add_enum("dump", kDumpModel)
      .add_enum("pred", kPredict)
      .describe("Task to be performed by the CLI program.");
    DMLC_DECLARE_FIELD(eval_train).set_default(false)
      .describe("Whether evaluate on training data during training.");
    DMLC_DECLARE_FIELD(num_round).set_default(10).set_lower_bound(1)
      .describe("Number of boosting iterations");
    DMLC_DECLARE_FIELD(save_period).set_default(0).set_lower_bound(0)
      .describe("The period to save the model, 0 means only save final model.");
    DMLC_DECLARE_FIELD(train_path).set_default("NULL")
      .describe("Training data path.");
    DMLC_DECLARE_FIELD(test_path).set_default("NULL")
      .describe("Test data path.");
    DMLC_DECLARE_FIELD(model_in).set_default("NULL")
      .describe("Input model path, if any.");
    DMLC_DECLARE_FIELD(model_out).set_default("NULL")
      .describe("Output model path, if any.");
    DMLC_DECLARE_FIELD(model_dir).set_default("./")
      .describe("Output directory of period checkpoint.");
    DMLC_DECLARE_FIELD(name_pred).set_default("pred.txt")
      .describe("Name of the prediction file.");
    DMLC_DECLARE_FIELD(dsplit).set_default(0)
      .add_enum("auto", 0)
      .add_enum("col", 1)
      .add_enum("row", 2)
      .describe("Data split mode.");
    DMLC_DECLARE_FIELD(ntree_limit).set_default(0).set_lower_bound(0)
      .describe("(Deprecated) Use iteration_begin/iteration_end instead.");
    DMLC_DECLARE_FIELD(iteration_begin).set_default(0).set_lower_bound(0)
      .describe("Begining of boosted tree iteration used for prediction.");
    DMLC_DECLARE_FIELD(iteration_end).set_default(0).set_lower_bound(0)
      .describe("End of boosted tree iteration used for prediction.  0 means all the trees.");
    DMLC_DECLARE_FIELD(pred_margin).set_default(false)
      .describe("Whether to predict margin value instead of probability.");
    DMLC_DECLARE_FIELD(dump_stats).set_default(false)
      .describe("Whether dump the model statistics.");
    DMLC_DECLARE_FIELD(dump_format).set_default("text")
      .describe("What format to dump the model in.");
    DMLC_DECLARE_FIELD(name_fmap).set_default("NULL")
      .describe("Name of the feature map file.");
    DMLC_DECLARE_FIELD(name_dump).set_default("dump.txt")
      .describe("Name of the output dump text file.");
    // alias
    DMLC_DECLARE_ALIAS(train_path, data);
    DMLC_DECLARE_ALIAS(test_path, test:data);
    DMLC_DECLARE_ALIAS(name_fmap, fmap);
  }
  // customized configure function of CLIParam
  inline void Configure(const std::vector<std::pair<std::string, std::string> >& _cfg) {
    // Don't copy the configuration to enable parameter validation.
    auto unknown_cfg = this->UpdateAllowUnknown(_cfg);
    this->cfg.emplace_back("validate_parameters", "True");
    for (const auto& kv : unknown_cfg) {
      if (!strncmp("eval[", kv.first.c_str(), 5)) {
        char evname[256];
        CHECK_EQ(sscanf(kv.first.c_str(), "eval[%[^]]", evname), 1)
          << "must specify evaluation name for display";
        eval_data_names.emplace_back(evname);
        eval_data_paths.push_back(kv.second);
      }
      else {
        this->cfg.emplace_back(kv);
      }
    }
    // constraint.
    if (name_pred == "stdout") {
      save_period = 0;
    }
    if (dsplit == 0 && rabit::IsDistributed()) {
      dsplit = 2;
    }
  }
};

class XgboostClassifier {
protected:
  std::unique_ptr<Learner> learner_;
  bool trained_; // variable that shows if the classifiers has already been trained
  CLIParam param_;
  std::vector<double> angles_for_simulation_; // angle-vector used to calculate the features for this algorithm
  LaserScannerRaycasting raycasting_;

public:

  XgboostClassifier();

  int resetLearner(std::vector<std::shared_ptr<DMatrix>> const& matrices) {
    learner_.reset(Learner::Create(matrices));
    int version = rabit::LoadCheckPoint(learner_.get());
    if (version == 0) {
      if (param_.model_in != CLIParam::kNull) {
        this->loadModel(param_.model_in, learner_.get());
        learner_->SetParams(param_.cfg);
      }
      else {
        learner_->SetParams(param_.cfg);
      }
    }
    learner_->Configure();
    return version;
  }

  void saveModel(std::string const& path, Learner* learner) const {
    learner->Configure();
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(path.c_str(), "w"));
    if (common::FileExtension(path) == "json") {
      Json out{ Object() };
      learner->SaveModel(&out);
      std::string str;
      Json::Dump(out, &str);
      fo->Write(str.c_str(), str.size());
    }
    else {
      learner->SaveModel(fo.get());
    }
  }

  void loadModel(std::string const& path, Learner* learner) const {
    if (common::FileExtension(path) == "json") {
      auto str = common::LoadSequentialFile(path);
      CHECK_GT(str.size(), 2);
      CHECK_EQ(str[0], '{');
      Json in{ Json::Load({str.c_str(), str.size()}) };
      learner->LoadModel(in);
    }
    else {
      std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(path.c_str(), "r"));
      learner->LoadModel(fi.get());
    }
  }

  //labeling-algorithm after the training
  bool segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
    const std::string& model_in);
};
