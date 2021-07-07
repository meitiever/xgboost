/*!
 * Copyright 2014-2020 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of xgboost.
 *  This file is not included in dynamic library.
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <dmlc/timer.h>

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include <iomanip>
#include <ctime>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/io.h"
#include "common/common.h"
#include "common/config.h"
#include "common/version.h"
#include "c_api/c_api_utils.h"

#include <utils.h>
#include <pre-processing.h>
#include <post-processing.h>
#include <boost/filesystem.hpp>

#include <ros/ipa_building_msgs/map-segmentation-results.h>
#include <ipa_room_segmentation/evaluation_segmentation.h>
#include <ipa_room_segmentation/distance_segmentation.h>
#include <ipa_room_segmentation/morphological_segmentation.h>
#include <ipa_room_segmentation/voronoi_segmentation.h>
#include <ipa_room_segmentation/adaboost_classifier.h>
 //#include <ipa_room_segmentation/xgboost_classifier.h>
#include <ipa_room_segmentation/voronoi_random_field_segmentation.h>

using namespace cv;
using namespace std;

namespace xgboost {
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

  constexpr char const* const CLIParam::kNull;

  DMLC_REGISTER_PARAMETER(CLIParam);

  std::string CliHelp() {
    return "Use xgboost -h for showing help information.\n";
  }

  void CLIError(dmlc::Error const& e) {
    std::cerr << "Error running xgboost:\n\n"
      << e.what() << "\n"
      << CliHelp()
      << std::endl;
  }

  std::shared_ptr<DMatrix> MatToDMat(cv::Mat& e) {
    std::shared_ptr<DMatrix> t;
    return t;
  }

  class CLI {
    CLIParam param_;
    std::unique_ptr<Learner> learner_;
    enum Print {
      kNone,
      kVersion,
      kHelp
    } print_info_{ kNone };

    int ResetLearner(std::vector<std::shared_ptr<DMatrix>> const& matrices) {
      learner_.reset(Learner::Create(matrices));
      int version = rabit::LoadCheckPoint(learner_.get());
      if (version == 0) {
        if (param_.model_in != CLIParam::kNull) {
          this->LoadModel(param_.model_in, learner_.get());
          learner_->SetParams(param_.cfg);
        }
        else {
          learner_->SetParams(param_.cfg);
        }
      }
      learner_->Configure();
      return version;
    }

    void CLITrain() {
      const double tstart_data_load = dmlc::GetTime();
      if (rabit::IsDistributed()) {
        std::string pname = rabit::GetProcessorName();
        LOG(CONSOLE) << "start " << pname << ":" << rabit::GetRank();
      }
      // load in data.
      std::shared_ptr<DMatrix> dtrain(DMatrix::Load(
        param_.train_path,
        ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
        param_.dsplit == 2));
      std::vector<std::shared_ptr<DMatrix>> deval;
      std::vector<std::shared_ptr<DMatrix>> cache_mats;
      std::vector<std::shared_ptr<DMatrix>> eval_datasets;
      cache_mats.push_back(dtrain);
      for (size_t i = 0; i < param_.eval_data_names.size(); ++i) {
        deval.emplace_back(std::shared_ptr<DMatrix>(DMatrix::Load(
          param_.eval_data_paths[i],
          ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
          param_.dsplit == 2)));
        eval_datasets.push_back(deval.back());
        cache_mats.push_back(deval.back());
      }
      std::vector<std::string> eval_data_names = param_.eval_data_names;
      if (param_.eval_train) {
        eval_datasets.push_back(dtrain);
        eval_data_names.emplace_back("train");
      }
      // initialize the learner.
      int32_t version = this->ResetLearner(cache_mats);
      LOG(INFO) << "Loading data: " << dmlc::GetTime() - tstart_data_load
        << " sec";

      // start training.
      const double start = dmlc::GetTime();
      for (int i = version / 2; i < param_.num_round; ++i) {
        double elapsed = dmlc::GetTime() - start;
        if (version % 2 == 0) {
          LOG(INFO) << "boosting round " << i << ", " << elapsed
            << " sec elapsed";
          learner_->UpdateOneIter(i, dtrain);
          if (learner_->AllowLazyCheckPoint()) {
            rabit::LazyCheckPoint(learner_.get());
          }
          else {
            rabit::CheckPoint(learner_.get());
          }
          version += 1;
        }
        CHECK_EQ(version, rabit::VersionNumber());
        std::string res = learner_->EvalOneIter(i, eval_datasets, eval_data_names);
        if (rabit::IsDistributed()) {
          if (rabit::GetRank() == 0) {
            LOG(TRACKER) << res;
          }
        }
        else {
          LOG(CONSOLE) << res;
        }
        if (param_.save_period != 0 && (i + 1) % param_.save_period == 0 &&
          rabit::GetRank() == 0) {
          std::ostringstream os;
          os << param_.model_dir << '/' << std::setfill('0') << std::setw(4)
            << i + 1 << ".model";
          this->SaveModel(os.str(), learner_.get());
        }

        if (learner_->AllowLazyCheckPoint()) {
          rabit::LazyCheckPoint(learner_.get());
        }
        else {
          rabit::CheckPoint(learner_.get());
        }
        version += 1;
        CHECK_EQ(version, rabit::VersionNumber());
      }
      LOG(INFO) << "Complete Training loop time: " << dmlc::GetTime() - start
        << " sec";
      // always save final round
      if ((param_.save_period == 0 ||
        param_.num_round % param_.save_period != 0) &&
        rabit::GetRank() == 0) {
        std::ostringstream os;
        if (param_.model_out == CLIParam::kNull) {
          os << param_.model_dir << '/' << std::setfill('0') << std::setw(4)
            << param_.num_round << ".model";
        }
        else {
          os << param_.model_out;
        }
        this->SaveModel(os.str(), learner_.get());
      }

      double elapsed = dmlc::GetTime() - start;
      LOG(INFO) << "update end, " << elapsed << " sec in all";
    }

    void CLIDumpModel() {
      FeatureMap fmap;
      if (param_.name_fmap != CLIParam::kNull) {
        std::unique_ptr<dmlc::Stream> fs(
          dmlc::Stream::Create(param_.name_fmap.c_str(), "r"));
        dmlc::istream is(fs.get());
        fmap.LoadText(is);
      }
      // load model
      CHECK_NE(param_.model_in, CLIParam::kNull) << "Must specify model_in for dump";
      this->ResetLearner({});

      // dump data
      std::vector<std::string> dump =
        learner_->DumpModel(fmap, param_.dump_stats, param_.dump_format);
      std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(param_.name_dump.c_str(), "w"));
      dmlc::ostream os(fo.get());
      if (param_.dump_format == "json") {
        os << "[" << std::endl;
        for (size_t i = 0; i < dump.size(); ++i) {
          if (i != 0) {
            os << "," << std::endl;
          }
          os << dump[i];  // Dump the previously generated JSON here
        }
        os << std::endl << "]" << std::endl;
      }
      else {
        for (size_t i = 0; i < dump.size(); ++i) {
          os << "booster[" << i << "]:\n";
          os << dump[i];
        }
      }
      // force flush before fo destruct.
      os.set_stream(nullptr);
    }

    void CLIPredict() {
      //std::vector<double> angles_for_simulation_;
      //for (double angle = 0; angle < 360; angle++) {
      //  angles_for_simulation_.push_back(angle);
      //}

      //LaserScannerRaycasting raycasting_;

      CHECK_NE(param_.test_path, CLIParam::kNull)
        << "Test dataset parameter test:data must be specified.";

      //boost::filesystem::path p(param_.test_path);
      //if (boost::filesystem::is_regular_file(p)) {
      //  cv::Mat original_map_to_be_labeled = cv::imread(param_.test_path, 0);

      //  for (int y = 0; y < original_map_to_be_labeled.rows; y++) {
      //    LaserScannerFeatures lsf;

      //    for (int x = 0; x < original_map_to_be_labeled.cols; x++) {
      //      if (original_map_to_be_labeled.at<unsigned char>(y, x) == 255) {
      //        std::vector<double> temporary_beams;
      //        raycasting_.raycasting(original_map_to_be_labeled, cv::Point(x, y), temporary_beams);
      //        std::vector<float> temporary_features;
      //        cv::Mat features_mat; //OpenCV expects a 32-floating-point Matrix as feature input
      //        lsf.get_features(temporary_beams, angles_for_simulation_, cv::Point(x, y), features_mat);
      //        //classify each Point
      //        std::shared_ptr<DMatrix> dtest = MatToDMat(features_mat);
      //        HostDeviceVector<bst_float> preds;
      //        learner_->Predict(dtest, param_.pred_margin, &preds, param_.iteration_begin,
      //          param_.iteration_end);
      //        LOG(CONSOLE) << "Writing prediction to " << param_.name_pred;

      //        double probability_for_room = 0.0;
      //        double probability_for_hallway = 0.0 * (1.0 - probability_for_room);

      //        if (probability_for_room > probability_for_hallway) {
      //          original_map_to_be_labeled.at<unsigned char>(y, x) = 150; //label it as room
      //        }
      //        else {
      //          original_map_to_be_labeled.at<unsigned char>(y, x) = 100; //label it as hallway
      //        }
      //      }
      //    }
      //  }
      //}

      // load data
      std::shared_ptr<DMatrix> dtest(DMatrix::Load(
        param_.test_path,
        ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
        param_.dsplit == 2));
      // load model
      CHECK_NE(param_.model_in, CLIParam::kNull) << "Must specify model_in for predict";
      this->ResetLearner({});

      LOG(INFO) << "Start prediction...";
      HostDeviceVector<bst_float> preds;
      if (param_.ntree_limit != 0) {
        param_.iteration_end = GetIterationFromTreeLimit(param_.ntree_limit, learner_.get());
        LOG(WARNING) << "`ntree_limit` is deprecated, use `iteration_begin` and "
          "`iteration_end` instead.";
      }
      learner_->Predict(dtest, param_.pred_margin, &preds, param_.iteration_begin,
        param_.iteration_end);
      LOG(CONSOLE) << "Writing prediction to " << param_.name_pred;

      std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(param_.name_pred.c_str(), "w"));
      dmlc::ostream os(fo.get());
      for (bst_float p : preds.ConstHostVector()) {
        os << std::setprecision(std::numeric_limits<bst_float>::max_digits10) << p
          << '\n';
      }
      // force flush before fo destruct.
      os.set_stream(nullptr);
    }

    void LoadModel(std::string const& path, Learner* learner) const {
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

    void SaveModel(std::string const& path, Learner* learner) const {
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

    void PrintHelp() const {
      std::cout << "Usage: xgboost [ -h ] [ -V ] [ config file ] [ arguments ]" << std::endl;
      std::stringstream ss;
      ss << R"(
  Options and arguments:

    -h, --help
       Print this message.

    -V, --version
       Print XGBoost version.

    arguments
       Extra parameters that are not specified in config file, see below.

  Config file specifies the configuration for both training and testing.  Each line
  containing the [attribute] = [value] configuration.

  General XGBoost parameters:

    https://xgboost.readthedocs.io/en/latest/parameter.html

  Command line interface specfic parameters:

)";

      std::string help = param_.__DOC__();
      auto splited = common::Split(help, '\n');
      for (auto str : splited) {
        ss << "    " << str << '\n';
      }
      ss << R"(    eval[NAME]: string, optional, default='NULL'
        Path to evaluation data, with NAME as data name.
)";

      ss << R"(
  Example:  train.conf

    # General parameters
    booster = gbtree
    objective = reg:squarederror
    eta = 1.0
    gamma = 1.0
    seed = 0
    min_child_weight = 0
    max_depth = 3

    # Training arguments for CLI.
    num_round = 2
    save_period = 0
    data = "demo/data/agaricus.txt.train?format=libsvm"
    eval[test] = "demo/data/agaricus.txt.test?format=libsvm"

  See demo/ directory in XGBoost for more examples.
)";
      std::cout << ss.str() << std::endl;
    }

    void PrintVersion() const {
      auto ver = Version::String(Version::Self());
      std::cout << "XGBoost: " << ver << std::endl;
    }

  public:
    CLI(int argc, char* argv[]) {
      if (argc < 2) {
        this->PrintHelp();
        exit(1);
      }
      for (int i = 0; i < argc; ++i) {
        std::string str{ argv[i] };
        if (str == "-h" || str == "--help") {
          print_info_ = kHelp;
          break;
        }
        else if (str == "-V" || str == "--version") {
          print_info_ = kVersion;
          break;
        }
      }
      if (print_info_ != kNone) {
        return;
      }

      rabit::Init(argc, argv);
      std::string config_path = argv[1];

      common::ConfigParser cp(config_path);
      auto cfg = cp.Parse();

      for (int i = 2; i < argc; ++i) {
        char name[256], val[256];
        if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
          cfg.emplace_back(std::string(name), std::string(val));
        }
      }

      param_.Configure(cfg);
    }

    int Run() {
      switch (this->print_info_) {
      case kNone:
        break;
      case kVersion: {
        this->PrintVersion();
        return 0;
      }
      case kHelp: {
        this->PrintHelp();
        return 0;
      }
      }

      try {
        switch (param_.task) {
        case kTrain:
          CLITrain();
          break;
        case kDumpModel:
          CLIDumpModel();
          break;
        case kPredict:
          CLIPredict();
          break;
        }
      }
      catch (dmlc::Error const& e) {
        xgboost::CLIError(e);
        return 1;
      }
      return 0;
    }

    ~CLI() {
      rabit::Finalize();
    }
  };
}  // namespace xgboost

static void help(char** argv)
{
  cout << "\nThis sample program demonstrates the room segmentation algorithm\n"
    << "Call:\n"
    << argv[0] << " --root D:\\Github\\Tools\\xgboost\\" << endl
    << " --train_semantic true" << endl
    << " --pase_to_xgboost true" << endl;
}

int main(int argc, char* argv[]) {
  cv::CommandLineParser parser(argc, argv,
    "{help h ? |      | help message}"
    "{root     | D:\\Github\\Tools\\xgboost\\ | root path of file }"
    "{train_semantic | false | train adaboost classifier. }"
    "{train_vrf      | false | train vrf classifier. }"
    "{pase_to_xgboost| false | parse features to xgboost libsvm format. }"
  );

  if (parser.has("help"))
  {
    help(argv);
    return 0;
  }

  bool save_to_csv = parser.get<bool>("pase_to_xgboost");
  if (save_to_csv) {
    std::cout << "parameter pase_to_xgboost set to true." << std::endl;
  }

  std::string package_path = parser.get<std::string>("root");

  if (!boost::filesystem::exists(package_path))
  {
    cerr << "Path : " << package_path << " does not exists." << endl;
    exit(-1);
  }

  std::vector<std::string> map_names;
  map_names.push_back("sineva01");
  map_names.push_back("sineva02");
  map_names.push_back("sineva03");
  map_names.push_back("sineva04");
  map_names.push_back("sineva05");
  map_names.push_back("sineva06");
  map_names.push_back("sineva07");
  map_names.push_back("sineva08");
  map_names.push_back("sineva09");
  map_names.push_back("sineva10");
  map_names.push_back("sineva11");
  map_names.push_back("sineva12");
  map_names.push_back("sineva13");
  map_names.push_back("sineva14");
  map_names.push_back("sineva15");
  map_names.push_back("sineva16");
  map_names.push_back("sineva17");
  map_names.push_back("sineva18");
  map_names.push_back("sineva19");
  map_names.push_back("sineva20");
  map_names.push_back("sineva21");
  map_names.push_back("sineva22");
  map_names.push_back("sineva23");
  map_names.push_back("sineva24");
  map_names.push_back("sineva25");
  map_names.push_back("sineva26");
  map_names.push_back("sineva27");
  map_names.push_back("sineva28");
  map_names.push_back("sineva29");
  map_names.push_back("sineva30");
  map_names.push_back("sineva31");
  map_names.push_back("sineva32");
  map_names.push_back("lab_ipa");
  map_names.push_back("lab_c_scan");
  map_names.push_back("Freiburg52_scan");
  map_names.push_back("Freiburg79_scan");
  map_names.push_back("lab_b_scan");
  map_names.push_back("lab_intel");
  map_names.push_back("Freiburg101_scan");
  map_names.push_back("lab_d_scan");
  map_names.push_back("lab_f_scan");
  map_names.push_back("lab_a_scan");
  map_names.push_back("NLB");
  map_names.push_back("office_a");
  map_names.push_back("office_b");
  map_names.push_back("office_c");
  map_names.push_back("office_d");
  map_names.push_back("office_e");
  map_names.push_back("office_f");
  map_names.push_back("office_g");
  map_names.push_back("office_h");
  map_names.push_back("office_i");
  map_names.push_back("lab_ipa_furnitures");
  map_names.push_back("lab_c_scan_furnitures");
  map_names.push_back("Freiburg52_scan_furnitures");
  map_names.push_back("Freiburg79_scan_furnitures");
  map_names.push_back("lab_b_scan_furnitures");
  map_names.push_back("lab_intel_furnitures");
  map_names.push_back("Freiburg101_scan_furnitures");
  map_names.push_back("lab_d_scan_furnitures");
  map_names.push_back("lab_f_scan_furnitures");
  map_names.push_back("lab_a_scan_furnitures");
  map_names.push_back("NLB_furnitures");
  map_names.push_back("office_a_furnitures");
  map_names.push_back("office_b_furnitures");
  map_names.push_back("office_c_furnitures");
  map_names.push_back("office_d_furnitures");
  map_names.push_back("office_e_furnitures");
  map_names.push_back("office_f_furnitures");
  map_names.push_back("office_g_furnitures");
  map_names.push_back("office_h_furnitures");
  map_names.push_back("office_i_furnitures");

  std::vector<uint> possible_labels(3); // vector that stores the possible labels that are drawn in the training maps. Order: room - hallway - doorway
  possible_labels[0] = 77;
  possible_labels[1] = 115;
  possible_labels[2] = 179;

  std::string map_path = package_path + "files\\test_maps\\";
  const std::string segmented_map_path = package_path + "files\\segmented_maps\\";

  // strings that stores the path to the saving files
  std::string conditional_weights_path = package_path + "files\\classifier_models\\conditional_field_weights.txt";
  std::string boost_file_path = package_path + "files\\classifier_models\\";

  // optimal result saving path
  std::string conditional_weights_optimal_path = package_path + "files\\classifier_models\\vrf_conditional_field_weights.txt";
  std::string boost_file_optimal_path = package_path + "files\\classifier_models\\";

  bool train_semantic_, train_vrf_;
  train_semantic_ = parser.get<bool>("train_semantic");
  train_vrf_ = parser.get<bool>("train_vrf");

  if (train_vrf_) {
    std::cout << "parameter train_vrf set to true." << std::endl;

    // load the training maps
    cv::Mat training_map;
    std::vector<cv::Mat> training_maps;
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_Fr52.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_Fr101.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_intel.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_d_furniture.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_ipa.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_NLB_furniture.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_office_e.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_office_h.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_c_furnitures.png", 0);
    training_maps.push_back(training_map);
    // load the voronoi maps
    std::vector<cv::Mat> voronoi_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/Fr52_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/Fr101_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_intel_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_d_furnitures_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_ipa_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/NLB_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/office_e_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/office_h_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_c_furnitures_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    // load the voronoi-nodes maps
    std::vector<cv::Mat> voronoi_node_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/Fr52_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/Fr101_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_intel_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_d_furnitures_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_ipa_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/NLB_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/office_e_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/office_h_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_c_furnitures_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    // load the original maps
    std::vector<cv::Mat> original_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/Fr52_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/Fr101_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_intel_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_d_furnitures_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_ipa_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/NLB_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/office_e_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/office_h_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_c_furnitures_original.png", 0);
    original_maps.push_back(training_map);

    std::cout << "voronoi_random_field_training maps:" << "training maps:" << training_maps.size() << " voronoi maps:" << voronoi_maps.size() << " voronoi node maps:" << voronoi_node_maps.size() << " original maps:" << original_maps.size() << std::endl;
  }

  // list of files containing maps with room labels for training the semantic segmentation
  std::vector<std::string> semantic_training_maps_room_file_list_;
  // list of files containing maps with hallway labels for training the semantic segmentation
  std::vector<std::string> semantic_training_maps_hallway_file_list_;

  if (train_semantic_) {
    std::cout << "parameter train_semantic set to true." << std::endl;
    std::vector<std::string> fileNames;
    if (boost::filesystem::exists(package_path + "files\\training_maps\\sineva\\")) {
      if (getFileNames(package_path + "files\\training_maps\\sineva\\", fileNames) < 1) {
        std::cout << package_path + "files\\training_maps\\sineva\\" << " does not exists or empty." << std::endl;
        return 0;
      }
      for (auto item : fileNames) {
        if (item.find("room_train") != std::string::npos)
          semantic_training_maps_room_file_list_.push_back(item);
        if (item.find("hallway_train") != std::string::npos)
          semantic_training_maps_hallway_file_list_.push_back(item);
      }
    }

    AdaboostClassifier semantic_segmentation;
    const std::string classifier_default_path = package_path + "files\\classifier_models\\";
    const std::string classifier_path = package_path + "files\\classifier_models\\";

    for (size_t i = 0; i < semantic_training_maps_room_file_list_.size(); ++i)
      std::cout << semantic_training_maps_room_file_list_[i] << std::endl;

    for (size_t i = 0; i < semantic_training_maps_hallway_file_list_.size(); ++i)
      std::cout << semantic_training_maps_hallway_file_list_[i] << std::endl;

    std::cout << "You have chosen to train an adaboost classifier for the semantic segmentation method." << std::endl;

    // load the training maps, change to your maps when you want to train different ones
    std::vector<cv::Mat> room_training_maps;

    for (size_t i = 0; i < semantic_training_maps_room_file_list_.size(); ++i) {
      cv::Mat training_map = cv::imread(semantic_training_maps_room_file_list_[i], 0);
      room_training_maps.push_back(training_map);
    }

    std::vector<cv::Mat> hallway_training_maps;

    for (size_t i = 0; i < semantic_training_maps_hallway_file_list_.size(); ++i) {
      cv::Mat training_map = cv::imread(semantic_training_maps_hallway_file_list_[i], 0);
      hallway_training_maps.push_back(training_map);
    }

    //train the algorithm
    semantic_segmentation.trainClassifiers(room_training_maps, hallway_training_maps, classifier_path, save_to_csv);
    std::exit(0);
  }

  if (!boost::filesystem::exists(segmented_map_path)) {
    const std::string command = "mkdir -p " + segmented_map_path;
    int return_value = system(command.c_str());
    std::cout << "segmented map folder " << segmented_map_path << " created.'" << std::endl;
  }

  double map_resolution = 0.05;
  std::vector<cv::Point> door_points;
  std::vector<cv::Point> doorway_points_; // vector that saves the found doorway points, when using the 5th algorithm (vrf)
  geometry_msgs::Pose map_origin;

  PreProcessor pre;
  std::vector<std::string> segmentation_names;
  segmentation_names.push_back("1morphological");
  segmentation_names.push_back("2distance");
  segmentation_names.push_back("3voronoi");
  segmentation_names.push_back("4semantic");
  segmentation_names.push_back("5vrf");
  segmentation_names.push_back("6xgboost");

  std::vector<cv::Mat> results(segmentation_names.size());
  for (size_t i = 0; i < segmentation_names.size(); ++i)
    results[i] = cv::Mat::zeros(26, map_names.size(), CV_64FC1);

  // loop through map files
  for (size_t image_index = 0; image_index < map_names.size(); ++image_index) {
    //define vectors to save the parameters
    std::vector<int> segments_number_vector(segmentation_names.size());
    std::vector<double> av_area_vector(segmentation_names.size()), max_area_vector(segmentation_names.size()), min_area_vector(segmentation_names.size()), dev_area_vector(segmentation_names.size());
    std::vector<double> av_per_vector(segmentation_names.size()), max_per_vector(segmentation_names.size()), min_per_vector(segmentation_names.size()), dev_per_vector(segmentation_names.size());
    std::vector<double> av_compactness_vector(segmentation_names.size()), max_compactness_vector(segmentation_names.size()), min_compactness_vector(segmentation_names.size()), dev_compactness_vector(segmentation_names.size());
    std::vector<double> av_bb_vector(segmentation_names.size()), max_bb_vector(segmentation_names.size()), min_bb_vector(segmentation_names.size()), dev_bb_vector(segmentation_names.size());
    std::vector<double> av_quo_vector(segmentation_names.size()), max_quo_vector(segmentation_names.size()), min_quo_vector(segmentation_names.size()), dev_quo_vector(segmentation_names.size());
    std::vector<bool> reachable(segmentation_names.size());

    //load map
    std::string map_name = map_names[image_index];
    std::string image_filename = map_path + map_name + ".png";
    std::cout << "map: " << image_filename << std::endl;
    cv::Mat map = cv::imread(image_filename.c_str(), 0);
    cv::Mat original_img;

    bool preprocess = true;
    if (preprocess) {
      //pre-process the image.
      pre.Process(map, original_img);
    }
    else
      original_img = map.clone();

    //segment the given map
    cv::Mat segmented_map;

    //calculate parameters for each segmentation and save it
    for (size_t segmentation_index = 0; segmentation_index < segmentation_names.size(); ++segmentation_index) {
      std::cout << "Evaluating image '" << map_name << "' with segmentation method " << segmentation_names[segmentation_index] << std::endl;

      const int room_segmentation_algorithm = segmentationNameToNumber(segmentation_names[segmentation_index]);

      if (room_segmentation_algorithm == 1) { //morpho
        double room_lower_limit_morphological_ = 0.8;
        double room_upper_limit_morphological_ = 47.0;
        std::cout << "You have chosen the morphological segmentation." << std::endl;
        MorphologicalSegmentation morphological_segmentation; //morphological segmentation method
        morphological_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_morphological_, room_upper_limit_morphological_);
      }

      if (room_segmentation_algorithm == 2) { //distance
        double room_lower_limit_distance_ = 0.35;
        double room_upper_limit_distance_ = 163.0;
        std::cout << "You have chosen the distance segmentation." << std::endl;
        DistanceSegmentation distance_segmentation; //distance segmentation method
        distance_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_distance_, room_upper_limit_distance_);
      }

      if (room_segmentation_algorithm == 3) { //voronoi
        double room_lower_limit_voronoi_ = 0.1;
        double room_upper_limit_voronoi_ = 1000000.;
        double voronoi_neighborhood_index_ = 280;
        double max_iterations_ = 150;
        double min_critical_point_distance_factor_ = 0.5;
        double max_area_for_merging_ = 12.5;
        bool display_segmented_map_ = false;
        std::cout << "You have chosen the Voronoi segmentation" << std::endl;
        VoronoiSegmentation voronoi_segmentation; //voronoi segmentation method
        voronoi_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_voronoi_, room_upper_limit_voronoi_,
          voronoi_neighborhood_index_, max_iterations_, min_critical_point_distance_factor_, max_area_for_merging_, display_segmented_map_);
      }

      if (room_segmentation_algorithm == 4) { //semantic
        std::cout << "You have chosen the semantic segmentation." << std::endl;
        double room_lower_limit_semantic_ = 1.0;
        double room_upper_limit_semantic_ = 1000000.;
        AdaboostClassifier semantic_segmentation; //semantic segmentation method
        const std::string classifier_path = package_path + "files\\classifier_models\\";
        semantic_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_semantic_, room_upper_limit_semantic_,
          classifier_path);
      }

      if (room_segmentation_algorithm == 5) { //voronoi random field
        double room_area_lower_limit_voronoi_random = 1.53;
        double room_area_upper_limit_voronoi_random = 1000000.;
        double max_iterations = 150;
        double voronoi_random_field_epsilon_for_neighborhood = 7;
        double min_neighborhood_size = 5;
        double min_voronoi_random_field_node_distance = 7.0;
        double max_voronoi_random_field_inference_iterations = 9000;
        double max_area_for_merging = 12.5;
        doorway_points_.clear();
        std::cout << "You have chosen the Voronoi random field segmentation." << std::endl;
        continue;
      }

      if (room_segmentation_algorithm == 6) { //xgboost
        doorway_points_.clear();
        const std::string classifier_path = package_path + "files\\classifier_models\\";
        //XgboostClassifier xgClassifier;
        //xgClassifier.segmentMap(original_img, segmented_map, map_resolution, classifier_path);
        std::cout << "You have chosen the Voronoi random field segmentation." << std::endl;
        continue;
      }

      if (room_segmentation_algorithm == 99) {
        // pass through segmentation: takes a map which is already separated into unconnected areas and returns these as the resulting segmentation in the format of this program
        // todo: closing operation explicitly for bad maps --> needs parameterization
        // original_img.convertTo(segmented_map, CV_32SC1, 256, 0);
        // occupied space = 0, free space = 65280
        cv::Mat original_img_eroded, temp;
        cv::erode(original_img, temp, cv::Mat(), cv::Point(-1, -1), 3);
        cv::dilate(temp, original_img_eroded, cv::Mat(), cv::Point(-1, -1), 3);
        original_img_eroded.convertTo(segmented_map, CV_32SC1, 256, 0);     // occupied space = 0, free space = 65280
        int label_index = 1;
        double room_upper_limit_passthrough_ = 1000000.0;
        double room_lower_limit_passthrough_ = 1.0;

        //cv::imshow("original_img", original_img_eroded);
        //cv::waitKey();

        for (int y = 0; y < segmented_map.rows; y++) {
          for (int x = 0; x < segmented_map.cols; x++) {
            // if original map is occupied space here or if the segmented map has already received a label for that cell --> skip
            if (original_img_eroded.at<uchar>(y, x) != 255 || segmented_map.at<int>(y, x) != 65280)
            {
              continue;
            }

            // fill each room area with a unique id
            cv::Rect rect;
            cv::floodFill(segmented_map, cv::Point(x, y), label_index, &rect, 0, 0, 4);

            // determine filled area
            double area = 0;

            for (int v = rect.y; v < segmented_map.rows; v++)
              for (int u = rect.x; u < segmented_map.cols; u++)
                if (segmented_map.at<int>(v, u) == label_index)
                {
                  area += 1.;
                }

            area = map_resolution * map_resolution * area;  // convert from cells to m^2

            // exclude too small and too big rooms
            if (area < room_lower_limit_passthrough_ || area > room_upper_limit_passthrough_) {
              for (int v = rect.y; v < segmented_map.rows; v++)
                for (int u = rect.x; u < segmented_map.cols; u++)
                  if (segmented_map.at<int>(v, u) == label_index)
                  {
                    segmented_map.at<int>(v, u) = 0;
                  }
            }
            else
            {
              label_index++;
            }
          }
        }
      }

      cv::Mat index_map;
      PostProcessor post;
      ipa_building_msgs::MapSegmentationResult result;
      post.Process(segmented_map, doorway_points_, map_origin, index_map, result, 0.3, map_resolution);

      //do not need original segmented map below.
      segmented_map = index_map.clone();
      std::cout << "Depth of Segmented Map: " << segmented_map.depth() << ", Number of Channels: " << segmented_map.channels() << std::endl;

      // generate colored segmented_map
      cv::Mat color_segmented_map;
      segmented_map.convertTo(color_segmented_map, CV_8U);
      cv::cvtColor(color_segmented_map, color_segmented_map, cv::COLOR_GRAY2BGR);

      for (size_t i = 1; i <= result.room_information_in_pixel.size(); ++i) {
        //choose random color for each room
        const cv::Vec3b color((rand() % 250) + 1, (rand() % 250) + 1, (rand() % 250) + 1);

        for (size_t v = 0; v < segmented_map.rows; ++v)
          for (size_t u = 0; u < segmented_map.cols; ++u)
            if (segmented_map.at<int>(v, u) == i)
              color_segmented_map.at<cv::Vec3b>(v, u) = color;
      }

      std::string image_filename = segmented_map_path + map_name + "_segmented_" + segmentation_names[segmentation_index] + ".png";
      cv::imwrite(image_filename, color_segmented_map);
      std::cout << "segmented map saved to " << image_filename << std::endl;

      // evaluation: numeric properties
      // ==============================
      std::cout << "evaluation: numeric properties " << std::endl;
      std::vector<double> areas;
      std::vector<double> perimeters;
      std::vector<double> area_perimeter_compactness;
      std::vector<double> bb_area_compactness;
      std::vector<double> pca_eigenvalue_ratio;

      calculate_basic_measures(segmented_map, map_resolution, static_cast<int>(result.room_information_in_pixel.size()), areas, perimeters, area_perimeter_compactness, bb_area_compactness, pca_eigenvalue_ratio);

      // runtime
      results[segmentation_index].at<double>(0, image_index) = 0.0;

      // number of segments
      results[segmentation_index].at<double>(1, image_index) = segments_number_vector[segmentation_index] = areas.size();

      // area
      //std::vector<double> areas = calculate_areas_from_segmented_map(segmented_map, (int)result->room_information_in_pixel.size());
      double average = 0.0;
      double max_area = 0.0;
      double min_area = 100000000;
      calculate_mean_min_max(areas, average, min_area, max_area);
      results[segmentation_index].at<double>(2, image_index) = av_area_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(3, image_index) = max_area_vector[segmentation_index] = max_area;
      results[segmentation_index].at<double>(4, image_index) = min_area_vector[segmentation_index] = min_area;
      results[segmentation_index].at<double>(5, image_index) = dev_area_vector[segmentation_index] = calculate_stddev(areas, average);

      // perimeters
      //std::vector<double> perimeters = calculate_perimeters(saved_contours);
      average = 0.0;
      double max_per = 0.0;
      double min_per = 100000000;
      calculate_mean_min_max(perimeters, average, min_per, max_per);
      results[segmentation_index].at<double>(6, image_index) = av_per_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(7, image_index) = max_per_vector[segmentation_index] = max_per;
      results[segmentation_index].at<double>(8, image_index) = min_per_vector[segmentation_index] = min_per;
      results[segmentation_index].at<double>(9, image_index) = dev_per_vector[segmentation_index] = calculate_stddev(perimeters, average);

      // area compactness
      //std::vector<double> area_perimeter_compactness = calculate_compactness(saved_contours);
      average = 0.0;
      double max_compactness = 0;
      double min_compactness = 100000000;
      calculate_mean_min_max(area_perimeter_compactness, average, min_compactness, max_compactness);
      results[segmentation_index].at<double>(10, image_index) = av_compactness_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(11, image_index) = max_compactness_vector[segmentation_index] = max_compactness;
      results[segmentation_index].at<double>(12, image_index) = min_compactness_vector[segmentation_index] = min_compactness;
      results[segmentation_index].at<double>(13, image_index) = dev_compactness_vector[segmentation_index] = calculate_stddev(area_perimeter_compactness, average);

      // bounding box
      //std::vector<double> bb_area_compactness = calculate_bounding_error(saved_contours);
      average = 0.0;
      double max_error = 0;
      double min_error = 10000000;
      calculate_mean_min_max(bb_area_compactness, average, min_error, max_error);
      results[segmentation_index].at<double>(14, image_index) = av_bb_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(15, image_index) = max_bb_vector[segmentation_index] = max_error;
      results[segmentation_index].at<double>(16, image_index) = min_bb_vector[segmentation_index] = min_error;
      results[segmentation_index].at<double>(17, image_index) = dev_bb_vector[segmentation_index] = calculate_stddev(bb_area_compactness, average);

      // quotient
      //std::vector<double> pca_eigenvalue_ratio = calc_Ellipse_axis(saved_contours);
      average = 0.0;
      double max_quo = 0.0;
      double min_quo = 100000000;
      calculate_mean_min_max(pca_eigenvalue_ratio, average, min_quo, max_quo);
      results[segmentation_index].at<double>(18, image_index) = av_quo_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(19, image_index) = max_quo_vector[segmentation_index] = max_quo;
      results[segmentation_index].at<double>(20, image_index) = min_quo_vector[segmentation_index] = min_quo;
      results[segmentation_index].at<double>(21, image_index) = dev_quo_vector[segmentation_index] = calculate_stddev(pca_eigenvalue_ratio, average);

      //          // retrieve room contours
      //          cv::Mat temporary_map = segmented_map.clone();
      //          std::vector<std::vector<cv::Point> > contours, saved_contours;
      //          std::vector<cv::Vec4i> hierarchy;
      //          for(size_t i = 1; i <= result->room_information_in_pixel.size(); ++i)
      //          {
      //              cv::Mat single_room_map = cv::Mat::zeros(segmented_map.rows, segmented_map.cols, CV_8UC1);
      //              for(size_t v = 0; v < segmented_map.rows; ++v)
      //                  for(size_t u = 0; u < segmented_map.cols; ++u)
      //                      if(segmented_map.at<int>(v,u) == i)
      //                          single_room_map.at<uchar>(v,u) = 255;
      //              cv::findContours(single_room_map, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
      //              cv::drawContours(temporary_map, contours, -1, cv::Scalar(0), cv::FILLED);
      //              for (int c = 0; c < contours.size(); c++)
      //              {
      //                  if (map_resolution * map_resolution * cv::contourArea(contours[c]) > 1.0)
      //                  {
      //                      saved_contours.push_back(contours[c]);
      //                  }
      //              }
      //          }
      //          // reachability
      //          if (check_reachability(saved_contours, segmented_map))
      //          {
      //              reachable[segmentation_index] = true;
      //          }
      //          else
      //          {
      //              reachable[segmentation_index] = false;
      //          }

      std::cout << "Basic measures computed." << std::endl;

      // evaluation: against ground truth segmentation
      // =============================================
      // load ground truth segmentation (just borders painted in between of rooms/areas, not colored yet --> coloring will be done here)
      bool has_gt = false;
      if (has_gt) {
        std::string map_name_basic = map_name;
        std::size_t pos = map_name.find("_furnitures");

        if (pos != std::string::npos)
          map_name_basic = map_name.substr(0, pos);

        std::string gt_image_filename = package_path + map_name_basic + "_gt_segmentation.png";
        std::cout << "Loading ground truth segmentation from: " << gt_image_filename << std::endl;
        cv::Mat gt_map = cv::imread(gt_image_filename.c_str(), CV_8U);

        // compute recall and precision, store colored gt segmentation
        double precision_micro, precision_macro, recall_micro, recall_macro;
        cv::Mat gt_map_color;
        EvaluationSegmentation es;
        es.computePrecisionRecall(gt_map, gt_map_color, segmented_map, precision_micro, precision_macro, recall_micro, recall_macro, true);
        std::string gt_image_filename_color = segmented_map_path + map_name + "_gt_color_segmentation.png"; //ros::package::getPath("ipa_room_segmentation") + "/common/files/test_maps/" + map_name + "_gt_color_segmentation.png";
        cv::imwrite(gt_image_filename_color.c_str(), gt_map_color);

        results[segmentation_index].at<double>(22, image_index) = recall_micro;
        results[segmentation_index].at<double>(23, image_index) = recall_macro;
        results[segmentation_index].at<double>(24, image_index) = precision_micro;
        results[segmentation_index].at<double>(25, image_index) = precision_macro;
      }
    }

    //write parameters into file
    std::stringstream output;
    output << "--------------Segmentierungsevaluierung----------------" << std::endl;

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << segmentation_names[i] << " & ";
    }

    output << std::endl;
    output << "Kompaktheitsmaße: " << std::endl;
    output << "Durschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "****************************" << std::endl;

    output << "Überflüssige Fläche Bounding Box: " << std::endl;
    output << "Durchschnitt Bounding Fehler: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Flächenmaße: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Umfangsmaße: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    //      output << "Erreichbarkeit für alle Raumzentren: " << std::endl;
    //      for(size_t i = 0; i < segmentation_names.size(); ++i)
    //      {
    //          if(reachable[i] == true)
    //              output << "Alle Raumzentren erreichbar" << std::endl;
    //          else
    //              output << "Nicht alle erreichbar" << std::endl;
    //      }
    //      output << "****************************" << std::endl;

    output << "Quotienten der Ellipsenachsen: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Anzahl Räume: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << segments_number_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Recall/Precision: " << std::endl;
    output << "recall_micro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(22, image_index) << " & ";
    }

    output << std::endl;
    output << "recall_macro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(23, image_index) << " & ";
    }

    output << std::endl;
    output << "precision_micro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(24, image_index) << " & ";
    }

    output << std::endl;
    output << "precision_macro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(25, image_index) << " & ";
    }

    output << std::endl;

    std::string log_filename = segmented_map_path + map_name + "_evaluation.txt";
    std::ofstream file(log_filename.c_str(), std::ios::out);

    if (file.is_open() == true)
    {
      file << output.str();
    }

    file.close();

    // write results summary to file (overwrite in each cycle in order to avoid loosing all data on crash)
    for (size_t segmentation_index = 0; segmentation_index < segmentation_names.size(); ++segmentation_index) {
      std::string log_filename = segmented_map_path + segmentation_names[segmentation_index] + "_evaluation_summary.txt";
      std::ofstream file(log_filename.c_str(), std::ios::out);

      if (file.is_open() == true) {
        for (int r = 0; r < results[segmentation_index].rows; ++r) {
          for (int c = 0; c < results[segmentation_index].cols; ++c)
          {
            file << results[segmentation_index].at<double>(r, c) << "\t";
          }

          file << std::endl;
        }
      }

      file.close();
    }
  }

  return 0;
}
