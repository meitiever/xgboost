#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <ipa_room_segmentation/timer.h>
#include <ipa_room_segmentation/contains.h>
#include <ipa_room_segmentation/features.h>

#include <ipa_room_segmentation/xgboost_classifier.h>
#include <ipa_room_segmentation/wavefront_region_growing.h>

#include "../src/common/config.h"

XgboostClassifier::XgboostClassifier(std::string& config_path) {
  for (double angle = 0; angle < 360; angle++) {
    angles_for_simulation_.push_back(angle);
  }

  trained_ = false;
  //std::string model = model_path + "semantic_room_xgboost_r100.model";

  common::ConfigParser cp(config_path);
  auto cfg = cp.Parse();

  param_.Configure(cfg);
}

bool XgboostClassifier::segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription) {
  if (!trained_) {
    // load model
    resetLearner({});
  }

  std::cout << "Start prediction...";
  HostDeviceVector<bst_float> preds;
  //if (param_.ntree_limit != 0) {
  //  param_.iteration_end = getIterationFromTreeLimit(param_.ntree_limit, learner_.get());
  //  LOG(WARNING) << "`ntree_limit` is deprecated, use `iteration_begin` and "
  //    "`iteration_end` instead.";
  //}
  std::vector<double> temporary_beams;
  std::vector<float> temporary_features;
  LaserScannerFeatures lsf;

  for (int y = 0; y < map_to_be_labeled.rows; y++) {
    for (int x = 0; x < map_to_be_labeled.cols; x++) {
      if (map_to_be_labeled.at<unsigned char>(y, x) != 0) {
        //simulate the beams and features for every position and save it
        raycasting_.raycasting(map_to_be_labeled, cv::Point(x, y), temporary_beams);
        cv::Mat features;
        lsf.get_features(temporary_beams, angles_for_simulation_, cv::Point(x, y), features);
        temporary_features.resize(features.cols);

        for (int i = 0; i < features.cols; ++i)
          temporary_features[i] = features.at<float>(0, i);

        // convert temporary_features to DMatrix
        std::shared_ptr<DMatrix> dtest(DMatrix::Load(param_.test_path, ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(), param_.dsplit == 2));
        learner_->Predict(dtest, param_.pred_margin, &preds, param_.iteration_begin, param_.iteration_end);
        LOG(CONSOLE) << "Writing prediction to " << param_.name_pred;
        temporary_features.clear();
      }
    }
  }

  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(param_.name_pred.c_str(), "w"));
  dmlc::ostream os(fo.get());
  for (bst_float p : preds.ConstHostVector()) {
    os << std::setprecision(std::numeric_limits<bst_float>::max_digits10) << p << '\n';
  }
  // force flush before fo destruct.
  os.set_stream(nullptr);
  trained_ = true;
  std::cout << "Finished Labeling the map." << std::endl;
  return true;
}
