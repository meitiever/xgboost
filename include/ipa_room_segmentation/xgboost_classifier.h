#pragma once

#include <string>
#include <xgboost/c_api.h>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <Eigen/Eigen>
#include <iostream>

#include <ipa_room_segmentation/features.h>
#include <ipa_room_segmentation/raycasting.h>

class XgboostClassifier
{
public:
  typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> Matrix;
  template<typename M>
  static void vector2Matrix(M& m, const typename M::Scalar* vec, Eigen::Index const rows, Eigen::Index const cols)
  {
    m = Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(vec, rows, cols);
  }

  XgboostClassifier(std::string const& model_path, uint64_t nlabels = 1) :
    _modelPath(model_path),
    _nlabels(nlabels)
  {
    for (double angle = 0; angle < 360; angle++) {
      angles_for_simulation_.push_back(angle);
    }

    if (XGBoosterCreate(NULL, 0, &_booster) == 0 && XGBoosterLoadModel(_booster, _modelPath.c_str()) == 0) {
      //LOG HERE
      trained_ = true;
    }
    else {
      //LOG HERE
      _booster = NULL;
    }
  }

  //labeling-algorithm after the training
  bool segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
    double room_area_factor_lower_limit, double room_area_factor_upper_limit);

  virtual ~XgboostClassifier() {
    XGBoosterFree(_booster);
  }

private:
  int predict(Matrix const& features, Matrix& result)
  {
    DMatrixHandle X;
    const float* data = features.data();
    auto const nrow = features.rows();
    auto const ncol = features.cols();

    XGDMatrixCreateFromMat(data, nrow, ncol, NAN, &X);

    const float* out;
    uint64_t l;
    auto ret = XGBoosterPredict(_booster, X, 0, 0, 0, &l, &out);
    if (ret < 0) {
      // LOG HERE
      return -1;
    }

    XGDMatrixFree(X);

    if (l != nrow * _nlabels) {
      //LOG HERE
      return -1;
    }

    vector2Matrix(result, out, nrow, _nlabels);
    return 0;
  }

private:
  bool trained_; // variable that shows if the classifiers has already been trained
  std::string const _modelPath;
  BoosterHandle _booster;
  uint64_t const _nlabels;
  std::string model_path;
  std::vector<double> angles_for_simulation_; // angle-vector used to calculate the features for this algorithm
  LaserScannerRaycasting raycasting_;
};
